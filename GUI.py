import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
import windrose
from windrose import WindroseAxes
import time
from joblib import Parallel, delayed
import os
from scipy import optimize
import cvxopt
import quadprog
from scipy import stats
# import pyscipopt
from multiprocessing import Pool
import multiprocessing
from timeit import default_timer as timer
import torch


dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0") # Uncomment this to run on GPU

print(device)

torch.manual_seed(0)

start_all = time.time()

class SensorPlacementGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimal Sensor Placement via SBA algorithm")
        self.root.geometry("1200x800")
        
        self.reset_state()

        # Main layout frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(side="top", fill="both", expand=True)

        # Create input frame
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Title
        ttk.Label(self.input_frame, text="Optimization Hyperparameters", font=("Arial", 14)).pack(pady=10)

        # Input fields
        self.entries = {}
        params = [
            ("Number of Sources (N_s)", "50"),
            ("Number of Sensors (P)", "20"),
            ("Monte Carlo Samples (N)", "10"),
            ("Outer Iteration Epochs (K)", "100"),
            ("Inner Loop Steps (q)", "20"),
            ("Batch Size (B)", "10"),
            ("Sensor Noise (sigma_epsilon)", "0.01"),
            ("Learning Rate (lr_inner_initial)", "0.0005"),
            ("Learning Rate (lr_outer_initial)", "1e-6"),
            ("Height of Stacks", "2"),
            ("Eddy Diffusion Coefficient", "0.4"),
            ("Upper Bound of Wind Speed", "1"),
            ("Lower Bound of Wind Speed", "2"),
            ("Upper Bound of Wind Direction", "225"),
            ("Lower Bound of Wind Direction", "315"),
            ("Lambda 1", "0.01"),
            ("Lambda 2", "0.01")
        ]

        for param, default in params:
            frame = ttk.Frame(self.input_frame)
            frame.pack(pady=5, fill="x")
            ttk.Label(frame, text=param, width=30).pack(side="left", padx=5)
            entry = ttk.Entry(frame)
            entry.insert(0, default)
            entry.pack(side="left", expand=True, fill="x")
            self.entries[param] = entry

        # Buttons
        btn_frame = ttk.Frame(self.input_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Run Optimization", command=self.submit).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Stop", command=self.stop).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Restart", command=self.restart).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Refresh", command=self.refresh).pack(side="left", padx=5)

        # Create figure area
        self.figure_frame = ttk.Frame(self.main_frame)
        self.figure_frame.pack(side="right", fill="both", expand=True)
        self.fig, self.ax1 = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.figure_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Status label and progress bar
        self.status_label = ttk.Label(self.figure_frame, text="")
        self.status_label.pack(pady=5)
        self.progress = ttk.Progressbar(self.figure_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=5)

        # # Global footer bar
        # self.footer = ttk.Label(self.root, text="© opondofnvdnv", anchor='center', font=("Arial", 9))
        # self.footer.pack(side="bottom", fill="x", pady=2)
        # Global footer bar
        self.footer = ttk.Label(self.root, text="Demo for paper 'Optimal Sensor Allocation for Emission Source Detection with Linear Dispersion Processes'", anchor='center', font=("Arial", 9))
        self.footer.pack(side="bottom", fill="x", pady=2)
    
    def reset_state(self):
        self.stop_requested = False
        self.trajectories_x = []
        self.trajectories_y = []
        self.sensor_x = None
        self.sensor_y = None
        self.sensor_markers = None
        self.trajectory_lines = []
        
    def submit(self):
        try:
            self.stop_requested = False
            self.reset_state()
            config = {
                "N_s": int(self.entries["Number of Sources (N_s)"].get()),
                "P": int(self.entries["Number of Sensors (P)"].get()),
                "N": int(self.entries["Monte Carlo Samples (N)"].get()),
                "K": int(self.entries["Outer Iteration Epochs (K)"].get()),
                "q": int(self.entries["Inner Loop Steps (q)"].get()),
                "B": int(self.entries["Batch Size (B)"].get()),
                "sigma_epsilon": float(self.entries["Sensor Noise (sigma_epsilon)"].get()),
                "lr_inner_initial": float(self.entries["Learning Rate (lr_inner_initial)"].get()),
                "lr_outer_initial": float(self.entries["Learning Rate (lr_outer_initial)"].get()),
                "H": float(self.entries["Height of Stacks"].get()),
                "coeff": float(self.entries["Eddy Diffusion Coefficient"].get()),
                "ws_u": float(self.entries["Upper Bound of Wind Speed"].get()),
                "ws_l": float(self.entries["Lower Bound of Wind Speed"].get()),
                "wd_u": float(self.entries["Upper Bound of Wind Direction"].get()),
                "wd_l": float(self.entries["Lower Bound of Wind Direction"].get()),
                "l1": float(self.entries["Lambda 1"].get()),
                "l2": float(self.entries["Lambda 2"].get()),
            }

            self.progress["value"] = 0
            self.progress["maximum"] = config["K"]
            threading.Thread(target=self.run_dynamic_simulation, args=(config,), daemon=True).start()

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")

    def stop(self):
        self.stop_requested = True
        self.status_label.config(text="Stopping optimization...")

    def restart(self):
        self.stop()
        self.refresh()
        self.submit()

    def refresh(self):
        self.stop_requested = False
        self.reset_state()
        self.ax1.clear()
        self.canvas.draw()
        self.status_label.config(text="")
        self.progress["value"] = 0
    
    def run_dynamic_simulation(self, config):
        self.ax1.clear()

        # User-defined Inputs:
        # source_location_x = [-15., -10., -9., -5., 5., 5., 8., 10., 15., 20.]  # X axis
        # source_location_y = [17., -5., 22., 10., 18., 0., -10., 19., -10, 5.]  # Y axis
        import torch
        N_of_source = config["N_s"]
        source_location_x = 40*torch.rand(N_of_source)-20  # X axis
        source_location_y = 40*torch.rand(N_of_source)-20  # Y axis

        N_sources = len(source_location_x)
        # Define the mean and variance of emission rates for truncated Gaussian distribution
        # mean = [8., 10., 9., 8., 10., 9., 8., 10., 9., 10.]  # the mean of emission rates for the above sources
        mean = 2*torch.rand(N_of_source) + 8  # the mean of emission rates for the above sources
        sigma_pior_abs = 20
        cov = sigma_pior_abs * torch.eye(N_sources)  # the covariance of these emission rates
        # Define the height of stacks and the eddy diffusion coefficient
        H = config["H"] * torch.ones(N_sources)  # the height of stacks
        K = config["coeff"] * torch.ones(N_sources)  # the eddy diffusion coefficient, which is simplified
        # Define the number of Monte Carlo samples
        N_samples = config["N"]
        N_samples_large = 1
        # Define the number of sensors
        N_sensors = config["P"]
        # Define the sensor noise level -> the standard deviation
        sigma_epsilon = config["sigma_epsilon"]
        # the wind condition
        ws_lower = config["ws_u"]  # the upper bound of wind speed
        ws_upper = config["ws_l"]  # the lower bound of wind speed
        wd_lower = config["wd_u"]  # the upper bound of wind direction
        wd_upper = config["wd_l"]  # the lower bound of wind direction
        # lambda
        lambda_1 = config["l1"]
        lambda_2 = config["l2"]

        # Here are some advanced setting parameters: (You can change those parameters for better performance)
        Num_iteration_k = config["K"]  # for the maximum iteration steps for the outer iteration
        Num_SGD_BatchSize = config["B"]  # for the batch size in SGD
        Num_Backtracking = 0  # for adapting the outer iteration step size (or called learning rate)
        lr_outer_initial = config["lr_outer_initial"]  # for the initial learning rate of the backtracking
        tol_G = 1e-10  # the tolerance for step length (i.e., the absolution value of the gradients of the outer iteration)
        q = config["q"]  # the maximum iteration number of inner loop
        lr_inner_initial = config["lr_inner_initial"]
        
        def objective(v):
            num_sensor = N_sensors
            x_all = v[0:num_sensor]
            y_all = v[num_sensor:(2 * num_sensor)]
            source_x = source_location_x
            source_y = source_location_y
            num_source = len(source_x)
            # Define the mean and variance of emission rates
            sigma_pr = sigma_pior_abs
            # number of monte carlo samplings for A-optimal design
            N = 5
            temp = 0
            for nk in range(N):
                ws = torch.random.uniform(ws_lower, ws_upper, 1)  # the wind speed distribution
                wd = torch.random.uniform(wd_lower, wd_upper, 1)  # the wind angle distribution
                w_x = torch.cos(wd / 180. * torch.pi)  # the x part of unit wind vector
                w_y = torch.sin(wd / 180. * torch.pi)  # the y part of unit wind vector
                u = torch.abs(ws)  # the wind speed
                # the F operator
                A_all = torch.zeros((num_sensor, num_source))
                x_new = torch.sqrt(
                    ((1. - w_x ** 2.) * (x_all.reshape(1,-1).T - source_x.reshape(1,-1)) - w_x * w_y * (y_all.reshape(1,-1).T - source_y.reshape(1,-1))) ** 2. + (
                            -w_x * w_y * (x_all.reshape(1,-1).T - source_x.reshape(1,-1)) + (1. - w_y ** 2.) * (y_all.reshape(1,-1).T - source_y.reshape(1,-1))) ** 2.)
                y_new = w_x * (x_all.reshape(1,-1).T - source_x.reshape(1,-1)) + w_y * (y_all.reshape(1,-1).T - source_y.reshape(1,-1))
                y_original = y_new.detach().clone()
                y_new[y_original<=0] = 1
                A_all = 1 / (2. * torch.pi * K[None,:] * y_new) * torch.exp(
                                -u * (x_new ** 2. + H[None,:] ** 2.) / (4. * K[None,:] * y_new))
                A_all[y_original<=0] = 0
                # the posterior covariance
                cov_post = torch.linalg.inv(1 / sigma_epsilon ** 2 * A_all.T @ A_all + 1 / sigma_pr ** 2 * torch.eye(num_source))
                temp += torch.linalg.norm(1 / sigma_pr * cov_post) + torch.linalg.norm(1 / sigma_epsilon * cov_post @ A_all.T)
            return temp / N


        def TwoDimenGauPlumeM_AllSource_Reading_VEC(x_tem, y_tem, source_x, source_y, w_x, w_y, u, q_tem, K_tem, H_tem, noise):
            x_new = torch.sqrt(((1. - w_x[:,None,None] ** 2.) * (x_tem.reshape(1,-1).T - source_x.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y.reshape(1,-1))) ** 2. + (
                    -w_x[:,None,None] * w_y[:,None,None] * (x_tem.reshape(1,-1).T - source_x.reshape(1,-1)) + (1. - w_y[:,None,None] ** 2.) * (y_tem.reshape(1,-1).T - source_y.reshape(1,-1))) ** 2.)
            y_new = w_x[:,None,None] * (x_tem.reshape(1,-1).T - source_x.reshape(1,-1)) + w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y.reshape(1,-1))
            y_new_origi = y_new.detach().clone()
            y_new[y_new_origi<=0] = 1
            Pi = q_tem[:,None,:] / (2. * torch.pi * K_tem[None,None,:] * y_new) * torch.exp(-u[:,None,None] * (x_new ** 2. + H_tem[None,None,:] ** 2.) / (4. * K_tem[None,None,:] * y_new))
            Pi[y_new_origi<=0] = 0
            out = Pi @ torch.ones(len(source_x)) + noise
            return out


        def GradientInnerNew(x_all, y_all, source_x, source_y, w_x, w_y, u, K_tem, H_tem, Phi_tem, sigma_e, lambda_1_tem,
                            lambda_2_tem):
            num_sensor = len(x_all)
            num_source = len(source_x)
            num_samples = len(w_x)
            A_all = torch.zeros((num_samples,num_sensor, num_source))
            x_new = torch.sqrt(
                ((1. - w_x[:,None,None] ** 2.) * (x_all.reshape(1,-1).T - source_x.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_all.reshape(1,-1).T - source_y.reshape(1,-1))) ** 2. + (
                        -w_x[:,None,None] * w_y[:,None,None] * (x_all.reshape(1,-1).T - source_x.reshape(1,-1)) + (1. - w_y[:,None,None] ** 2.) * (y_all.reshape(1,-1).T - source_y.reshape(1,-1))) ** 2.)
            y_new = w_x[:,None,None] * (x_all.reshape(1,-1).T - source_x.reshape(1,-1)) + w_y[:,None,None] * (y_all.reshape(1,-1).T - source_y.reshape(1,-1))
            y_new_orig = y_new.detach().clone()
            y_new[y_new_orig<=0]=1
            A_all = 1 / (2. * torch.pi * K_tem[None,None,:] * y_new) * torch.exp(
                    -u[:,None,None] * (x_new ** 2. + H_tem[None,None,:] ** 2.) / (4. * K_tem[None,None,:] * y_new))
            A_all[y_new_orig<=0]=0 

            C_coef = torch.zeros((num_samples,num_source, num_source))
            D_coef_T = torch.zeros((num_source,num_samples))
            # print(Phi_tem.shape)
            start = timer()
            for jk in range(num_samples):
                C_coef[jk] = 1. / (sigma_e ** 2) * (A_all[jk].T @ A_all[jk]) + lambda_1_tem * torch.eye(num_source)
                if num_sensor == 1:
                    D_coef_T[:,jk] = lambda_2_tem * torch.ones((num_source, 1)) - 1. / (sigma_e ** 2) * A_all[jk].T * Phi_tem[:,jk]
                else:
                    tem = torch.zeros((num_sensor, 1))
                    tem[:, 0] = Phi_tem[:,jk]
                    D_coef_T[:,jk:(jk+1)] = lambda_2_tem * torch.ones((num_source, 1)) - 1. / (sigma_e ** 2) * A_all[jk].T @ tem
            end = timer()
            # print('time C:', end-start)
            return [C_coef, D_coef_T]


        def Gradient_AA_x_VEC(x_tem, y_tem, source_x1, source_y1, source_x2, source_y2, w_x, w_y, u, K1, H1, K2, H2):
            r_per1 = torch.sqrt(((1. - w_x[:,None,None] ** 2.) * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) ** 2. + (
                    -w_x[:,None,None] * w_y[:,None,None] * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) + (1. - w_y[:,None,None] ** 2.) * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) ** 2.)
            r_para1 = w_x[:,None,None] * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) + w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))
            r_per2 = torch.sqrt(((1. - w_x[:,None,None] ** 2.) * (x_tem.reshape(1,-1).T - source_x2.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y2.reshape(1,-1))) ** 2. + (
                    -w_x[:,None,None] * w_y[:,None,None] * (x_tem.reshape(1,-1).T - source_x2.reshape(1,-1)) + (1. - w_y[:,None,None] ** 2.) * (y_tem.reshape(1,-1).T - source_y2.reshape(1,-1))) ** 2.)
            r_para2 = w_x[:,None,None] * (x_tem.reshape(1,-1).T - source_x2.reshape(1,-1)) + w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y2.reshape(1,-1))
            # print(r_para1)

            r_para1_orig = r_para1.detach().clone()
            r_para2_orig = r_para2.detach().clone()
            r_para1[(r_para1_orig <= 0) | (r_para2_orig <= 0)]=1
            r_para2[(r_para1_orig <= 0) | (r_para2_orig <= 0)]=1
            start_1 = timer()
            G_AA_x = -(1 / (4 * torch.pi ** 2 * K1[None,None,:] * K2[None,None,:] * r_para1 * r_para2) ** 2) * 4 * torch.pi ** 2 * K1[None,None,:] * K2[None,None,:] * (
                    w_x[:,None,None] * (r_para1 + r_para2)) * torch.exp(
                -u[:,None,None] * (r_per1 ** 2 + H1[None,None,:] ** 2) / (4 * K1[None,None,:] * r_para1) - u[:,None,None] * (r_per2 ** 2 + H2[None,None,:] ** 2) / (4 * K2[None,None,:] * r_para2)) \
                        + 1 / (4 * torch.pi ** 2 * K1[None,None,:] * K2[None,None,:] * r_para1 * r_para2) * torch.exp(
                -u[:,None,None] * (r_per1 ** 2 + H1[None,None,:] ** 2) / (4 * K1[None,None,:] * r_para1) - u[:,None,None] * (r_per2 ** 2 + H2[None,None,:] ** 2) / (4 * K2[None,None,:] * r_para2)) \
                        * ((-u[:,None,None] * (2 * ((1 - w_x[:,None,None] ** 2) * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) * (
                    1 - w_x[:,None,None] ** 2) + 2 * (-w_x[:,None,None] * w_y[:,None,None] * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) + (1 - w_y[:,None,None] ** 2) * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) * (
                                        -w_x[:,None,None] * w_y[:,None,None])) * 4 * K1[None,None,:] * r_para1 + u[:,None,None] * (r_per1 ** 2 + H1[None,None,:] ** 2) * 4 * K1[None,None,:] * w_x[:,None,None]) / (
                                4 * K1[None,None,:] * r_para1) ** 2 + (-u[:,None,None] * (
                    2 * ((1 - w_x[:,None,None] ** 2) * (x_tem.reshape(1,-1).T - source_x2.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y2.reshape(1,-1))) * (
                    1 - w_x[:,None,None] ** 2) + 2 * (
                            -w_x[:,None,None] * w_y[:,None,None] * (x_tem.reshape(1,-1).T - source_x2.reshape(1,-1)) + (1 - w_y[:,None,None] ** 2) * (y_tem.reshape(1,-1).T - source_y2.reshape(1,-1))) * (
                            -w_x[:,None,None] * w_y[:,None,None])) * 4 * K2[None,None,:] * r_para2 + u[:,None,None] * (r_per2 ** 2 + H2[None,None,:] ** 2) * 4 * K2[None,None,:] * w_x[:,None,None]) / (
                                4 * K2[None,None,:] * r_para2) ** 2)
            G_AA_x[(r_para1_orig <= 0) | (r_para2_orig <= 0)]=0.
            end_1 = timer()

            return G_AA_x


        def Gradient_AA_y_VEC(x_tem, y_tem, source_x1, source_y1, source_x2, source_y2, w_x, w_y, u, K1, H1, K2, H2):
            r_per1 = torch.sqrt(((1. - w_x[:,None,None] ** 2.) * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) ** 2. + (
                    -w_x[:,None,None] * w_y[:,None,None] * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) + (1. - w_y[:,None,None] ** 2.) * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) ** 2.)
            r_para1 = w_x[:,None,None] * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) + w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))
            r_per2 = torch.sqrt(((1. - w_x[:,None,None] ** 2.) * (x_tem.reshape(1,-1).T - source_x2.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y2.reshape(1,-1))) ** 2. + (
                    -w_x[:,None,None] * w_y[:,None,None] * (x_tem.reshape(1,-1).T - source_x2.reshape(1,-1)) + (1. - w_y[:,None,None] ** 2.) * (y_tem.reshape(1,-1).T - source_y2.reshape(1,-1))) ** 2.)
            r_para2 = w_x[:,None,None] * (x_tem.reshape(1,-1).T - source_x2.reshape(1,-1)) + w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y2.reshape(1,-1))

            r_para1_orig = r_para1.detach().clone()
            r_para2_orig = r_para2.detach().clone()
            r_para1[(r_para1_orig <= 0) | (r_para2_orig <= 0)]=1
            r_para2[(r_para1_orig <= 0) | (r_para2_orig <= 0)]=1
            G_AA_y = -(1 / (4 * torch.pi ** 2 * K1[None,None,:] * K2[None,None,:] * r_para1 * r_para2) ** 2) * 4 * torch.pi ** 2 * K1[None,None,:] * K2[None,None,:] * (
                        w_y[:,None,None] * (r_para1 + r_para2)) * torch.exp(
                    -u[:,None,None] * (r_per1 ** 2 + H1[None,None,:] ** 2) / (4 * K1[None,None,:] * r_para1) - u[:,None,None] * (r_per2 ** 2 + H2[None,None,:] ** 2) / (4 * K2[None,None,:] * r_para2)) \
                        + 1 / (4 * torch.pi ** 2 * K1[None,None,:] * K2[None,None,:] * r_para1 * r_para2) * torch.exp(
                    -u[:,None,None] * (r_per1 ** 2 + H1[None,None,:] ** 2) / (4 * K1[None,None,:] * r_para1) - u[:,None,None] * (r_per2 ** 2 + H2[None,None,:] ** 2) / (4 * K2[None,None,:] * r_para2)) \
                        * ((-u[:,None,None] * (
                        2 * ((1 - w_x[:,None,None] ** 2) * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) * (-w_x[:,None,None] * w_y[:,None,None]) + 2 * (
                        -w_x[:,None,None] * w_y[:,None,None] * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) + (1 - w_y[:,None,None] ** 2) * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) * (
                                1 - w_y[:,None,None] ** 2)) * 4 * K1[None,None,:] * r_para1 + u[:,None,None] * (r_per1 ** 2 + H1[None,None,:] ** 2) * 4 * K1[None,None,:] * w_y[:,None,None]) / (
                                    4 * K1[None,None,:] * r_para1) ** 2 + (-u[:,None,None] * (
                        2 * ((1 - w_x[:,None,None] ** 2) * (x_tem.reshape(1,-1).T - source_x2.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y2.reshape(1,-1))) * (-w_x[:,None,None] * w_y[:,None,None]) + 2 * (
                        -w_x[:,None,None] * w_y[:,None,None] * (x_tem.reshape(1,-1).T - source_x2.reshape(1,-1)) + (1 - w_y[:,None,None] ** 2) * (y_tem.reshape(1,-1).T - source_y2.reshape(1,-1))) * (
                                1 - w_y[:,None,None] ** 2)) * 4 * K2[None,None,:] * r_para2 + u[:,None,None] * (r_per2 ** 2 + H2[None,None,:] ** 2) * 4 * K2[None,None,:] * w_y[:,None,None]) / (
                                    4 * K2[None,None,:] * r_para2) ** 2)
            G_AA_y[(r_para1_orig <= 0)|(r_para2_orig <= 0)]=0

            return G_AA_y


        def Gradient_A_x_VEC(x_tem, y_tem, source_x1, source_y1, w_x, w_y, u, K_tem, H_tem):
            r_per1 = torch.sqrt(((1. - w_x[:,None,None] ** 2.) * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) ** 2. + (
                    -w_x[:,None,None] * w_y[:,None,None] * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) + (1. - w_y[:,None,None] ** 2.) * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) ** 2.)
            r_para1 = w_x[:,None,None] * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) + w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))

            r_para1_orig = r_para1.detach().clone()
            r_para1[r_para1_orig <= 0]=1

            G_A_x = -(1 / (2 * torch.pi * K_tem[None,None,:] * r_para1) ** 2) * 2 * torch.pi * K_tem[None,None,:] * w_x[:,None,None] * torch.exp(
                    -u[:,None,None] * (r_per1 ** 2 + H_tem[None,None,:] ** 2) / (4 * K_tem[None,None,:] * r_para1)) \
                        + 1 / (2 * torch.pi * K_tem[None,None,:] * r_para1) * torch.exp(-u[:,None,None] * (r_per1 ** 2 + H_tem[None,None,:] ** 2) / (4 * K_tem[None,None,:] * r_para1)) \
                        * ((-u[:,None,None] * (2 * ((1 - w_x[:,None,None] ** 2) * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) * (
                        1 - w_x[:,None,None] ** 2) + 2 * (-w_x[:,None,None] * w_y[:,None,None] * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) + (1 - w_y[:,None,None] ** 2) * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) * (
                                        -w_x[:,None,None] * w_y[:,None,None])) * 4 * K_tem[None,None,:] * r_para1 + u[:,None,None] * (
                                    r_per1 ** 2 + H_tem[None,None,:] ** 2) * 4 * K_tem[None,None,:] * w_x[:,None,None]) / (4 * K_tem[None,None,:] * r_para1) ** 2)
            G_A_x[r_para1_orig <= 0]=0

            return G_A_x


        def Gradient_A_y_VEC(x_tem, y_tem, source_x1, source_y1, w_x, w_y, u, K_tem, H_tem):
            r_per1 = torch.sqrt(((1. - w_x[:,None,None] ** 2.) * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) ** 2. + (
                    -w_x[:,None,None] * w_y[:,None,None] * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) + (1. - w_y[:,None,None] ** 2.) * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) ** 2.)
            r_para1 = w_x[:,None,None] * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) + w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))

            r_para1_orig = r_para1.detach().clone()
            r_para1[r_para1_orig <= 0]=1
            G_A_y = -(1 / (2 * torch.pi * K_tem[None,None,:] * r_para1) ** 2) * 2 * torch.pi * K_tem[None,None,:] * w_y[:,None,None] * torch.exp(
                    -u[:,None,None] * (r_per1 ** 2 + H_tem[None,None,:] ** 2) / (4 * K_tem[None,None,:] * r_para1)) \
                        + 1 / (2 * torch.pi * K_tem[None,None,:] * r_para1) * torch.exp(-u[:,None,None] * (r_per1 ** 2 + H_tem[None,None,:] ** 2) / (4 * K_tem[None,None,:] * r_para1)) \
                        * ((-u[:,None,None] * (
                        2 * ((1 - w_x[:,None,None] ** 2) * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) * (-w_x[:,None,None] * w_y[:,None,None]) + 2 * (
                        -w_x[:,None,None] * w_y[:,None,None] * (x_tem.reshape(1,-1).T - source_x1.reshape(1,-1)) + (1 - w_y[:,None,None] ** 2) * (y_tem.reshape(1,-1).T - source_y1.reshape(1,-1))) * (
                                1 - w_y[:,None,None] ** 2)) * 4 * K_tem[None,None,:] * r_para1 + u[:,None,None] * (
                                    r_per1 ** 2 + H_tem[None,None,:] ** 2) * 4 * K_tem[None,None,:] * w_y[:,None,None]) / (4 * K_tem[None,None,:] * r_para1) ** 2)
            G_A_y[r_para1_orig <= 0]=0

            return G_A_y


        def is_pos_def(x):
            return torch.all(torch.linalg.eigvals(x) > 0)

        def GradientOuterNew(x_all, y_all, source_x, source_y, w_x, w_y, u, K_tem, H_tem, Phi_tem, sigma_e, lambda_1_tem,
                            theta_curr, multiplier_esti):
            num_sensor = len(x_all)
            num_source = len(source_x)
            num_samples = len(w_x)
            A_all = torch.zeros((num_samples, num_sensor, num_source))

            x_new = torch.sqrt(
                ((1. - w_x[:,None,None] ** 2.) * (x_all.reshape(1,-1).T - source_x.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_all.reshape(1,-1).T - source_y.reshape(1,-1))) ** 2. + (
                        -w_x[:,None,None] * w_y[:,None,None] * (x_all.reshape(1,-1).T - source_x.reshape(1,-1)) + (1. - w_y[:,None,None] ** 2.) * (y_all.reshape(1,-1).T - source_y.reshape(1,-1))) ** 2.)
            y_new = w_x[:,None,None] * (x_all.reshape(1,-1).T - source_x.reshape(1,-1)) + w_y[:,None,None] * (y_all.reshape(1,-1).T - source_y.reshape(1,-1))
            y_new_orig = y_new.detach().clone()
            y_new[y_new_orig<=0]=1
            A_all = 1 / (2. * torch.pi * K_tem[None,None,:] * y_new) * torch.exp(
                    -u[:,None,None] * (x_new ** 2. + H_tem[None,None,:] ** 2.) / (4. * K_tem[None,None,:] * y_new))
            A_all[y_new_orig<=0]=0 

            C_coef = torch.zeros((num_samples,num_source, num_source))
            for mk in range(num_samples):
                C_coef[mk] = 1. / (sigma_e ** 2) * (A_all[mk].T @ A_all[mk]) + lambda_1_tem * torch.eye(num_source)
                # if is_pos_def(C_coef[mk])==False:  # check if the coefficient matrix is 
                #     print('Not positive definite!!!')

            arr = torch.arange(0,num_source)
            arr1 = torch.repeat_interleave(arr,num_source)
            arr2 = arr.repeat(num_source)
            start = timer()
            # print(arr2.view(num_source, num_source))

            Gradient_AA_x_cal=Gradient_AA_x_VEC(x_all, y_all, source_x[arr1], source_y[arr1],source_x[arr2], source_y[arr2],w_x, w_y, u, K_tem[arr1], H_tem[arr1], K_tem[arr2], H_tem[arr2])
            end22 = timer()
            Gradient_AA_y_cal=Gradient_AA_y_VEC(x_all, y_all, source_x[arr1], source_y[arr1],source_x[arr2], source_y[arr2],w_x, w_y, u, K_tem[arr1], H_tem[arr1], K_tem[arr2], H_tem[arr2])
            end221 = timer()
            Gradient_A_x_cal=Gradient_A_x_VEC(x_all, y_all, source_x, source_y, w_x, w_y, u, K_tem, H_tem)
            Gradient_A_y_cal=Gradient_A_y_VEC(x_all, y_all, source_x, source_y, w_x, w_y, u, K_tem, H_tem)

            end223 = timer()

            # G_x = torch.zeros((num_samples, num_sensor, num_source))
            # G_y = torch.zeros((num_samples, num_sensor, num_source))
            # for mk in range(num_sensor):
            #     for i in range(num_source):
            #         for j in range(num_source):
            #             G_x[:, mk, i] += 1. / (sigma_e ** 2) * Gradient_AA_x_cal[:,mk,(i*num_source+j)] * theta_curr[:,j]
            #             G_y[:, mk, i] += 1. / (sigma_e ** 2) * Gradient_AA_y_cal[:,mk,(i*num_source+j)] * theta_curr[:,j]
            #         G_x[:, mk, i] -= 1. / (sigma_e ** 2) * Gradient_A_x_cal[:,mk,i] * Phi_tem[mk]
            #         G_y[:, mk, i] -= 1. / (sigma_e ** 2) * Gradient_A_y_cal[:,mk,i] * Phi_tem[mk]

            # G_x_1 = G_x

            scaling_factor = 1. / (sigma_e ** 2)
            # Reshape theta_curr to match the dimensions required for broadcasting
            theta_curr_expanded = theta_curr.unsqueeze(1).unsqueeze(2)  # shape: [num_samples, 1, num_source, 1]
            G_x = scaling_factor * (Gradient_AA_x_cal.view(num_samples, num_sensor, num_source, num_source) * theta_curr_expanded).sum(dim=3)
            G_y = scaling_factor * (Gradient_AA_y_cal.view(num_samples, num_sensor, num_source, num_source) * theta_curr_expanded).sum(dim=3)
            Phi_T = Phi_tem.T
            G_x -= scaling_factor * (Phi_T[:,:,None] * Gradient_A_x_cal)  # shape: [num_samples, num_sensor, num_source]
            G_y -= scaling_factor * (Phi_T[:,:,None] * Gradient_A_y_cal)

            # print(G_x_1[0,0,:])
            # print(G_x[0,0,:])

            end33 = timer()
            # print('time inner 11:',end22 - start)
            # print('time inner 22:',end33 - end22)
            # print('time inner 221:',end221 - end22)
            # print('time inner 222:',end223 - end221)

            time2 = timer()
            # print('Chao2: ',G_x[0])
            # print('G_x:', G_x)
            # note that we used the linear regression package as the linear solver
            # it is better to develop our own linear solver, which should be efficient
            coef_x = torch.zeros((num_samples, num_sensor, num_source))
            coef_y = torch.zeros((num_samples, num_sensor, num_source))
            # G_x = torch.zeros((num_samples, num_sensor, num_source))
            # G_y = torch.zeros((num_samples, num_sensor, num_source))
            # C_coef_inv =  [torch.zeros((num_source, num_source)) for ik in range(num_samples)]
            C_coef_inv = [torch.linalg.pinv(C_coef[ik].detach().clone()) for ik in range(num_samples)]
            for mk in range(num_samples):
                coef_matrix = torch.eye(num_source)
                index_active = torch.where(multiplier_esti[mk, :] > 0)[0]
                coef_matrix_active = coef_matrix[index_active, :]
                for i in range(num_sensor):
                    G_x_T = torch.zeros(num_source)
                    G_x_T[:] = G_x[mk, i, :].detach().clone()
                    if torch.any(torch.isnan(G_x_T)):
                        coef_x[mk, i, :] = torch.zeros(num_source)
                    else:
                        # reg1 = torch.linalg.solve(-torch.array(C_coef), G_x_T)
                        reg1 = -C_coef_inv[mk] @ (
                                    G_x_T - coef_matrix_active.reshape(num_source, index_active.shape[0]) @ torch.linalg.pinv(
                                coef_matrix_active @ C_coef_inv[mk] @ torch.reshape(coef_matrix_active, (num_source, index_active.shape[0]))) @ coef_matrix_active @ C_coef_inv[mk] @ G_x_T)
                        coef_x[mk, i, :] = reg1 
                    G_y_T = torch.zeros(num_source)
                    G_y_T[:] = G_y[mk, i, :].detach().clone()
                    if torch.any(torch.isnan(G_y_T)):
                        coef_y[mk, i, :] = torch.zeros(num_source)
                    else:
                        # reg2 = torch.linalg.solve(-torch.array(C_coef), G_y_T)
                        reg2 = -C_coef_inv[mk] @ (
                                    G_y_T - coef_matrix_active.reshape(num_source, index_active.shape[0]) @ torch.linalg.pinv(
                                coef_matrix_active @ C_coef_inv[mk] @ torch.reshape(coef_matrix_active, (num_source, index_active.shape[0]))) @ coef_matrix_active @ C_coef_inv[mk] @ G_y_T)
                        coef_y[mk, i, :] = reg2
                    if torch.sum(coef_x[mk, i, :]) > 1000000000. or torch.sum(
                            coef_y[mk, i, :]) > 1000000000.:  # to avoid the abnormal results of linear solver
                        coef_x[mk, i, :] = torch.zeros(num_source)
                        coef_y[mk, i, :] = torch.zeros(num_source)
            time3 = timer()
            # print('time inner 12:',time3 - time2)
            return [coef_x, coef_y]


        def Inner_loop(q_tem, C, D_T, N_sources_tem, para_lr_inner, lr_inner):
            theta_esti_all_tem = torch.zeros((q_tem, N_sources_tem))
            multiplier_esti_all_tem = torch.zeros((q_tem, N_sources_tem))
            Gradient_theta_All = torch.zeros((q_tem, N_sources_tem))
            Gradient_multiplier_All = torch.zeros((q_tem, N_sources_tem))
            coef_matrix = torch.eye(N_sources_tem)
            identity_matrix = torch.eye(N_sources_tem)
            gamma_r = 1
            # Set the initial guess of theta and Lagrangian multiplier
            theta_esti_all_tem[0, :] = torch.zeros((1, N_sources_tem))  # Here we start from zeros as the initial values
            multiplier_esti_all_tem[0, :] = torch.zeros((1, N_sources_tem))  # Here we start from zeros as the initial values
            # print('trytry:',q_tem)
            for j in range(q_tem-1):
                # print('trytry:',j)
                Gradient_theta_All[j, :] = torch.matmul(C, theta_esti_all_tem[j, :]) + D_T.T + \
                                        sum(max(gamma_r*(-coef_matrix[mk, :] @ theta_esti_all_tem[j, :]) + multiplier_esti_all_tem[j, mk], 0)
                                            *-coef_matrix[mk, :] for mk in range(N_sources_tem))
                Gradient_multiplier_All[j, :] = sum(1/gamma_r*(max(gamma_r*(-coef_matrix[mk, :] @ theta_esti_all_tem[j, :]) + \
                                            multiplier_esti_all_tem[j, mk], 0) - multiplier_esti_all_tem[j, mk])*identity_matrix[mk, :] for mk in range(N_sources_tem))
                if j > para_lr_inner:
                    theta_esti_all_tem[j+1, :] = theta_esti_all_tem[j, :] - lr_inner / torch.sqrt(j + 1) * Gradient_theta_All[j, :]
                    multiplier_esti_all_tem[j + 1, :] = multiplier_esti_all_tem[j, :] + lr_inner / torch.sqrt(
                        j + 1) * Gradient_multiplier_All[j, :]
                else:
                    theta_esti_all_tem[j+1, :] = theta_esti_all_tem[j, :] - lr_inner * Gradient_theta_All[j, :]
                    multiplier_esti_all_tem[j + 1, :] = multiplier_esti_all_tem[j, :] + lr_inner * Gradient_multiplier_All[j, :]
                for mk in range(N_sources_tem):
                    # projected to be non-negative
                    theta_esti_all_tem[j + 1, mk] = max(0, theta_esti_all_tem[j + 1, mk])
                    multiplier_esti_all_tem[j+1, mk] = max(0, multiplier_esti_all_tem[j+1, mk])
            return theta_esti_all_tem[q_tem-1, :], multiplier_esti_all_tem[q_tem-1, :]


        # def cvxopt_solve_qp(P, q_tem, G, h, A=None, b=None):
        #     P = .5 * (P + P.T)  # make sure P is symmetric
        #     args = [cvxopt.matrix(P), cvxopt.matrix(q_tem)]
        #     args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        #     if A is not None:
        #         args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
        #     sol = cvxopt.solvers.qp(*args, options={'show_progress': False})
        #     if 'optimal' not in sol['status']:
        #         return None
        #     return torch.array(sol['x']).reshape((P.shape[1],))


        # def quadprog_solve_qp(P, q_tem, G, h, A=None, b=None):
        #     qp_G = .5 * (P + P.T)  # make sure P is symmetric
        #     qp_a = -q_tem
        #     if A is not None:
        #         qp_C = -torch.vstack([A, G]).T
        #         qp_b = -torch.hstack([b, h])
        #         meq = A.shape[0]
        #     else:  # no equality constraint
        #         qp_C = -G.T
        #         qp_b = -h
        #         meq = 0
        #     return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


        def Update_Inner_OuterStep(x_sensor_tem, y_sensor_tem, source_location_x_tem, source_location_y_tem, Wr_x_tem, Wr_y_tem,
                                w_speed_tem,
                                K_tem, H_tem, Phi_input, sigma_epsilon_tem, lambda_1_tem, lambda_2_tem, theta_true_tem,
                                N_sources_tem, q_tem, lr_inner_temp,kk):
            num_sensors = len(x_sensor_tem)
            num_samples = len(Wr_x_tem)
            Phi_tem = Phi_input
            theta_esti_all_tem = torch.zeros((num_samples, 1, N_sources_tem))
            theta_esti_all_tem_true = torch.zeros((num_samples, 1, N_sources_tem))
            multiplier_esti_all = torch.zeros((num_samples, 1, N_sources_tem))
            theta_error_all = torch.zeros((num_samples, 1, N_sources_tem))
            theta_error_all_QP = torch.zeros((num_samples, 1, N_sources_tem))
            start = timer()
            [C, D_T] = GradientInnerNew(x_sensor_tem, y_sensor_tem, source_location_x_tem, source_location_y_tem, Wr_x_tem,
                                        Wr_y_tem, w_speed_tem,
                                        K_tem, H_tem, Phi_tem, sigma_epsilon_tem, lambda_1_tem, lambda_2_tem)
            # # Set the initial guess of theta
            # theta_esti_all_tem[0, :] = torch.zeros((1, N_sources_tem))  # Here we start from zeros
            end = timer()
            para_lr_inner = q_tem  # now we ignore this parameter by setting a large number
            # call the inner loop
            for mk in range(num_samples):
                theta_esti_all_tem[mk], multiplier_esti_all[mk] = Inner_loop(q_tem, C[mk], D_T[:,mk], N_sources_tem, para_lr_inner, lr_inner_temp)

            # print('test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            
            # print('time00: ',end-start)
            # for mk in range(num_samples):
            #     theta_esti_all_tem_true[mk] = quadprog_solve_qp(C[mk], D_T[:,mk].reshape((N_sources_tem,)), -1. * torch.eye(N_sources_tem),
            #                                             torch.zeros(N_sources_tem).reshape((N_sources_tem,)))
            # print('test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # theta_esti_all_tem_true[0, :] = torch.zeros((1, N_sources_tem))
            
            # theta_esti_all_tem[0, :] = cvxopt_solve_qp(C, D_T, -1 * torch.eye(N_sources_tem), torch.zeros(N_sources_tem))

            [Gradient_outerAll_x, Gradient_outerAll_y] = GradientOuterNew(x_sensor_tem, y_sensor_tem, source_location_x_tem,
                                                                        source_location_y_tem, Wr_x_tem, Wr_y_tem,
                                                                        w_speed_tem, K_tem, H_tem,
                                                                        Phi_tem, sigma_epsilon_tem, lambda_1_tem,
                                                                        theta_esti_all_tem[:, 0, :], multiplier_esti_all[:, 0, :])
            # Gradient_outerAll_x, Gradient_outerAll_y = torch.zeros(N_sources_tem),torch.zeros(N_sources_tem)
            # print('testing22222222222222222222222222222222222222222')
            # print(theta_esti_all_tem[:, 0, :])
            theta_error_all[:, 0, :] = theta_esti_all_tem[:, 0, :] - theta_true_tem
            theta_error_all_QP[:, 0, :] = theta_esti_all_tem_true[:, 0, :] - theta_true_tem
            # print('ttt1: ',Gradient_outerAll_x[0])

            out = list([torch.matmul(Gradient_outerAll_x[jk], theta_error_all[jk, 0, :]), torch.matmul(Gradient_outerAll_y[jk], theta_error_all[jk, 0, :]), theta_error_all_QP[jk, 0, :], theta_esti_all_tem[jk, 0, :], theta_esti_all_tem_true[0, :]] for jk in range(num_samples))

            # Gradient_OuterSize_All_x_tem = torch.matmul(Gradient_outerAll_x, theta_error_all[0, :])
            # Gradient_OuterSize_All_y_tem = torch.matmul(Gradient_outerAll_y, theta_error_all[0, :])
            # return [Gradient_OuterSize_All_x_tem, Gradient_OuterSize_All_y_tem, theta_error_all[0, :], theta_esti_all_tem[0, :], theta_esti_all_tem_true[0, :]]
            return out



        def GradientInnerNew_1(x_all, y_all, source_x, source_y, w_x, w_y, u, K_tem, H_tem, Phi_tem, sigma_e, lambda_1_tem,
                            lambda_2_tem):
            num_sensor = len(x_all)
            num_source = len(source_x)
            A_all = torch.zeros((num_sensor, num_source))
            for i in range(num_sensor):
                for j in range(num_source):
                    x_new = torch.sqrt(
                        ((1. - w_x ** 2.) * (x_all[i] - source_x[j]) - w_x * w_y * (y_all[i] - source_y[j])) ** 2. + (
                                -w_x * w_y * (x_all[i] - source_x[j]) + (1. - w_y ** 2.) * (y_all[i] - source_y[j])) ** 2.)
                    y_new = w_x * (x_all[i] - source_x[j]) + w_y * (y_all[i] - source_y[j])
                    if y_new > 0.:
                        A_all[i, j] = 1 / (2. * torch.pi * K_tem[j] * y_new) * torch.exp(
                            -u * (x_new ** 2. + H_tem[j] ** 2.) / (4. * K_tem[j] * y_new))
                    else:
                        A_all[i, j] = 0.
            C_coef = 1. / (sigma_e ** 2) * (A_all.T @ A_all) + lambda_1_tem * torch.eye(num_source)
            if num_sensor == 1:
                D_coef_T = lambda_2_tem * torch.ones((num_source, 1)) - 1. / (sigma_e ** 2) * A_all.T * Phi_tem
            else:
                D_coef_T = lambda_2_tem * torch.ones((num_source, 1)) - 1. / (sigma_e ** 2) * A_all.T @ Phi_tem

            return [C_coef, D_coef_T]

        def Evaluation(x_sensor_tem, y_sensor_tem, source_location_x_tem, source_location_y_tem, Wr_x_tem, Wr_y_tem,
                                w_speed_tem,
                                K_tem, H_tem, Phi_input, sigma_epsilon_tem, lambda_1_tem, lambda_2_tem, theta_true_tem,
                                N_sources_tem, q_tem, lr_inner_temp):
            num_sensors = len(Phi_input)
            Phi_tem = torch.zeros((num_sensors, 1))
            Phi_tem[:, 0] = Phi_input
            theta_esti_all_tem = torch.zeros((1, N_sources_tem))
            theta_esti_all_tem_true = torch.zeros((1, N_sources_tem))
            theta_error_all = torch.zeros((1, N_sources_tem))
            [C, D_T] = GradientInnerNew_1(x_sensor_tem, y_sensor_tem, source_location_x_tem, source_location_y_tem, Wr_x_tem,
                                        Wr_y_tem, w_speed_tem,
                                        K_tem, H_tem, Phi_tem, sigma_epsilon_tem, lambda_1_tem, lambda_2_tem)
            # Set the initial guess of theta
            theta_esti_all_tem[0, :] = torch.zeros((1, N_sources_tem))  # Here we start from zeros

            # # call the inner loop
            # theta_esti_all_tem_true[0, :] = quadprog_solve_qp(C, D_T.reshape((N_sources_tem,)), -1. * torch.eye(N_sources_tem),
            #                                          torch.zeros(N_sources).reshape((N_sources,)))
            # theta_esti_all_tem[0, :] = cvxopt_solve_qp(C, D_T, -1 * torch.eye(N_sources_tem), torch.zeros(N_sources_tem))

            theta_error_all[0, :] = theta_esti_all_tem_true[0, :] - theta_true_tem
            return [theta_error_all[0, :]]

        #**********************************************************************************************  initial guess 
        # # main
        # # start the local solver to fine-tune the results above
        # start = time.time()
        # cpu_count = os.cpu_count() + 1
        # # the domain of the concentration field
        # x = torch.linspace(-25, 25, 200)
        # y = torch.linspace(-25, 25, 200)

        # # start the global solver to get the initial solution
        # bounds = []
        # min_max = [-25, 25]
        # for ik in range(2 * N_sensors):
        #     bounds.append(min_max)
        # result = optimize.dual_annealing(objective, torch.array(bounds))
        # # evaluate solution
        # solution = result['x']
        # evaluation = objective(solution)
        # print('Based on A-optimal, the initial solution by global solver: f(%s) = %.5f' % (solution, evaluation))

        # # the SGD-based Bi-level approximation method
        # # Initialize the locations of sensors
        # x_sensor = solution[0:N_sensors]
        # y_sensor = solution[N_sensors:]

        # x_sensor = 40*torch.rand(N_sensors)-20
        # y_sensor = 40*torch.rand(N_sensors)-20

        ##################################
        all_source = torch.transpose(torch.stack((source_location_x, source_location_y)),0,1)
        # x = np.random.randn(100, 2) / 6
        # x = torch.from_numpy(x)
        from kmeans_pytorch import kmeans, kmeans_predict
        # set device
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        cluster_ids_x, cluster_centers = kmeans(
            X=all_source, num_clusters=N_sensors, distance='euclidean', device=device
        )
        print('test:',cluster_centers)

        x_sensor = cluster_centers[:,0]
        y_sensor = cluster_centers[:,1]
        ##################################

        # initial = np.loadtxt("initial_cumu_SP_20_10.txt", delimiter=',') 
        # x_sensor = torch.from_numpy(initial[:,0]).to(torch.float32)
        # y_sensor = torch.from_numpy(initial[:,1]).to(torch.float32)
        #**********************************************************************************************

        n_k = Num_iteration_k  # the maximum iteration number of outer loop
        batch_size = Num_SGD_BatchSize  # the size of mini_batch
        num_batch = int(N_samples / batch_size)
        # q = 300  # the maximum iteration number of inner loop
        random_state = None  # the seed to control the shuffling. Here we consider randomness by 'None'

        # fixed samplings
        ws = (ws_upper-ws_lower)*torch.rand(N_samples)+ws_lower  # the wind speed distribution
        wd = (wd_upper-wd_lower)*torch.rand(N_samples)+wd_lower  # the wind angle distribution
        Wr_x = torch.cos(wd / 180. * torch.pi)  # the x part of unit wind vector
        Wr_y = torch.sin(wd / 180. * torch.pi)  # the y part of unit wind vector
        w_speed = torch.abs(ws)  # the wind speed
        # theta_true_all = torch.abs(torch.random.multivariate_normal(mean, cov, N_samples))
        theta_true_all = torch.zeros((N_samples, N_sources))
        X = mean.detach().clone()
        D = sigma_pior_abs * torch.ones(N_sources)
        bound1 = torch.zeros(N_sources)
        bound2 = float('inf') * torch.ones(N_sources)
        for i in range(N_samples):
            theta_true_all[i, :] = torch.from_numpy(stats.truncnorm.rvs((bound1 - X) / D, (bound2 - X) / D, loc=X,
                                                        scale=D))  # sample from the truncated normal distribution
        sensor_noise_all = torch.normal(0, sigma_epsilon, size=(N_samples, N_sensors))
        # define the random setting for shuffling
        all_samplings = torch.cat((Wr_x.reshape(-1,1), Wr_y.reshape(-1,1), w_speed.reshape(-1,1), theta_true_all, sensor_noise_all),dim=1)  # combine all data together
        seed = None if random_state is None else int(random_state)
        # rng = torch.random.default_rng(seed=seed)

        ###############################################################################################
        np.savetxt('source_location_x.txt', source_location_x.numpy(), delimiter=',')
        np.savetxt('source_location_y.txt', source_location_y.numpy(), delimiter=',')
        np.savetxt('ws.txt', ws.numpy(), delimiter=',')
        np.savetxt('wd.txt', wd.numpy(), delimiter=',')
        np.savetxt('theta_true_all.txt', theta_true_all.numpy(), delimiter=',')
        ###############################################################################################

        Theta_error_norm_step_k = torch.zeros(
            n_k * num_batch) + torch.inf  # here we add 100 to the inital settings for while loop requirements
        Theta_error_norm_step_k_eval = torch.zeros(
            n_k) + torch.inf  # here we add 100 to the inital settings for while loop requirements
        Theta_error_norm_step_k_true = torch.zeros(
            n_k * num_batch) + torch.inf  # here we add 100 to the inital settings for while loop requirements
        # print(Theta_error_norm_step_k)
        theta_esti_monitor = torch.zeros((n_k * num_batch, N_sources))
        step_alpha = torch.zeros(n_k * num_batch)
        step_all_x = torch.zeros((n_k * num_batch, N_sensors))
        step_all_y = torch.zeros((n_k * num_batch, N_sensors))
        all_sensor_x = torch.zeros((n_k * num_batch, N_sensors))
        all_sensor_y = torch.zeros((n_k * num_batch, N_sensors))
        stepsize_x = torch.zeros((n_k * num_batch, N_sensors))
        stepsize_y = torch.zeros((n_k * num_batch, N_sensors))
        theta_esti_all = torch.zeros((batch_size, N_sources))
        Gradient_OuterSize_All_x = torch.zeros((batch_size, N_sensors))
        Gradient_OuterSize_All_y = torch.zeros((batch_size, N_sensors))
             
        P = N_sensors
        source_x = source_location_x
        source_y = source_location_y
        sensor_x = x_sensor
        sensor_y = y_sensor
        # Initialize history for plotting
        trajectories_x = [[] for _ in range(P)]
        trajectories_y = [[] for _ in range(P)]

        self.ax1.set_xlim([-26, 26])
        self.ax1.set_ylim([-26, 26])
        self.ax1.set_aspect('equal')
        self.ax1.scatter(source_x, source_y, color='cyan', label='Sources', marker='x')
        trajectory_lines = [self.ax1.plot([], [], linestyle='-', color='green', alpha=0.8)[0] for _ in range(P)]
        sensor_markers_0 = self.ax1.scatter(x_sensor, y_sensor, color='red', marker='*', label='Initial Guess')
        sensor_markers = self.ax1.scatter(x_sensor, y_sensor, color='blue', marker='^', label='Sensors')
        self.ax1.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        for k in range(n_k):
            if self.stop_requested:
                self.status_label.config(text="Optimization stopped at iteration {}".format(k))
                return
            # re-samplings
            seed = None if random_state is None else int(random_state)
            # seed = 111
            # rng = torch.random.default_rng(seed=seed)
            # ws = (ws_upper-ws_lower)*torch.rand(N_samples)+ws_lower  # the wind speed distribution
            # wd = (wd_upper-wd_lower)*torch.rand(N_samples)+wd_lower  # the wind angle distribution
            # Wr_x = torch.cos(wd / 180. * torch.pi)  # the x part of unit wind vector
            # Wr_y = torch.sin(wd / 180. * torch.pi)  # the y part of unit wind vector
            # w_speed = torch.abs(ws)  # the wind speed
            # # theta_true_all = torch.abs(torch.random.multivariate_normal(mean, cov, N_samples))
            # theta_true_all = torch.zeros((N_samples, N_sources))
            # X = mean.detach().clone()
            # D = sigma_pior_abs * torch.ones(N_sources)
            # bound1 = torch.zeros(N_sources)
            # bound2 = float('inf') * torch.ones(N_sources)
            # for i in range(N_samples):
            #     theta_true_all[i, :] = torch.from_numpy(stats.truncnorm.rvs((bound1 - X) / D, (bound2 - X) / D, loc=X,
            #                                                scale=D))  # sample from the truncated normal distribution
            # sensor_noise_all = torch.normal(0, sigma_epsilon, size=(N_samples, N_sensors))
            # # define the random setting for shuffling
            # all_samplings = torch.cat((Wr_x.reshape(-1,1), Wr_y.reshape(-1,1), w_speed.reshape(-1,1), theta_true_all, sensor_noise_all),dim=1)  # combine all data together
            # seed = None if random_state is None else int(random_state)
            # # rng = torch.random.default_rng(seed=seed)

            # shuffle all the samplings
            # rng.shuffle(all_samplings)

            # performing minibatch moves
            count_temp = 0
            for start_batch in range(0, N_samples, batch_size):
                # print('start_batch:',start_batch)
                stop_batch = start_batch + batch_size
                Wr_x_batch, Wr_y_batch, w_speed_batch, theta_true_batch, sensor_noise_batch = \
                    all_samplings[start_batch:stop_batch, 0], all_samplings[start_batch:stop_batch, 1], \
                        all_samplings[start_batch:stop_batch, 2], all_samplings[start_batch:stop_batch, 3:(3 + N_sources)], \
                        all_samplings[start_batch:stop_batch, (3 + N_sources):]
                # Generate the sensor readings
                Phi_T = TwoDimenGauPlumeM_AllSource_Reading_VEC(x_sensor, y_sensor, source_location_x, source_location_y,
                                                                Wr_x_batch, Wr_y_batch, w_speed_batch,
                                                                theta_true_batch, K, H,
                                                                sensor_noise_batch)
                Phi = torch.zeros((N_sensors, batch_size))
                Phi = Phi_T.T

                lr_outer = lr_outer_initial * 2
                if k > 0:
                    count_while = 0
                    while (Theta_error_norm_step_k[k * num_batch + count_temp] - Theta_error_norm_step_k[
                        k * num_batch + count_temp - 1]) > 0:
                        # print('Count:',count_while,Theta_error_norm_step_k[k * num_batch + count_temp] - Theta_error_norm_step_k[k * num_batch + count_temp - 1])
                        lr_outer = lr_outer / 2. # decaying stepsize
                        temp_Gx = 2. * torch.mean(Gradient_OuterSize_All_x, 0)
                        temp_Gy = 2. * torch.mean(Gradient_OuterSize_All_y, 0)
                        temp_Gx[torch.isnan(temp_Gx)] = 0  # to avoid abnormal gradients
                        temp_Gy[torch.isnan(temp_Gy)] = 0  # to avoid abnormal gradients
                        # update
                        x_sensor = all_sensor_x[k * num_batch + count_temp - 1, :] - lr_outer * temp_Gx
                        y_sensor = all_sensor_y[k * num_batch + count_temp - 1, :] - lr_outer * temp_Gy
                        for mk in range(N_sensors):
                            x_sensor[mk] = max(-25, x_sensor[mk])
                            x_sensor[mk] = min(25, x_sensor[mk])
                            y_sensor[mk] = max(-25, y_sensor[mk])
                            y_sensor[mk] = min(25, y_sensor[mk])

                        start = timer()

                        tempdata = Update_Inner_OuterStep(x_sensor, y_sensor, source_location_x, source_location_y,
                                                            Wr_x_batch, Wr_y_batch,
                                                            w_speed_batch,
                                                            K, H, Phi, sigma_epsilon, lambda_1, lambda_2,
                                                            theta_true_batch, N_sources, q, lr_inner_initial,k)
                        end = timer()
                        # print('time1:',end - start)

                        start = timer()
                        # Split the data from parallel computing
                        Theta_error_norm_step_k[k * num_batch + count_temp] = 0
                        for jk in range(batch_size):
                            temp_xy = tempdata[jk]
                            if torch.any(torch.isinf(temp_xy[2])) or torch.any(
                                    torch.isnan(temp_xy[2])):  # avoid the Inf and NaN value from the inner solver
                                Theta_error_norm_step_k[k * num_batch + count_temp] += (torch.linalg.norm(
                                    theta_true_batch[jk, :])) ** 2 / batch_size
                            else:
                                Theta_error_norm_step_k[k * num_batch + count_temp] += (torch.linalg.norm(
                                    temp_xy[2])) ** 2 / batch_size
                        end = timer()
                        # print('time3:',end - start)
                        if count_while == Num_Backtracking:
                            break
                        count_while += 1
                        end = timer()
                        

                    all_sensor_x[k * num_batch + count_temp, :] = x_sensor
                    all_sensor_y[k * num_batch + count_temp, :] = y_sensor
                    stepsize_x[k * num_batch + count_temp, :] = temp_Gx  # check the step size of sensor locations
                    stepsize_y[k * num_batch + count_temp, :] = temp_Gy

                    # Split the data from parallel computing
                    Theta_error_norm_step_k[k * num_batch + count_temp] = 0
                    Theta_error_norm_step_k_true[k * num_batch + count_temp] = 0
                    for jk in range(batch_size):
                        temp_xy = tempdata[jk]
                        Gradient_OuterSize_All_x[jk, :] = temp_xy[0]
                        Gradient_OuterSize_All_y[jk, :] = temp_xy[1]
                        if torch.any(torch.isinf(temp_xy[2])) or torch.any(
                                torch.isnan(temp_xy[2])):  # avoid the Inf and NaN value from the inner solver
                            Theta_error_norm_step_k[k * num_batch + count_temp] += (torch.linalg.norm(
                                theta_true_batch[jk, :])) ** 2 / batch_size
                            Theta_error_norm_step_k_true[k * num_batch + count_temp] += (torch.linalg.norm(
                                theta_true_batch[jk, :])) ** 2 / batch_size
                        else:
                            Theta_error_norm_step_k[k * num_batch + count_temp] += (torch.linalg.norm(
                                temp_xy[2])) ** 2 / batch_size
                            Theta_error_norm_step_k_true[k * num_batch + count_temp] += (torch.linalg.norm(
                                temp_xy[4] - theta_true_batch[jk, :])) ** 2 / batch_size
                        theta_esti_all[jk, :] = temp_xy[3]
                        # print(temp_xy[3])
                        # print(temp_xy[4])
                    theta_esti_monitor[k * num_batch + count_temp, :] = temp_xy[3]
                    step_alpha[k * num_batch + count_temp] = lr_outer
                    step_all_x[k * num_batch + count_temp] = - lr_outer * temp_Gx
                    step_all_y[k * num_batch + count_temp] = - lr_outer * temp_Gy
                    # print('the step size:', lr_outer)
                    end = timer()
                    # print('time4:',end - start)

                if k == 0:
                    
                    import torch
                    from torch import autograd

                    lr_outer = 0.
                    temp_Gx = 2. * torch.mean(Gradient_OuterSize_All_x, 0)
                    temp_Gy = 2. * torch.mean(Gradient_OuterSize_All_y, 0)
                    temp_Gx[torch.isnan(temp_Gx)] = 0  # to avoid abnormal gradients
                    temp_Gy[torch.isnan(temp_Gy)] = 0  # to avoid abnormal gradients
                    # update
                    x_sensor = x_sensor - lr_outer * temp_Gx
                    y_sensor = y_sensor - lr_outer * temp_Gy
                    for mk in range(N_sensors):
                        x_sensor[mk] = max(-25, x_sensor[mk])
                        x_sensor[mk] = min(25, x_sensor[mk])
                        y_sensor[mk] = max(-25, y_sensor[mk])
                        y_sensor[mk] = min(25, y_sensor[mk])
                    all_sensor_x[k * num_batch + count_temp, :] = x_sensor
                    all_sensor_y[k * num_batch + count_temp, :] = y_sensor
                    stepsize_x[k * num_batch + count_temp, :] = temp_Gx  # check the step size of sensor locations
                    stepsize_y[k * num_batch + count_temp, :] = temp_Gy

                    tempdata = Update_Inner_OuterStep(x_sensor, y_sensor, source_location_x, source_location_y,
                                                        Wr_x_batch, Wr_y_batch,
                                                        w_speed_batch,
                                                        K, H, Phi, sigma_epsilon, lambda_1, lambda_2,
                                                        theta_true_batch, N_sources, q, lr_inner_initial,k)        

                    # Split the data from parallel computing
                    Theta_error_norm_step_k[k * num_batch + count_temp] = 0
                    Theta_error_norm_step_k_true[k * num_batch + count_temp] = 0
                    for jk in range(batch_size):
                        temp_xy = tempdata[jk]
                        Gradient_OuterSize_All_x[jk, :] = temp_xy[0]
                        Gradient_OuterSize_All_y[jk, :] = temp_xy[1]
                        if torch.any(torch.isinf(temp_xy[2])) or torch.any(
                                torch.isnan(temp_xy[2])):  # avoid the Inf and NaN value from the inner solver
                            Theta_error_norm_step_k[k * num_batch + count_temp] += (torch.linalg.norm(
                                theta_true_batch[jk, :])) ** 2 / batch_size
                            Theta_error_norm_step_k_true[k * num_batch + count_temp] += (torch.linalg.norm(
                                theta_true_batch[jk, :])) ** 2 / batch_size
                        else:
                            Theta_error_norm_step_k[k * num_batch + count_temp] += (torch.linalg.norm(
                                temp_xy[2])) ** 2 / batch_size
                            Theta_error_norm_step_k_true[k * num_batch + count_temp] += (torch.linalg.norm(
                                temp_xy[4] - theta_true_batch[jk, :])) ** 2 / batch_size
                        theta_esti_all[jk, :] = temp_xy[3]
                        # print(temp_xy[3])
                        # print(temp_xy[4])
                    theta_esti_monitor[k * num_batch + count_temp, :] = temp_xy[3]
                    step_alpha[k * num_batch + count_temp] = lr_outer
                count_temp += 1

            # ###########################################################################################
            # # evaluate the objective using large sample set at this iteration k
            # # torch.random.seed(0)
            # ws = torch.random.uniform(ws_lower, ws_upper, N_samples_large)  # the wind speed distribution
            # wd = torch.random.uniform(wd_lower, wd_upper, N_samples_large)  # the wind angle distribution
            # Wr_x = torch.cos(wd / 180. * torch.pi)  # the x part of unit wind vector
            # Wr_y = torch.sin(wd / 180. * torch.pi)  # the y part of unit wind vector
            # w_speed = torch.abs(ws)  # the wind speed
            # theta_true_all = torch.zeros((N_samples_large, N_sources))
            # X = torch.array(mean)
            # D = sigma_pior_abs * torch.ones(N_sources)
            # bound1 = torch.zeros(N_sources)
            # bound2 = float('inf') * torch.ones(N_sources)
            # for i in range(N_samples_large):
            #     theta_true_all[i, :] = stats.truncnorm.rvs((bound1 - X) / D, (bound2 - X) / D, loc=X,
            #                                                scale=D)  # sample from the truncated normal distribution
            # sensor_noise_all = torch.random.normal(0, sigma_epsilon, size=(N_samples_large, N_sensors))
            # # define the random setting for shuffling
            # all_samplings = torch.c_[Wr_x, Wr_y, w_speed, theta_true_all, sensor_noise_all]  # combine all data together
            # seed = None if random_state is None else int(random_state)
            # # rng = torch.random.default_rng(seed=seed)

            # Wr_x_batch, Wr_y_batch, w_speed_batch, theta_true_batch, sensor_noise_batch = \
            #     all_samplings[:, 0], all_samplings[:, 1], \
            #         all_samplings[:, 2], all_samplings[:, 3:(3 + N_sources)], \
            #         all_samplings[:, (3 + N_sources):]
            # # Generate the sensor readings
            # Phi_T = TwoDimenGauPlumeM_AllSource_Reading_VEC(x_sensor, y_sensor, source_location_x, source_location_y,
            #                                                 Wr_x_batch, Wr_y_batch, w_speed_batch,
            #                                                 theta_true_batch, K, H,
            #                                                 sensor_noise_batch)
            # Phi = torch.zeros((N_sensors, batch_size))
            # Phi = Phi_T.T

            # # note that this for-loop is implemented in parallel
            # parallel = Parallel(n_jobs=-1, prefer="processes")
            # tempdata = parallel(
            #     delayed(Evaluation)(x_sensor, y_sensor, source_location_x, source_location_y,
            #                                     Wr_x_batch[i], Wr_y_batch[i],
            #                                     w_speed_batch[i],
            #                                     K, H, Phi[:, i], sigma_epsilon, lambda_1, lambda_2,
            #                                     theta_true_batch[i, :], N_sources, q, lr_inner_initial) for i in range(N_samples_large))
            # # Split the data from parallel computing
            # Theta_error_norm_step_k_eval[k] = 0
            # for jk in range(N_samples_large):
            #     temp_xy = tempdata[jk]
            #     if torch.any(torch.isinf(temp_xy[0])) or torch.any(
            #             torch.isnan(temp_xy[0])):  # avoid the Inf and NaN value from the inner solver
            #         Theta_error_norm_step_k_eval[k] += (torch.linalg.norm(
            #             theta_true_batch[jk, :])) ** 2 / N_samples_large
            #     else:
            #         Theta_error_norm_step_k_eval[k] += (torch.linalg.norm(
            #             temp_xy[0])) ** 2 / N_samples_large
            # ############################################################################################
            for p in range(P):
                trajectories_x[p].append(x_sensor[p])
                trajectories_y[p].append(y_sensor[p])
                trajectory_lines[p].set_data(trajectories_x[p], trajectories_y[p])
            
            sensor_markers.remove()
            sensor_markers = self.ax1.scatter(x_sensor, y_sensor, color='blue', marker='^')

            self.status_label.config(text=f"Iteration: {k + 1} / {n_k}")
            self.progress["value"] = k + 1
            self.canvas.draw()
            # time.sleep(0.05)
            
            print('Progress: k=', k, ' / ', n_k - 1)
            # print('Objective: ', Theta_error_norm_step_k_eval[k])
            # print('Objective: ', Theta_error_norm_step_k[k])
            if (torch.all(torch.abs(temp_Gx) < tol_G) and torch.all(torch.abs(temp_Gy) < tol_G)) and k > 1:
                if torch.all(0 < torch.abs(temp_Gx)) and torch.all(0 < torch.abs(temp_Gy)):
                    print('the solution has converged!')
                    break
        end = time.time()
        self.status_label.config(text='Finished. The total calculation time is {:.4f} mins.'.format((end - start_all)/60.0))
        print('the total calculation time is {:.4f} mins'.format((end - start_all)/60.0))  # the computational time

        # print('The trajectory of the X value for the 1st sensor:', all_sensor_x[:, 0])
        # print('The trajectory of the Y value for the 1st sensor:', all_sensor_y[:, 0])
        # print('The gradients of objective w.r.t X:', stepsize_x)
        # print('The gradients of objective w.r.t Y:', stepsize_y)
        # # print('The objective values:', Theta_error_norm_step_k)
        # # print(theta_esti_all)
        # # print(theta_esti_monitor)
        # print('The learning rates at each step:', step_alpha)


if __name__ == "__main__":
    root = tk.Tk()
    app = SensorPlacementGUI(root)
    root.mainloop()
