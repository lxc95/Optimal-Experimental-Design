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

"""
    A. Inputs: (User-defined parameters)
        1. Source locations of interest, e.g., (source_location[0] = -15., source_location[0] = 17.)...
           You can define any number of sources as you needed, just input the lists of X and Y.
        2. The distribution of emission rates; You can define any distributions here, but we use truncated Gaussian 
           distribution for illustration purpose.
        3. The heights of stacks, i.e., H, and the eddy diffusion coefficient, which is simplified, i.e., K.
        4. The number of Monte Carlo samples, i.e., N_samples.
        5. The number of sensors you would like to place, i.e., N_sensors
        6. The sensor noise level -> the standard deviation, sigma_epsilon.
        7. The distributions of wind speed and wind directions. Here we use uniform distribution which 
           needs upper bounds and lower bounds.
        8. The hyper_parameters for the inverse model, i.e., lambda_1 and lambda_2.
        9. (optional) The maximum iteration steps for the outer iteration, i.e., Num_iteration_k.
        10. (optional) The batch size in SGD, i.e., Num_SGD_BatchSize.
        11. (optional) Backtracking steps, i.e., Num_Backtracking.
        12. (optional) The initial learning rate of the backtracking, i.e., lr_outer_initial
        13. (optional) The number of monte carlo samplings in A-optimal design, i.e., N in the 'Objective' function.
        14. (optional) The tolerance for step length (i.e., the gradient of the outer iteration), tol_G
        15. (optional) The maximum iteration number of inner loop, q
    B. Outputs:
        1. The final designs of all required sensors.
        2. The trajectory of sensors' locations during iterations and the corresponding objective values which are 
           supposed to decrease step by step.
    Note:
        Our solver, which is a local solver, is heavily dependent on the initial guess of source locations. So you 
        should try different initial guess with our solver. Here we provide an option of using A-optimal design as 
        initial guess, but the A-optimal design assumes Gaussian distribution for emission rates. Hence, if the 
        data-generating distribution of your emission rates is far away from a Gaussian distribution, the A-optimal 
        design is not a good option. In addition, you have to input the covariance information, 'sigma_pr', of the 
        assumed Gaussian distribution in the function 'Objective'. It is also noted that the number of Monte Carlo 
        samplings 'N' and the global solver in the function 'Objective' should be tuned according to your requirement.
        If you don't care about the A-optimal designs, you can just take the A-optimal designs as some arbitrary 
        initial guesses of sensor locations.

        The inner solver is also optional. The existing quadratic programming solvers 'cvxopt_solve_qp', 
        'quadprog_solve_qp' and the customized solver 'Inner_loop' are all available, but the customized solver needs 
        to be tuned well.
"""


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
        source_location_x = 40*torch.rand(N_of_source).to(device)-20  # X axis
        source_location_y = 40*torch.rand(N_of_source).to(device)-20  # Y axis

        N_sources = len(source_location_x)
        # Define the mean and variance of emission rates for truncated Gaussian distribution
        # mean = [8., 10., 9., 8., 10., 9., 8., 10., 9., 10.]  # the mean of emission rates for the above sources
        mean = 2*torch.rand(N_of_source).to(device) + 8  # the mean of emission rates for the above sources
        sigma_pior_abs = 20
        cov = sigma_pior_abs * torch.eye(N_sources).to(device)  # the covariance of these emission rates
        # Define the height of stacks and the eddy diffusion coefficient
        H = config["H"] * torch.ones(N_sources).to(device)  # the height of stacks
        K = config["coeff"] * torch.ones(N_sources).to(device)  # the eddy diffusion coefficient, which is simplified
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
                ws = torch.random.uniform(ws_lower, ws_upper, 1).to(device)  # the wind speed distribution
                wd = torch.random.uniform(wd_lower, wd_upper, 1).to(device)  # the wind angle distribution
                w_x = torch.cos(wd / 180. * torch.pi)  # the x part of unit wind vector
                w_y = torch.sin(wd / 180. * torch.pi)  # the y part of unit wind vector
                u = torch.abs(ws)  # the wind speed
                # the F operator
                A_all = torch.zeros((num_sensor, num_source)).to(device)
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
                cov_post = torch.linalg.inv(1 / sigma_epsilon ** 2 * A_all.T @ A_all + 1 / sigma_pr ** 2 * torch.eye(num_source)).to(device)
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
            out = Pi @ torch.ones(len(source_x), device=Pi.device).to(device) + noise
            return out


        def GradientInnerNew(x_all, y_all, source_x, source_y, w_x, w_y, u, K_tem, H_tem, Phi_tem, sigma_e, lambda_1_tem,
                            lambda_2_tem):
            x_all = x_all.to(device)
            y_all = y_all.to(device)
            source_x = source_x.to(device)
            source_y = source_y.to(device)
            w_x = w_x.to(device)
            w_y = w_y.to(device)
            u = u.to(device)
            K_tem = K_tem.to(device)
            H_tem = H_tem.to(device)
            Phi_tem = Phi_tem.to(device)
            sigma_e = torch.tensor(sigma_e, device=device)
            lambda_1_tem = torch.tensor(lambda_1_tem, device=device)
            lambda_2_tem = torch.tensor(lambda_2_tem, device=device)
            num_sensor = len(x_all)
            num_source = len(source_x)
            num_samples = len(w_x)
            A_all = torch.zeros((num_samples,num_sensor, num_source)).to(device)
            x_new = torch.sqrt(
                ((1. - w_x[:,None,None] ** 2.) * (x_all.reshape(1,-1).T - source_x.reshape(1,-1)) - w_x[:,None,None] * w_y[:,None,None] * (y_all.reshape(1,-1).T - source_y.reshape(1,-1))) ** 2. + (
                        -w_x[:,None,None] * w_y[:,None,None] * (x_all.reshape(1,-1).T - source_x.reshape(1,-1)) + (1. - w_y[:,None,None] ** 2.) * (y_all.reshape(1,-1).T - source_y.reshape(1,-1))) ** 2.)
            y_new = w_x[:,None,None] * (x_all.reshape(1,-1).T - source_x.reshape(1,-1)) + w_y[:,None,None] * (y_all.reshape(1,-1).T - source_y.reshape(1,-1))
            y_new_orig = y_new.detach().clone()
            y_new[y_new_orig<=0]=1
            A_all = 1 / (2. * torch.pi * K_tem[None,None,:] * y_new) * torch.exp(
                    -u[:,None,None] * (x_new ** 2. + H_tem[None,None,:] ** 2.) / (4. * K_tem[None,None,:] * y_new))
            A_all[y_new_orig<=0]=0 

            C_coef = torch.zeros((num_samples,num_source, num_source)).to(device)
            D_coef_T = torch.zeros((num_source,num_samples)).to(device)
            # print(Phi_tem.shape)
            start = timer()
            C_coef = torch.bmm(A_all.transpose(1, 2), A_all) / (sigma_e**2)
            C_coef += lambda_1_tem * torch.eye(num_source, device=A_all.device).unsqueeze(0)
            if num_sensor == 1:
                # Phi_tem.T shape becomes (num_samples, 1), unsqueeze to (num_samples, 1, 1)
                D_coef_T = lambda_2_tem - (1. / sigma_e**2) * A_all.squeeze(1).transpose(1, 0) * Phi_tem  # Broadcasting
            else:
                Phi_tem_exp = Phi_tem.T.unsqueeze(2)  # (num_samples, num_sensor, 1)
                D_term = torch.bmm(A_all.transpose(1, 2), Phi_tem_exp)  # (num_samples, num_source, 1)
                D_coef_T
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


        def GradientOuterNew(x_all, y_all, source_x, source_y, w_x, w_y, u, 
                            K_tem, H_tem, Phi_tem, sigma_e, lambda_1_tem, 
                            theta_curr, multiplier_esti):
            # Ensure all inputs are on the target device (GPU)
            x_all = x_all.to(device)
            y_all = y_all.to(device)
            source_x = source_x.to(device)
            source_y = source_y.to(device)
            w_x = w_x.to(device)
            w_y = w_y.to(device)
            u = u.to(device)
            theta_curr = theta_curr.to(device)
            K_tem = K_tem.to(device)
            H_tem = H_tem.to(device)
            Phi_tem = Phi_tem.to(device)
            multiplier_esti = multiplier_esti.to(device)

            # Convert sigma_e and lambda_1_tem to device tensors (scalar values)
            sigma_e = torch.tensor(sigma_e, device=device)
            lambda_1_tem = torch.tensor(lambda_1_tem, device=device)

            num_sensor = len(x_all)
            num_source = len(source_x)
            num_samples = len(w_x)

            # Compute A_all (design matrix) for all samples, sensors, and sources in a batched manner
            x_new = torch.sqrt(
                ((1. - w_x[:, None, None] ** 2.) * (x_all.reshape(1, -1).T - source_x.reshape(1, -1))
                - w_x[:, None, None] * w_y[:, None, None] * (y_all.reshape(1, -1).T - source_y.reshape(1, -1))) ** 2.
                + (
                    -w_x[:, None, None] * w_y[:, None, None] * (x_all.reshape(1, -1).T - source_x.reshape(1, -1))
                    + (1. - w_y[:, None, None] ** 2.) * (y_all.reshape(1, -1).T - source_y.reshape(1, -1))
                ) ** 2.
            )
            y_new = w_x[:, None, None] * (x_all.reshape(1, -1).T - source_x.reshape(1, -1)) \
                    + w_y[:, None, None] * (y_all.reshape(1, -1).T - source_y.reshape(1, -1))
            # Avoid division by zero by replacing non-positive y_new with 1 (and mark for zeroing A_all)
            y_new_orig = y_new.detach().clone()
            y_new[y_new_orig <= 0] = 1.0

            # Compute A_all using broadcasting over samples, sensors, and sources
            A_all = 1.0 / (2. * torch.pi * K_tem[None, None, :] * y_new) * torch.exp(
                -u[:, None, None] * (x_new ** 2. + H_tem[None, None, :] ** 2.) / (4. * K_tem[None, None, :] * y_new)
            )
            # Zero out entries where original y_new was non-positive
            A_all[y_new_orig <= 0] = 0.0

            # Compute C_coef = (1/sigma_e^2) * (A_all^T A_all) + lambda_1_tem * I for each sample (batched)
            # A_all^T A_all gives shape (num_samples, num_source, num_source) via batched matrix multiplication
            C_coef = (1.0 / (sigma_e ** 2)) * torch.bmm(A_all.transpose(1, 2), A_all)  # batch inner products:contentReference[oaicite:0]{index=0}
            I = torch.eye(num_source, device=device)  # identity matrix for regularization
            C_coef = C_coef + lambda_1_tem * I  # adds lambda_1_tem * I to each sample's matrix (broadcast along batch)

            # Compute gradients G_x and G_y without loops using the pre-defined vectorized gradient functions
            # Prepare index arrays for source pairs (to compute second-order gradients w.rt source positions)
            arr = torch.arange(num_source, device=device)
            arr1 = torch.repeat_interleave(arr, num_source)  # e.g., [0,0,...,1,1,...,N-1,...] length N^2
            arr2 = arr.repeat(num_source)                    # e.g., [0,1,2,...,0,1,2,...] length N^2

            # Vectorized gradient computations (assuming the _VEC functions return batched gradients as described)
            Gradient_AA_x_cal = Gradient_AA_x_VEC(x_all, y_all,
                                                source_x[arr1], source_y[arr1],
                                                source_x[arr2], source_y[arr2],
                                                w_x, w_y, u,
                                                K_tem[arr1], H_tem[arr1],
                                                K_tem[arr2], H_tem[arr2])
            Gradient_AA_y_cal = Gradient_AA_y_VEC(x_all, y_all,
                                                source_x[arr1], source_y[arr1],
                                                source_x[arr2], source_y[arr2],
                                                w_x, w_y, u,
                                                K_tem[arr1], H_tem[arr1],
                                                K_tem[arr2], H_tem[arr2])
            Gradient_A_x_cal = Gradient_A_x_VEC(x_all, y_all, source_x, source_y,
                                            w_x, w_y, u, K_tem, H_tem)
            Gradient_A_y_cal = Gradient_A_y_VEC(x_all, y_all, source_x, source_y,
                                            w_x, w_y, u, K_tem, H_tem)

            # Combine gradients to form G_x and G_y (shape: [num_samples, num_sensor, num_source])
            scaling_factor = 1.0 / (sigma_e ** 2)
            # Expand theta_curr for broadcasting: shape (num_samples, 1, 1, num_source)
            theta_curr_expanded = theta_curr.unsqueeze(1).unsqueeze(1)
            # Compute G_x by summing over source-pair gradients weighted by theta (vectorized form of double sum)
            G_x = scaling_factor * (Gradient_AA_x_cal.view(num_samples, num_sensor, num_source, num_source)
                                    * theta_curr_expanded).sum(dim=3)
            G_y = scaling_factor * (Gradient_AA_y_cal.view(num_samples, num_sensor, num_source, num_source)
                                    * theta_curr_expanded).sum(dim=3)
            # Subtract contributions from Phi_tem (linear term)
            # Phi_tem is assumed to have shape [num_sensor, num_samples] or similar; transpose to align with [num_samples, num_sensor]
            Phi_T = Phi_tem.T.to(device)
            G_x -= scaling_factor * (Phi_T[:, :, None] * Gradient_A_x_cal)
            G_y -= scaling_factor * (Phi_T[:, :, None] * Gradient_A_y_cal)

            # Prepare outputs
            coef_x = torch.zeros((num_samples, num_sensor, num_source), device=device)
            coef_y = torch.zeros((num_samples, num_sensor, num_source), device=device)

            # Use batched pseudoinverse to solve linear systems for all samples and sensors simultaneously
            # Compute C_coef_inv for each sample as pseudoinverse (supports batched input):contentReference[oaicite:1]{index=1}
            C_coef_inv = torch.linalg.pinv(C_coef.detach().clone())

            # Create mask and diagonal mask for active constraints (multiplier_esti > 0 indicates active constraints)
            mask_active = (multiplier_esti > 0)  # shape: (num_samples, num_source) boolean
            mask_active_f = mask_active.to(dtype=C_coef_inv.dtype)
            # Construct masked C_coef_inv (zeroing out rows/cols for inactive sources)
            B = C_coef_inv * mask_active_f.unsqueeze(2) * mask_active_f.unsqueeze(1)
            # Pseudoinverse of masked matrix B for each sample (handles singular cases)
            B_pinv = torch.linalg.pinv(B)

            # Detach gradients to avoid backprop through linear solver
            G_x_det = G_x.detach()
            G_y_det = G_y.detach()

            # Solve for coef_x (regression coefficients) in batch:
            # reg1_final = -C_coef_inv @ G_x^T + C_coef_inv @ [P^T * (P * C_coef_inv * P^T)^+ * P * C_coef_inv @ G_x^T]
            # Compute Y_x = C_coef_inv @ G_x^T (initial unconstrained solution for each sensor)
            Y_x = torch.bmm(C_coef_inv, G_x_det.transpose(1, 2))            # shape: (num_samples, num_source, num_sensor)
            # Apply mask: zero out contributions of inactive sources
            Y_x_masked = Y_x * mask_active_f.unsqueeze(2)                  # shape: (num_samples, num_source, num_sensor)
            # Compute adjustment term: Z_x = B_pinv @ Y_x_masked
            Z_x = torch.bmm(B_pinv, Y_x_masked)                            # shape: (num_samples, num_source, num_sensor)
            # Final solution (transposed): reg1_transposed_x = -Y_x + C_coef_inv @ Z_x
            reg1_transposed_x = -Y_x + torch.bmm(C_coef_inv, Z_x)           # shape: (num_samples, num_source, num_sensor)
            coef_x = reg1_transposed_x.transpose(1, 2)                     # shape: (num_samples, num_sensor, num_source)

            # Solve for coef_y similarly (reuse C_coef_inv, B_pinv, etc. as they do not depend on G):
            Y_y = torch.bmm(C_coef_inv, G_y_det.transpose(1, 2))
            Y_y_masked = Y_y * mask_active_f.unsqueeze(2)
            Z_y = torch.bmm(B_pinv, Y_y_masked)
            reg1_transposed_y = -Y_y + torch.bmm(C_coef_inv, Z_y)
            coef_y = reg1_transposed_y.transpose(1, 2)

            # Handle any NaN values in gradients: if any component of G_x or G_y for a sensor is NaN, set that sensor's output to zero
            mask_nan_x = torch.isnan(G_x_det).any(dim=2)  # shape: (num_samples, num_sensor)
            mask_nan_y = torch.isnan(G_y_det).any(dim=2)
            if mask_nan_x.any():
                coef_x = coef_x * (~mask_nan_x).unsqueeze(2)
            if mask_nan_y.any():
                coef_y = coef_y * (~mask_nan_y).unsqueeze(2)

            # Safety check: zero-out any abnormal large solutions (if sum of coefficients > 1e9, treat as invalid)
            sum_x = coef_x.sum(dim=2)
            sum_y = coef_y.sum(dim=2)
            mask_abnormal_x = (sum_x > 1e9)
            mask_abnormal_y = (sum_y > 1e9)
            if mask_abnormal_x.any():
                coef_x = coef_x * (~mask_abnormal_x).unsqueeze(2)
            if mask_abnormal_y.any():
                coef_y = coef_y * (~mask_abnormal_y).unsqueeze(2)

            return [coef_x, coef_y]



        # def Inner_loop(q_tem, C, D_T, N_sources_tem, para_lr_inner, lr_inner):
        #     theta_esti_all_tem = torch.zeros((q_tem, N_sources_tem)).to(device)
        #     multiplier_esti_all_tem = torch.zeros((q_tem, N_sources_tem)).to(device)
        #     Gradient_theta_All = torch.zeros((q_tem, N_sources_tem)).to(device)
        #     Gradient_multiplier_All = torch.zeros((q_tem, N_sources_tem)).to(device)
        #     coef_matrix = torch.eye(N_sources_tem).to(device)
        #     identity_matrix = torch.eye(N_sources_tem).to(device)
        #     gamma_r = 1
        #     # Set the initial guess of theta and Lagrangian multiplier
        #     theta_esti_all_tem[0, :] = torch.zeros((1, N_sources_tem)).to(device)  # Here we start from zeros as the initial values
        #     multiplier_esti_all_tem[0, :] = torch.zeros((1, N_sources_tem)).to(device)  # Here we start from zeros as the initial values
        #     # print('trytry:',q_tem)

        #     # Assume all tensors are already on GPU
        #     identity_matrix = identity_matrix.to(device)
        #     theta_esti_all_tem = theta_esti_all_tem.to(device)
        #     multiplier_esti_all_tem = multiplier_esti_all_tem.to(device)
        #     C = C.to(device)
        #     D_T = D_T.to(device)
        #     coef_matrix = coef_matrix.to(device)

        #     for j in range(q_tem - 1):
        #         theta_j = theta_esti_all_tem[j]  # shape: [N_sources_tem]
                
        #         # Compute projection values for all mk in one batch
        #         prod = -torch.matmul(coef_matrix, theta_j)  # [N_sources_tem]
        #         penalty = gamma_r * prod + multiplier_esti_all_tem[j]  # [N_sources_tem]
        #         projection = torch.clamp(penalty, min=0.0)  # [N_sources_tem]
                
        #         # Weighted sum of projection * -coef_matrix
        #         gradient_proj = torch.einsum('m,mn->n', projection, -coef_matrix)  # shape: [N_sources_tem]
        #         Gradient_theta_All[j] = torch.matmul(C, theta_j) + D_T.T + gradient_proj

        #         # Compute Gradient for multipliers
        #         term_m = gamma_r * prod + multiplier_esti_all_tem[j]  # [N_sources_tem]
        #         delta_m = torch.clamp(term_m, min=0.0) - multiplier_esti_all_tem[j]
        #         Gradient_multiplier_All[j] = (1. / gamma_r) * torch.einsum('m,mn->n', delta_m, identity_matrix)

        #         # Update theta and multipliers
        #         lr_j = lr_inner / torch.sqrt(torch.tensor(j + 1., device=device)) if j > para_lr_inner else lr_inner
        #         theta_esti_all_tem[j + 1] = theta_esti_all_tem[j] - lr_j * Gradient_theta_All[j]
        #         multiplier_esti_all_tem[j + 1] = multiplier_esti_all_tem[j] + lr_j * Gradient_multiplier_All[j]

        #         # Projection (non-negativity constraint)
        #         theta_esti_all_tem[j + 1] = torch.clamp(theta_esti_all_tem[j + 1], min=0.0)
        #         multiplier_esti_all_tem[j + 1] = torch.clamp(multiplier_esti_all_tem[j + 1], min=0.0)

        #     return theta_esti_all_tem[q_tem-1, :], multiplier_esti_all_tem[q_tem-1, :]

        def Inner_loop(q_tem, C_batch, D_batch, N_sources_tem, para_lr_inner, lr_inner):
            """
            Batched version of Inner_loop for GPU execution.
            Inputs:
                q_tem: int, number of inner-loop steps
                C_batch: (B, N, N) batch of C matrices
                D_batch: (B, N) batch of D vectors
                N_sources_tem: int, number of variables
                para_lr_inner: int, step index for adaptive lr
                lr_inner: float, learning rate

            Returns:
                theta_all: (B, q_tem, N)
                multiplier_all: (B, q_tem, N)
            """
            B = C_batch.shape[0]  # num_samples
            device = C_batch.device

            theta_all = torch.zeros(B, q_tem, N_sources_tem, device=device)
            multiplier_all = torch.zeros(B, q_tem, N_sources_tem, device=device)

            identity_matrix = torch.eye(N_sources_tem, device=device).unsqueeze(0).expand(B, -1, -1)  # (B, N, N)

            for j in range(q_tem - 1):
                theta_j = theta_all[:, j, :]  # (B, N)
                multiplier_j = multiplier_all[:, j, :]  # (B, N)

                coef_theta = -torch.bmm(C_batch, theta_j.unsqueeze(2)).squeeze(2)  # (B, N)
                proj = torch.clamp(coef_theta + multiplier_j, min=0.0)  # (B, N)
                grad_proj = torch.bmm(proj.unsqueeze(1), -C_batch).squeeze(1)  # (B, N)
                grad_theta = torch.bmm(C_batch, theta_j.unsqueeze(2)).squeeze(2) + D_batch + grad_proj  # (B, N)

                delta_m = torch.clamp(coef_theta + multiplier_j, min=0.0) - multiplier_j
                grad_multiplier = torch.bmm(delta_m.unsqueeze(1), identity_matrix).squeeze(1)  # (B, N)

                lr_j = lr_inner / torch.sqrt(torch.tensor(j + 1., device=device)) if j > para_lr_inner else lr_inner

                theta_all[:, j + 1, :] = torch.clamp(theta_j - lr_j * grad_theta, min=0.0)
                multiplier_all[:, j + 1, :] = torch.clamp(multiplier_j + lr_j * grad_multiplier, min=0.0)

            return theta_all, multiplier_all



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
            theta_esti_all_tem = torch.zeros((num_samples, 1, N_sources_tem)).to(device)
            theta_esti_all_tem_true = torch.zeros((num_samples, 1, N_sources_tem)).to(device)
            multiplier_esti_all = torch.zeros((num_samples, 1, N_sources_tem)).to(device)
            theta_error_all = torch.zeros((num_samples, 1, N_sources_tem)).to(device)
            theta_error_all_QP = torch.zeros((num_samples, 1, N_sources_tem)).to(device)
            start = timer()
            [C, D_T] = GradientInnerNew(x_sensor_tem, y_sensor_tem, source_location_x_tem, source_location_y_tem, Wr_x_tem,
                                        Wr_y_tem, w_speed_tem,
                                        K_tem, H_tem, Phi_tem, sigma_epsilon_tem, lambda_1_tem, lambda_2_tem)
            # # Set the initial guess of theta
            # theta_esti_all_tem[0, :] = torch.zeros((1, N_sources_tem))  # Here we start from zeros
            end = timer()
            para_lr_inner = q_tem  # now we ignore this parameter by setting a large number
            # call the inner loop
            # for mk in range(num_samples):
            #     theta_esti_all_tem[mk], multiplier_esti_all[mk] = Inner_loop(q_tem, C[mk], D_T[:,mk], N_sources_tem, para_lr_inner, lr_inner_temp)
            theta_esti_all_tem, multiplier_esti_all = Inner_loop(
                q_tem,                  # scalar
                C,                      # (num_samples, N_sources, N_sources)
                D_T.T,                  # (num_samples, N_sources)
                N_sources_tem,          # scalar
                para_lr_inner,          # scalar
                lr_inner_temp           # scalar
            )
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



        # def GradientInnerNew_1(x_all, y_all, source_x, source_y, w_x, w_y, u, K_tem, H_tem, Phi_tem, sigma_e, lambda_1_tem,
        #                     lambda_2_tem):
        #     num_sensor = len(x_all)
        #     num_source = len(source_x)
        #     A_all = torch.zeros((num_sensor, num_source)).to(device)
        #     for i in range(num_sensor):
        #         for j in range(num_source):
        #             x_new = torch.sqrt(
        #                 ((1. - w_x ** 2.) * (x_all[i] - source_x[j]) - w_x * w_y * (y_all[i] - source_y[j])) ** 2. + (
        #                         -w_x * w_y * (x_all[i] - source_x[j]) + (1. - w_y ** 2.) * (y_all[i] - source_y[j])) ** 2.)
        #             y_new = w_x * (x_all[i] - source_x[j]) + w_y * (y_all[i] - source_y[j])
        #             if y_new > 0.:
        #                 A_all[i, j] = 1 / (2. * torch.pi * K_tem[j] * y_new) * torch.exp(
        #                     -u * (x_new ** 2. + H_tem[j] ** 2.) / (4. * K_tem[j] * y_new))
        #             else:
        #                 A_all[i, j] = 0.
        #     C_coef = 1. / (sigma_e ** 2) * (A_all.T @ A_all) + lambda_1_tem * torch.eye(num_source).to(device)
        #     if num_sensor == 1:
        #         D_coef_T = lambda_2_tem * torch.ones((num_source, 1)).to(device) - 1. / (sigma_e ** 2) * A_all.T * Phi_tem
        #     else:
        #         D_coef_T = lambda_2_tem * torch.ones((num_source, 1)).to(device) - 1. / (sigma_e ** 2) * A_all.T @ Phi_tem

        #     return [C_coef, D_coef_T]

        # def Evaluation(x_sensor_tem, y_sensor_tem, source_location_x_tem, source_location_y_tem, Wr_x_tem, Wr_y_tem,
        #                         w_speed_tem,
        #                         K_tem, H_tem, Phi_input, sigma_epsilon_tem, lambda_1_tem, lambda_2_tem, theta_true_tem,
        #                         N_sources_tem, q_tem, lr_inner_temp):
        #     num_sensors = len(Phi_input)
        #     Phi_tem = torch.zeros((num_sensors, 1)).to(device)
        #     Phi_tem[:, 0] = Phi_input
        #     theta_esti_all_tem = torch.zeros((1, N_sources_tem)).to(device)
        #     theta_esti_all_tem_true = torch.zeros((1, N_sources_tem)).to(device)
        #     theta_error_all = torch.zeros((1, N_sources_tem)).to(device)
        #     [C, D_T] = GradientInnerNew_1(x_sensor_tem, y_sensor_tem, source_location_x_tem, source_location_y_tem, Wr_x_tem,
        #                                 Wr_y_tem, w_speed_tem,
        #                                 K_tem, H_tem, Phi_tem, sigma_epsilon_tem, lambda_1_tem, lambda_2_tem)
        #     # Set the initial guess of theta
        #     theta_esti_all_tem[0, :] = torch.zeros((1, N_sources_tem)).to(device)  # Here we start from zeros

        #     # # call the inner loop
        #     # theta_esti_all_tem_true[0, :] = quadprog_solve_qp(C, D_T.reshape((N_sources_tem,)), -1. * torch.eye(N_sources_tem),
        #     #                                          torch.zeros(N_sources).reshape((N_sources,)))
        #     # theta_esti_all_tem[0, :] = cvxopt_solve_qp(C, D_T, -1 * torch.eye(N_sources_tem), torch.zeros(N_sources_tem))

        #     theta_error_all[0, :] = theta_esti_all_tem_true[0, :] - theta_true_tem
        #     return [theta_error_all[0, :]]

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
        # if torch.cuda.is_available():
        #     device = torch.device('cuda:0')
        # else:
        #     device = torch.device('cpu')
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
        ws = (ws_upper-ws_lower)*torch.rand(N_samples).to(device)+ws_lower  # the wind speed distribution
        wd = (wd_upper-wd_lower)*torch.rand(N_samples).to(device)+wd_lower  # the wind angle distribution
        Wr_x = torch.cos(wd / 180. * torch.pi)  # the x part of unit wind vector
        Wr_y = torch.sin(wd / 180. * torch.pi)  # the y part of unit wind vector
        w_speed = torch.abs(ws)  # the wind speed
        # theta_true_all = torch.abs(torch.random.multivariate_normal(mean, cov, N_samples))
        theta_true_all = torch.zeros((N_samples, N_sources)).to(device)
        X = mean.detach().clone()
        D = sigma_pior_abs * torch.ones(N_sources).to(device)
        bound1 = torch.zeros(N_sources).to(device)
        bound2 = float('inf') * torch.ones(N_sources).to(device)

        # Ensure all are on GPU first (or CPU consistently if you prefer)
        X = mean.detach().clone().to(device)
        D = sigma_pior_abs * torch.ones(N_sources).to(device)
        bound1 = torch.zeros(N_sources).to(device)
        bound2 = float('inf') * torch.ones(N_sources).to(device)
        # Now safe to compute
        a = ((bound1 - X) / D).cpu().numpy()
        b = ((bound2 - X) / D).cpu().numpy()
        X_cpu = X.cpu().numpy()
        D_cpu = D.cpu().numpy()
        # Sampling and transferring to GPU
        theta_true_all = torch.zeros((N_samples, N_sources), device=device)
        for i in range(N_samples):
            sample_np = stats.truncnorm.rvs(a, b, loc=X_cpu, scale=D_cpu)
            theta_true_all[i, :] = torch.from_numpy(sample_np).to(device)

        sensor_noise_all = torch.normal(0, sigma_epsilon, size=(N_samples, N_sensors))
        # define the random setting for shuffling
        all_samplings = torch.cat((
            Wr_x.reshape(-1, 1).to(device),
            Wr_y.reshape(-1, 1).to(device),
            w_speed.reshape(-1, 1).to(device),
            theta_true_all,              # already on device from earlier
            sensor_noise_all.to(device)
        ), dim=1)

        seed = None if random_state is None else int(random_state)
        # rng = torch.random.default_rng(seed=seed)

        ###############################################################################################
        # np.savetxt('source_location_x.txt', source_location_x.numpy(), delimiter=',')
        # np.savetxt('source_location_y.txt', source_location_y.numpy(), delimiter=',')
        # np.savetxt('ws.txt', ws.numpy(), delimiter=',')
        # np.savetxt('wd.txt', wd.numpy(), delimiter=',')
        # np.savetxt('theta_true_all.txt', theta_true_all.numpy(), delimiter=',')
        ###############################################################################################

        Theta_error_norm_step_k = torch.zeros(
            n_k * num_batch).to(device) + torch.inf  # here we add 100 to the inital settings for while loop requirements
        Theta_error_norm_step_k_eval = torch.zeros(
            n_k).to(device) + torch.inf  # here we add 100 to the inital settings for while loop requirements
        Theta_error_norm_step_k_true = torch.zeros(
            n_k * num_batch).to(device) + torch.inf  # here we add 100 to the inital settings for while loop requirements
        # print(Theta_error_norm_step_k)
        theta_esti_monitor = torch.zeros((n_k * num_batch, N_sources)).to(device)
        step_alpha = torch.zeros(n_k * num_batch).to(device)
        step_all_x = torch.zeros((n_k * num_batch, N_sensors)).to(device)
        step_all_y = torch.zeros((n_k * num_batch, N_sensors)).to(device)
        all_sensor_x = torch.zeros((n_k * num_batch, N_sensors)).to(device)
        all_sensor_y = torch.zeros((n_k * num_batch, N_sensors)).to(device)
        stepsize_x = torch.zeros((n_k * num_batch, N_sensors)).to(device)
        stepsize_y = torch.zeros((n_k * num_batch, N_sensors)).to(device)
        theta_esti_all = torch.zeros((batch_size, N_sources)).to(device)
        Gradient_OuterSize_All_x = torch.zeros((batch_size, N_sensors)).to(device)
        Gradient_OuterSize_All_y = torch.zeros((batch_size, N_sensors)).to(device)
             
        P = N_sensors
        source_x = source_location_x
        source_y = source_location_y
        sensor_x = x_sensor
        sensor_y = y_sensor
        source_x_plot = source_x.detach().cpu().numpy()
        source_y_plot = source_y.detach().cpu().numpy()
        sensor_x_plot = sensor_x.detach().cpu().numpy()
        sensor_y_plot = sensor_y.detach().cpu().numpy()
        self.ax1.set_xlim([-26, 26])
        self.ax1.set_ylim([-26, 26])
        self.ax1.set_aspect('equal')
        self.ax1.scatter(source_x_plot, source_y_plot, color='cyan', label='Sources', marker='x')
        sensor_markers_0 = self.ax1.scatter(sensor_x_plot,sensor_y_plot, color='red', marker='*', label='Initial Guess')
        # Initialize trajectory storage
        trajectories_x = [[] for _ in range(P)]
        trajectories_y = [[] for _ in range(P)]
        trajectory_lines = [self.ax1.plot([], [], linestyle='-', color='green', alpha=0.8)[0] for _ in range(P)]
        # Plot initial sensor positions
        x_coords = x_sensor.detach().cpu().numpy()
        y_coords = y_sensor.detach().cpu().numpy()
        sensor_markers = self.ax1.scatter(x_coords, y_coords, color='blue', marker='^', label='Sensors')
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
                Phi_T = TwoDimenGauPlumeM_AllSource_Reading_VEC(
                                x_sensor.to(device), y_sensor.to(device),
                                source_location_x.to(device), source_location_y.to(device),
                                Wr_x_batch.to(device), Wr_y_batch.to(device), w_speed_batch.to(device),
                                theta_true_batch.to(device), K.to(device), H.to(device),
                                sensor_noise_batch.to(device))
                Phi = torch.zeros((N_sensors, batch_size)).to(device)
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
                        x_sensor = torch.clamp(x_sensor, min=-25.0, max=25.0)
                        y_sensor = torch.clamp(y_sensor, min=-25.0, max=25.0)

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
                    temp_Gx = temp_Gx.to(x_sensor.device)
                    x_sensor = x_sensor - lr_outer * temp_Gx
                    temp_Gy = temp_Gy.to(y_sensor.device)
                    y_sensor = y_sensor - lr_outer * temp_Gy
                    x_sensor = torch.clamp(x_sensor, min=-25.0, max=25.0)
                    y_sensor = torch.clamp(y_sensor, min=-25.0, max=25.0)

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
            # Update coordinates
            x_coords = x_sensor.detach().cpu().numpy()
            y_coords = y_sensor.detach().cpu().numpy()
            # Append to trajectory and update plot
            for j in range(P):
                trajectories_x[j].append(x_coords[j])
                trajectories_y[j].append(y_coords[j])
                trajectory_lines[j].set_data(trajectories_x[j], trajectories_y[j])
            # Update sensor marker positions
            sensor_xy_np = np.column_stack((x_coords, y_coords))
            sensor_markers.set_offsets(sensor_xy_np)
            # Refresh canvas
            self.canvas.draw()
            self.canvas.get_tk_widget().update()
            self.status_label.config(text=f"Iteration: {k + 1} / {n_k}")
            self.progress["value"] = k + 1

            # # Ensure tensors are on CPU before plotting
            # x_sensor_plot = x_sensor.detach().cpu().numpy()
            # y_sensor_plot = y_sensor.detach().cpu().numpy()
            # # Plot sensor locations
            # sensor_markers.remove()
            # sensor_markers = self.ax1.scatter(x_sensor_plot, y_sensor_plot, color='blue', marker='^', label='Sensors')
            # # Add legend and labels (if needed)
            # self.ax1.set_title("Sensor and Source Locations")
            # self.ax1.set_xlabel("X")
            # self.ax1.set_ylabel("Y")
            # self.ax1.legend()
            # # Draw the canvas
            # self.canvas.draw()

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
