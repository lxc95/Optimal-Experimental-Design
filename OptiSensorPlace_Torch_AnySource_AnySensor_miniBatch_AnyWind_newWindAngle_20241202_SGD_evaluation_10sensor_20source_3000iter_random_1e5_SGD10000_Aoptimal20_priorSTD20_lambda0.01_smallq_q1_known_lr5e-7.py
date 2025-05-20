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

# User-defined Inputs:

# source_location_x = [-15., -10., -9., -5., 5., 5., 8., 10., 15., 20.]  # X axis
# source_location_y = [17., -5., 22., 10., 18., 0., -10., 19., -10, 5.]  # Y axis
source_location_x = 40*torch.rand(100)-20  # X axis
source_location_y = 40*torch.rand(100)-20  # Y axis

N_sources = len(source_location_x)
# Define the mean and variance of emission rates for truncated Gaussian distribution
# mean = [8., 10., 9., 8., 10., 9., 8., 10., 9., 10.]  # the mean of emission rates for the above sources
mean = 2*torch.rand(100) + 8  # the mean of emission rates for the above sources
sigma_pior_abs = 20
cov = sigma_pior_abs * torch.eye(N_sources)  # the covariance of these emission rates
# Define the height of stacks and the eddy diffusion coefficient
H = 2 * torch.ones(N_sources)  # the height of stacks
K = 0.4 * torch.ones(N_sources)  # the eddy diffusion coefficient, which is simplified
# Define the number of Monte Carlo samples
N_samples = 10
N_samples_large = 1
# Define the number of sensors
N_sensors = 50
# Define the sensor noise level -> the standard deviation
sigma_epsilon = 0.01
# the wind condition
ws_lower = 1  # the upper bound of wind speed
ws_upper = 2  # the lower bound of wind speed
wd_lower = 225  # the upper bound of wind direction
wd_upper = 315  # the lower bound of wind direction
# lambda
lambda_1 = 1. / 100.
lambda_2 = 1. / 100.

# Here are some advanced setting parameters: (You can change those parameters for better performance)
Num_iteration_k = 1000  # for the maximum iteration steps for the outer iteration
Num_SGD_BatchSize = N_samples  # for the batch size in SGD
Num_Backtracking = 0  # for adapting the outer iteration step size (or called learning rate)
lr_outer_initial = 0.000001  # for the initial learning rate of the backtracking
tol_G = 1e-10  # the tolerance for step length (i.e., the absolution value of the gradients of the outer iteration)
q = 20  # the maximum iteration number of inner loop
lr_inner_initial = 0.0005

# 'objective' is the function for A-optimal design, which can provide initial guess or can be compared with our
# solutions
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
    print('time C:', end-start)
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
    print('time inner 11:',end22 - start)
    print('time inner 22:',end33 - end22)
    print('time inner 221:',end221 - end22)
    print('time inner 222:',end223 - end221)

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
    print('time inner 12:',time3 - time2)
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
    
    print('time00: ',end-start)
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
# all_source = torch.transpose(torch.stack((source_location_x, source_location_y)),0,1)
# # x = np.random.randn(100, 2) / 6
# # x = torch.from_numpy(x)
# from kmeans_pytorch import kmeans, kmeans_predict
# # set device
# if torch.cuda.is_available():
#     device = torch.device('cuda:0')
# else:
#     device = torch.device('cpu')
# cluster_ids_x, cluster_centers = kmeans(
#     X=all_source, num_clusters=N_sensors, distance='euclidean', device=device
# )
# print('test:',cluster_centers)

# x_sensor = cluster_centers[:,0]
# y_sensor = cluster_centers[:,1]
##################################

initial = np.loadtxt("initial_cumu_SP_20_10.txt", delimiter=',') 
x_sensor = torch.from_numpy(initial[:,0]).to(torch.float32)
y_sensor = torch.from_numpy(initial[:,1]).to(torch.float32)



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
for k in range(n_k):
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
                print('time1:',end - start)

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
                print('time3:',end - start)
                if count_while == Num_Backtracking:
                    break
                count_while += 1
                end = timer()
                

            all_sensor_x[k * num_batch + count_temp, :] = x_sensor
            all_sensor_y[k * num_batch + count_temp, :] = y_sensor
            stepsize_x[k * num_batch + count_temp, :] = temp_Gx  # check the step size of sensor locations
            stepsize_y[k * num_batch + count_temp, :] = temp_Gy

            start = timer()
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
            print('time4:',end - start)

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

    print('Progress: k=', k, ' / ', n_k - 1)
    # print('Objective: ', Theta_error_norm_step_k_eval[k])
    print('Objective: ', Theta_error_norm_step_k[k])
    if (torch.all(torch.abs(temp_Gx) < tol_G) and torch.all(torch.abs(temp_Gy) < tol_G)) and k > 1:
        if torch.all(0 < torch.abs(temp_Gx)) and torch.all(0 < torch.abs(temp_Gy)):
            print('the solution has converged!')
            break
end = time.time()
print('the total calculation time is {:.4f} s'.format(end - start))  # the computational time

print('The trajectory of the X value for the 1st sensor:', all_sensor_x[:, 0])
print('The trajectory of the Y value for the 1st sensor:', all_sensor_y[:, 0])
print('The gradients of objective w.r.t X:', stepsize_x)
print('The gradients of objective w.r.t Y:', stepsize_y)
# print('The objective values:', Theta_error_norm_step_k)
# print(theta_esti_all)
# print(theta_esti_monitor)
print('The learning rates at each step:', step_alpha)

# # plot the concentration field
# C = torch.zeros((len(x), len(y)))
# mk = 0  # the index of the sample
# for i in range(len(x)):
#     for j in range(len(y)):
#         for k in range(N_sources):
#             C[i, j] += TwoDimenGauPlumeM(x[i], y[j], source_location_x[k], source_location_y[k], Wr_x[mk], Wr_y[mk],
#                                          w_speed[mk], theta_true_all[mk, k], K[k], H[k])
#
# xx, yy = torch.meshgrid(x, y)
# plt.figure()
# # plt.ion()
# plt.pcolor(xx, yy, C.T, cmap='jet')
# # plt.clim((0, 1e2));
# # plt.title(stability_str + '\n' + wind_dir_str);
# plt.xlabel('x')
# plt.ylabel('y')
# cb1 = plt.colorbar()
# # cb1.set_label('$\mu$ g m$^{-3}$');
# #plt.show()

# figure 1
fig = plt.figure(figsize=(14,13))
l1 = plt.scatter(source_location_x, source_location_y, marker='x', s=50, color='cyan')
for i in range(N_sensors):
    plt.scatter(all_sensor_x[:, i], all_sensor_y[:, i], marker='.', color='green')
l2 = plt.scatter(x_sensor, y_sensor, marker='^', s=50, color='blue')
l3 = plt.scatter(all_sensor_x[0, :], all_sensor_y[0, :], marker='*', s=50, color='red')
plt.xlim([-28, 28])
plt.ylim([-28, 28])
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('X', fontsize=24)
plt.ylabel('Y', fontsize=24)
plt.legend((l1, l2, l3), ('emission sources', 'final sensor locations', 'initial sensor locations'), fontsize = 24,
           bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', ncols=2)
# plt.show()
plt.savefig(str(N_sensors)+'_Sensors_1_SGD100_ite'+str(Num_iteration_k)+'_STD20_Aoptimal2_lr0.01_20source_0928.png')

# figure 2
fig = plt.figure(figsize=(14,13))
l1 = plt.scatter(source_location_x, source_location_y, marker='x', s=50, color='cyan')
plt.plot(all_sensor_x, all_sensor_y, color='green')
l2 = plt.scatter(x_sensor, y_sensor, marker='^', s=50, color='blue')
l3 = plt.scatter(all_sensor_x[0, :], all_sensor_y[0, :], marker='*', s=50, color='red')
plt.xlim([-28, 28])
plt.ylim([-28, 28])
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('X', fontsize=24)
plt.ylabel('Y', fontsize=24)
plt.legend((l1, l2, l3), ('emission sources', 'final sensor locations', 'initial sensor locations'), fontsize=24,
           bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', ncols=2)
# plt.show()
plt.savefig(str(N_sensors)+'_Sensors_2_SGD100_ite'+str(Num_iteration_k)+'_STD20_Aoptimal2_lr0.01_20source_0928.png')


# figure 3
fig = plt.figure(figsize=(14,13))
# plot without contour
l1 = plt.plot(source_location_x, source_location_y, marker='x', markersize=10, linestyle='None')
l2 = plt.plot(x_sensor, y_sensor, marker='^', markersize=10, linestyle='None')
plt.xlim([-28, 28])
plt.ylim([-28, 28])
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('X', fontsize=24)
plt.ylabel('Y', fontsize=24)
# plt.show()
plt.savefig(str(N_sensors)+'_Sensors_final_SGD100_ite'+str(Num_iteration_k)+'_STD20_Aoptimal2_lr0.01_20source_0928.png')

# figure 4
fig = plt.figure(figsize=(14,13))
# plot the step
for i in range(N_sensors):
    plt.plot(stepsize_x[:, i])
    plt.plot(stepsize_y[:, i])
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('the index of iteration', fontsize=24)
plt.ylabel('the gradient of objective w.r.t sensor locations', fontsize=24)
# plt.show()
plt.savefig(str(N_sensors)+'_Sensors_gradient_SGD100_ite'+str(Num_iteration_k)+'_STD20_Aoptimal2_lr0.01_20source_0928.png')

# figure 5
fig = plt.figure(figsize=(14,13))
for i in range(N_sensors):
    plt.plot(step_all_x[:, i])
    plt.plot(step_all_y[:, i])
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('the index of iteration', fontsize=24)
plt.ylabel('the actual updates of sensor locations', fontsize=24)
# plt.show()
plt.savefig(str(N_sensors)+'_Sensors_update_SGD100_ite'+str(Num_iteration_k)+'_STD20_Aoptimal2_lr0.01_20source_0928.png')

# figure 6
fig = plt.figure(figsize=(14,13))
# plot the objective value to show convergence
plt.plot(Theta_error_norm_step_k)
# plt.plot(Theta_error_norm_step_k_eval)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel('the index of iteration', fontsize=24)
plt.ylabel('the objective value', fontsize=24)
# plt.show()
plt.savefig(str(N_sensors)+'_Sensors_convergence_SGD100_ite'+str(Num_iteration_k)+'_STD20_Aoptimal2_lr0.01_20source_0928.png')

def angle_Windrose(all_angle):
    converted = torch.zeros(len(all_angle))
    for i in range(len(all_angle)):
        if 0 <= all_angle[i] < 270:
            converted[i] = 270-all_angle[i]
        elif 270 <= all_angle[i] <= 360:
            converted[i] = 360- (all_angle[i]-270)
    return converted


# # figure 7
# ws2 = torch.random.uniform(ws_lower, ws_upper, 100000)  # the wind speed distribution
# wd2 = torch.random.uniform(wd_lower, wd_upper, 100000)  # the wind speed distribution
# ax = windrose.WindroseAxes.from_ax()
# ax.bar(angle_Windrose(wd2), ws2, normed=True, opening=0.8, edgecolor='white', bins=torch.arange(1, 2, 0.2))
# # ax.box(angle_Windrose(wd2), ws2, bins=torch.arange(1, 2, 0.1))
# # ax.set_legend(title="Wind speed", fontsize=16)
# ax.set_legend(title="Wind speed", fontsize=16, loc=4, bbox_to_anchor=(1., -0.07))
# for t in ax.get_xticklabels():
#     plt.setp(t, fontsize=20)
# plt.title("the wind profile", fontsize=20)
# # plt.show()
# plt.savefig(str(N_sensors)+'_Sensors_windrose_SGD100_ite'+str(Num_iteration_k)+'_STD20_Aoptimal2_lr0.01_20source_0928.png')

# # save the datset
# torch.savetxt(str(N_sensors)+"_evaluated_obj_SGD100_ite"+str(Num_iteration_k)+"_STD20_Aoptimal2_lr0.01_20source_0928.csv", Theta_error_norm_step_k_eval)
# torch.savetxt(str(N_sensors)+"_sensor_X_SGD100_ite"+str(Num_iteration_k)+"_STD20_Aoptimal2_lr0.01_20source_0928.csv", all_sensor_x)
# torch.savetxt(str(N_sensors)+"_sensor_Y_SGD100_ite"+str(Num_iteration_k)+"_STD20_Aoptimal2_lr0.01_20source_0928.csv", all_sensor_y)