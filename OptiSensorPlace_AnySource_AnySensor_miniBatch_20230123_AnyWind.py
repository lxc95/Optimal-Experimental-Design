import numpy as nmp
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import time
from joblib import Parallel, delayed
import os
from scipy import optimize
import cvxopt
import quadprog
from scipy import stats
import pyscipopt

nmp.random.seed(0)

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
    B. Outputs:
        1. The final designs of all required sensors.
        2. The trajectory of sensors' locations during iterations and the corresponding objective values which is 
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

source_location_x = [-15., -10., -9., -5., 5., 5., 8., 10., 15., 20.]  # X axis
source_location_y = [17., -5., 22., 10., 18., 0., -10., 19., -10, 5.]  # Y axis
N_sources = len(source_location_x)
# Define the mean and variance of emission rates for truncated Gaussian distribution
mean = [8., 10., 9., 8., 10., 9., 8., 10., 9., 10.]  # the mean of emission rates for the above sources
sigma_pior_abs = 8
cov = sigma_pior_abs * nmp.eye(N_sources)  # the covariance of these emission rates
# Define the height of stacks and the eddy diffusion coefficient
H = 2 * nmp.ones(N_sources)  # the height of stacks
K = 0.4 * nmp.ones(N_sources)  # the eddy diffusion coefficient, which is simplified
# Define the number of Monte Carlo samples
N_samples = 1000
# Define the number of sensors
N_sensors = 3
# Define the sensor noise level -> the standard deviation
sigma_epsilon = 0.01
# the wind condition
ws_lower = 1  # the upper bound of wind speed
ws_upper = 2  # the lower bound of wind speed
wd_lower = 0.511  # the upper bound of wind direction
wd_upper = 0.512  # the lower bound of wind direction
# lambda
lambda_1 = 1. / 100.
lambda_2 = 1. / 100.

# Here are some advanced setting parameters: (You can change those parameters for better performance)
Num_iteration_k = 2000  # for the maximum iteration steps for the outer iteration
Num_SGD_BatchSize = 1000  # for the batch size in SGD
Num_Backtracking = 10  # for adapting the outer iteration step size (or called learning rate)
lr_outer_initial = 0.025  # for the initial learning rate of the backtracking
tol_G = 1e-10  # the tolerance for step length (i.e., the absolution value of the gradients of the outer iteration)


# 'objective' is the function for A-optimal design, which can provide initial guess or can be compared with our
# solutions
def objective(v):
    num_sensor = N_sensors
    x_all = v[0:num_sensor]
    y_all = v[num_sensor:(2*num_sensor)]
    source_x = source_location_x
    source_y = source_location_y
    num_source = len(source_x)
    # Define the mean and variance of emission rates
    sigma_pr = sigma_pior_abs
    # number of monte carlo samplings for A-optimal design
    N = 5
    temp = 0
    for nk in range(N):
        ws = nmp.random.uniform(ws_lower, ws_upper, 1)  # the wind speed distribution
        wd = nmp.random.uniform(wd_lower, wd_upper, 1) * 360  # the wind angle distribution
        w_x = nmp.cos((450. - wd) / 180. * nmp.pi)  # the x part of unit wind vector
        w_y = nmp.sin((450. - wd) / 180. * nmp.pi)  # the y part of unit wind vector
        u = nmp.abs(ws)  # the wind speed
        # the F operator
        A_all = nmp.zeros((num_sensor, num_source))
        for i in range(num_sensor):
            for j in range(num_source):
                x_new = nmp.sqrt(
                    ((1. - w_x ** 2.) * (x_all[i] - source_x[j]) - w_x * w_y * (y_all[i] - source_y[j])) ** 2. + (
                            -w_x * w_y * (x_all[i] - source_x[j]) + (1. - w_y ** 2.) * (y_all[i] - source_y[j])) ** 2.)
                y_new = w_x * (x_all[i] - source_x[j]) + w_y * (y_all[i] - source_y[j])
                if y_new > 0.:
                    A_all[i, j] = 1 / (2. * nmp.pi * K[j] * y_new) * nmp.exp(
                        -u * (x_new ** 2. + H[j] ** 2.) / (4. * K[j] * y_new))
                else:
                    A_all[i, j] = 0.
        # the posterior covariance
        cov_post = nmp.linalg.inv(1 / sigma_epsilon ** 2 * A_all.T @ A_all + 1 / sigma_pr ** 2 * nmp.eye(num_source))
        temp += nmp.linalg.norm(1/sigma_pr*cov_post) + nmp.linalg.norm(1/sigma_epsilon*cov_post@A_all.T)
    return temp/N


def TwoDimenGauPlumeM(x_tem, y_tem, source_x, source_y, w_x, w_y, u, Q_tem, K_tem, H_tem):
    x_new = nmp.sqrt(((1. - w_x ** 2.) * (x_tem - source_x) - w_x * w_y * (y_tem - source_y)) ** 2. + (
            -w_x * w_y * (x_tem - source_x) + (1. - w_y ** 2.) * (y_tem - source_y)) ** 2.)
    y_new = w_x * (x_tem - source_x) + w_y * (y_tem - source_y)
    if y_new > 0.:
        Pi = Q_tem / (2. * nmp.pi * K_tem * y_new) * nmp.exp(-u * (x_new ** 2. + H_tem ** 2.) / (4. * K_tem * y_new))
    else:
        Pi = 0.
    return Pi


def TwoDimenGauPlumeM_AllSource_Reading(x_tem, y_tem, source_x, source_y, w_x, w_y, u, q_tem, K_tem, H_tem, noise):
    Phi_tem = 0
    for _ in range(len(source_x)):
        Phi_tem += TwoDimenGauPlumeM(x_tem, y_tem, source_x[_], source_y[_], w_x, w_y, u, q_tem[_], K_tem[_], H_tem[_])
    return Phi_tem + noise


def GradientInnerNew(x_all, y_all, source_x, source_y, w_x, w_y, u, K_tem, H_tem, Phi_tem, sigma_e, lambda_1_tem, lambda_2_tem):
    num_sensor = len(x_all)
    num_source = len(source_x)
    A_all = nmp.zeros((num_sensor, num_source))
    for i in range(num_sensor):
        for j in range(num_source):
            x_new = nmp.sqrt(
                ((1. - w_x ** 2.) * (x_all[i] - source_x[j]) - w_x * w_y * (y_all[i] - source_y[j])) ** 2. + (
                        -w_x * w_y * (x_all[i] - source_x[j]) + (1. - w_y ** 2.) * (y_all[i] - source_y[j])) ** 2.)
            y_new = w_x * (x_all[i] - source_x[j]) + w_y * (y_all[i] - source_y[j])
            if y_new > 0.:
                A_all[i, j] = 1 / (2. * nmp.pi * K_tem[j] * y_new) * nmp.exp(
                    -u * (x_new ** 2. + H_tem[j] ** 2.) / (4. * K_tem[j] * y_new))
            else:
                A_all[i, j] = 0.
    C_coef = 1. / (sigma_e ** 2) * (A_all.T @ A_all) + lambda_1_tem * nmp.identity(num_source)
    if num_sensor == 1:
        D_coef_T = lambda_2_tem * nmp.ones((num_source, 1)) - 1. / (sigma_e ** 2) * A_all.T * Phi_tem
    else:
        D_coef_T = lambda_2_tem * nmp.ones((num_source, 1)) - 1. / (sigma_e ** 2) * A_all.T @ Phi_tem

    return [C_coef, D_coef_T]


def Gradient_AA_x(x_tem, y_tem, source_x1, source_y1, source_x2, source_y2, w_x, w_y, u, K1, H1, K2, H2):
    r_per1 = nmp.sqrt(((1. - w_x ** 2.) * (x_tem - source_x1) - w_x * w_y * (y_tem - source_y1)) ** 2. + (
            -w_x * w_y * (x_tem - source_x1) + (1. - w_y ** 2.) * (y_tem - source_y1)) ** 2.)
    r_para1 = w_x * (x_tem - source_x1) + w_y * (y_tem - source_y1)
    r_per2 = nmp.sqrt(((1. - w_x ** 2.) * (x_tem - source_x2) - w_x * w_y * (y_tem - source_y2)) ** 2. + (
            -w_x * w_y * (x_tem - source_x2) + (1. - w_y ** 2.) * (y_tem - source_y2)) ** 2.)
    r_para2 = w_x * (x_tem - source_x2) + w_y * (y_tem - source_y2)
    if r_para1 > 0 and r_para2 > 0:
        G_AA_x = -(1 / (4 * nmp.pi ** 2 * K1 * K2 * r_para1 * r_para2) ** 2) * 4 * nmp.pi ** 2 * K1 * K2 * (
                    w_x * (r_para1 + r_para2)) * nmp.exp(
            -u * (r_per1 ** 2 + H1 ** 2) / (4 * K1 * r_para1) - u * (r_per2 ** 2 + H2 ** 2) / (4 * K2 * r_para2)) \
                 + 1 / (4 * nmp.pi ** 2 * K1 * K2 * r_para1 * r_para2) * nmp.exp(
            -u * (r_per1 ** 2 + H1 ** 2) / (4 * K1 * r_para1) - u * (r_per2 ** 2 + H2 ** 2) / (4 * K2 * r_para2)) \
                 * ((-u * (2 * ((1 - w_x ** 2) * (x_tem - source_x1) - w_x * w_y * (y_tem - source_y1)) * (
                    1 - w_x ** 2) + 2 * (-w_x * w_y * (x_tem - source_x1) + (1 - w_y ** 2) * (y_tem - source_y1)) * (
                                       -w_x * w_y)) * 4 * K1 * r_para1 + u * (r_per1 ** 2 + H1 ** 2) * 4 * K1 * w_x) / (
                                4 * K1 * r_para1) ** 2 + (-u * (
                    2 * ((1 - w_x ** 2) * (x_tem - source_x2) - w_x * w_y * (y_tem - source_y2)) * (
                        1 - w_x ** 2) + 2 * (
                                -w_x * w_y * (x_tem - source_x2) + (1 - w_y ** 2) * (y_tem - source_y2)) * (
                                -w_x * w_y)) * 4 * K2 * r_para2 + u * (r_per2 ** 2 + H2 ** 2) * 4 * K2 * w_x) / (
                                4 * K2 * r_para2) ** 2)
    else:
        G_AA_x = 0
    return G_AA_x


def Gradient_AA_y(x_tem, y_tem, source_x1, source_y1, source_x2, source_y2, w_x, w_y, u, K1, H1, K2, H2):
    r_per1 = nmp.sqrt(((1. - w_x ** 2.) * (x_tem - source_x1) - w_x * w_y * (y_tem - source_y1)) ** 2. + (
            -w_x * w_y * (x_tem - source_x1) + (1. - w_y ** 2.) * (y_tem - source_y1)) ** 2.)
    r_para1 = w_x * (x_tem - source_x1) + w_y * (y_tem - source_y1)
    r_per2 = nmp.sqrt(((1. - w_x ** 2.) * (x_tem - source_x2) - w_x * w_y * (y_tem - source_y2)) ** 2. + (
            -w_x * w_y * (x_tem - source_x2) + (1. - w_y ** 2.) * (y_tem - source_y2)) ** 2.)
    r_para2 = w_x * (x_tem - source_x2) + w_y * (y_tem - source_y2)
    if r_para1 > 0 and r_para2 > 0:
        G_AA_y = -(1 / (4 * nmp.pi ** 2 * K1 * K2 * r_para1 * r_para2) ** 2) * 4 * nmp.pi ** 2 * K1 * K2 * (
                    w_y * (r_para1 + r_para2)) * nmp.exp(
            -u * (r_per1 ** 2 + H1 ** 2) / (4 * K1 * r_para1) - u * (r_per2 ** 2 + H2 ** 2) / (4 * K2 * r_para2)) \
                 + 1 / (4 * nmp.pi ** 2 * K1 * K2 * r_para1 * r_para2) * nmp.exp(
            -u * (r_per1 ** 2 + H1 ** 2) / (4 * K1 * r_para1) - u * (r_per2 ** 2 + H2 ** 2) / (4 * K2 * r_para2)) \
                 * ((-u * (
                    2 * ((1 - w_x ** 2) * (x_tem - source_x1) - w_x * w_y * (y_tem - source_y1)) * (-w_x * w_y) + 2 * (
                        -w_x * w_y * (x_tem - source_x1) + (1 - w_y ** 2) * (y_tem - source_y1)) * (
                                1 - w_y ** 2)) * 4 * K1 * r_para1 + u * (r_per1 ** 2 + H1 ** 2) * 4 * K1 * w_y) / (
                                4 * K1 * r_para1) ** 2 + (-u * (
                    2 * ((1 - w_x ** 2) * (x_tem - source_x2) - w_x * w_y * (y_tem - source_y2)) * (-w_x * w_y) + 2 * (
                        -w_x * w_y * (x_tem - source_x2) + (1 - w_y ** 2) * (y_tem - source_y2)) * (
                                1 - w_y ** 2)) * 4 * K2 * r_para2 + u * (r_per2 ** 2 + H2 ** 2) * 4 * K2 * w_y) / (
                                4 * K2 * r_para2) ** 2)
    else:
        G_AA_y = 0
    return G_AA_y


def Gradient_A_x(x_tem, y_tem, source_x1, source_y1, w_x, w_y, u, K_tem, H_tem):
    r_per1 = nmp.sqrt(((1. - w_x ** 2.) * (x_tem - source_x1) - w_x * w_y * (y_tem - source_y1)) ** 2. + (
            -w_x * w_y * (x_tem - source_x1) + (1. - w_y ** 2.) * (y_tem - source_y1)) ** 2.)
    r_para1 = w_x * (x_tem - source_x1) + w_y * (y_tem - source_y1)
    if r_para1 > 0:
        G_A_x = -(1 / (2 * nmp.pi * K_tem * r_para1) ** 2) * 2 * nmp.pi * K_tem * w_x * nmp.exp(
            -u * (r_per1 ** 2 + H_tem ** 2) / (4 * K_tem * r_para1)) \
                + 1 / (2 * nmp.pi * K_tem * r_para1) * nmp.exp(-u * (r_per1 ** 2 + H_tem ** 2) / (4 * K_tem * r_para1)) \
                * ((-u * (2 * ((1 - w_x ** 2) * (x_tem - source_x1) - w_x * w_y * (y_tem - source_y1)) * (
                    1 - w_x ** 2) + 2 * (-w_x * w_y * (x_tem - source_x1) + (1 - w_y ** 2) * (y_tem - source_y1)) * (
                                      -w_x * w_y)) * 4 * K_tem * r_para1 + u * (
                                r_per1 ** 2 + H_tem ** 2) * 4 * K_tem * w_x) / (4 * K_tem * r_para1) ** 2)
    else:
        G_A_x = 0
    return G_A_x


def Gradient_A_y(x_tem, y_tem, source_x1, source_y1, w_x, w_y, u, K_tem, H_tem):
    r_per1 = nmp.sqrt(((1. - w_x ** 2.) * (x_tem - source_x1) - w_x * w_y * (y_tem - source_y1)) ** 2. + (
            -w_x * w_y * (x_tem - source_x1) + (1. - w_y ** 2.) * (y_tem - source_y1)) ** 2.)
    r_para1 = w_x * (x_tem - source_x1) + w_y * (y_tem - source_y1)
    if r_para1 > 0:
        G_A_y = -(1 / (2 * nmp.pi * K_tem * r_para1) ** 2) * 2 * nmp.pi * K_tem * w_y * nmp.exp(
            -u * (r_per1 ** 2 + H_tem ** 2) / (4 * K_tem * r_para1)) \
                + 1 / (2 * nmp.pi * K_tem * r_para1) * nmp.exp(-u * (r_per1 ** 2 + H_tem ** 2) / (4 * K_tem * r_para1)) \
                * ((-u * (
                    2 * ((1 - w_x ** 2) * (x_tem - source_x1) - w_x * w_y * (y_tem - source_y1)) * (-w_x * w_y) + 2 * (
                        -w_x * w_y * (x_tem - source_x1) + (1 - w_y ** 2) * (y_tem - source_y1)) * (
                                1 - w_y ** 2)) * 4 * K_tem * r_para1 + u * (
                                r_per1 ** 2 + H_tem ** 2) * 4 * K_tem * w_y) / (4 * K_tem * r_para1) ** 2)
    else:
        G_A_y = 0
    return G_A_y


def GradientOuterNew(x_all, y_all, source_x, source_y, w_x, w_y, u, K_tem, H_tem, Phi_tem, sigma_e, lambda_1_tem,
                     theta_curr):
    num_sensor = len(x_all)
    num_source = len(source_x)
    A_all = nmp.zeros((num_sensor, num_source))
    for i in range(num_sensor):
        for j in range(num_source):
            x_new = nmp.sqrt(
                ((1. - w_x ** 2.) * (x_all[i] - source_x[j]) - w_x * w_y * (y_all[i] - source_y[j])) ** 2. + (
                        -w_x * w_y * (x_all[i] - source_x[j]) + (1. - w_y ** 2.) * (y_all[i] - source_y[j])) ** 2.)
            y_new = w_x * (x_all[i] - source_x[j]) + w_y * (y_all[i] - source_y[j])
            if y_new > 0.:
                A_all[i, j] = 1 / (2. * nmp.pi * K_tem[j] * y_new) * nmp.exp(
                    -u * (x_new ** 2. + H_tem[j] ** 2.) / (4. * K_tem[j] * y_new))
            else:
                A_all[i, j] = 0.
    C_coef = 1. / (sigma_e ** 2) * (A_all.T @ A_all) + lambda_1_tem * nmp.identity(num_source)
    G_x = nmp.zeros((num_sensor, num_source))
    G_y = nmp.zeros((num_sensor, num_source))
    for mk in range(num_sensor):
        for i in range(num_source):
            for j in range(num_source):
                G_x[mk, i] += 1. / (sigma_e ** 2) * Gradient_AA_x(x_all[mk], y_all[mk], source_x[i], source_y[i], source_x[j], source_y[j],
                                            w_x, w_y, u, K_tem[i], H_tem[i], K_tem[j], H_tem[j]) * theta_curr[j]
                G_y[mk, i] += 1. / (sigma_e ** 2) * Gradient_AA_y(x_all[mk], y_all[mk], source_x[i], source_y[i], source_x[j], source_y[j],
                                            w_x, w_y, u, K_tem[i], H_tem[i], K_tem[j], H_tem[j]) * theta_curr[j]
            G_x[mk, i] -= 1. / (sigma_e ** 2) * Gradient_A_x(x_all[mk], y_all[mk], source_x[i], source_y[i], w_x, w_y, u, K_tem[i],
                                       H_tem[i]) * Phi_tem[mk]
            G_y[mk, i] -= 1. / (sigma_e ** 2) * Gradient_A_y(x_all[mk], y_all[mk], source_x[i], source_y[i], w_x, w_y, u, K_tem[i],
                                       H_tem[i]) * Phi_tem[mk]
    print('G_x:', G_x)
    # note that we used the linear regression package as the linear solver
    # it is better to develop our own linear solver, which should be efficient
    coef_x = nmp.zeros((num_sensor, num_source))
    coef_y = nmp.zeros((num_sensor, num_source))
    for i in range(num_sensor):
        G_x_T = nmp.zeros(num_source)
        G_x_T[:] = nmp.array(G_x[i, :])
        if nmp.any(nmp.isnan(G_x_T)):
            coef_x[i, :] = nmp.zeros(num_source)
        else:
            reg1 = nmp.linalg.solve(-nmp.array(C_coef), G_x_T)
            coef_x[i, :] = reg1
        G_y_T = nmp.zeros(num_source)
        G_y_T[:] = nmp.array(G_y[i, :])
        if nmp.any(nmp.isnan(G_y_T)):
            coef_y[i, :] = nmp.zeros(num_source)
        else:
            reg2 = nmp.linalg.solve(-nmp.array(C_coef), G_y_T)
            coef_y[i, :] = reg2
        if nmp.sum(coef_x[i, :]) > 1000000000. or nmp.sum(
                coef_y[i, :]) > 1000000000.:  # to avoid the abnormal results of linear solver
            coef_x[i, :] = nmp.zeros(num_source)
            coef_y[i, :] = nmp.zeros(num_source)
    # reg.coef_
    print('coef_x:', coef_x)
    return [coef_x, coef_y]


def Inner_loop(q_tem, C, D_T, N_sources_tem, para_lr_inner, lr_inner, theta_true):
    theta_esti_all_tem = nmp.zeros((q_tem, N_sources_tem))
    Gradient_InerSize_All = nmp.zeros((q_tem, N_sources_tem))
    # Set the initial guess of theta
    theta_esti_all_tem[0, :] = nmp.zeros((1, N_sources_tem))  # Here we start from zeros
    for j in range(q_tem):
        Gradient_InerSize_All[j, :] = nmp.matmul(C, theta_esti_all_tem[j, :]) + D_T.T
        if j > para_lr_inner:
            theta_esti_all_tem[j, :] = theta_esti_all_tem[j, :] - lr_inner / nmp.sqrt(j + 1) * Gradient_InerSize_All[j, :]
        else:
            theta_esti_all_tem[j, :] = theta_esti_all_tem[j, :] - lr_inner * Gradient_InerSize_All[j, :]
        for mk in range(N_sources_tem):
            # projected to be non-negative
            theta_esti_all_tem[j, mk] = max(0, theta_esti_all_tem[j, mk])
    return theta_esti_all_tem[q_tem - 1, :]


def cvxopt_solve_qp(P, q_tem, G, h, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q_tem)]
    args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return nmp.array(sol['x']).reshape((P.shape[1],))


def quadprog_solve_qp(P, q_tem, G, h, A=None, b=None):
    qp_G = .5 * (P + P.T)  # make sure P is symmetric
    qp_a = -q_tem
    if A is not None:
        qp_C = -nmp.vstack([A, G]).T
        qp_b = -nmp.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def Update_Inner_OuterStep(x_sensor_tem, y_sensor_tem, source_location_x_tem, source_location_y_tem, Wr_x_tem, Wr_y_tem,
                           w_speed_tem,
                           K_tem, H_tem, Phi_input, sigma_epsilon_tem, lambda_1_tem, lambda_2_tem, theta_true_tem, N_sources_tem, q):
    num_sensors = len(Phi_input)
    Phi_tem = nmp.zeros((num_sensors, 1))
    Phi_tem[:, 0] = Phi_input
    theta_esti_all_tem = nmp.zeros((1, N_sources_tem))
    theta_error_all = nmp.zeros((1, N_sources_tem))
    [C, D_T] = GradientInnerNew(x_sensor_tem, y_sensor_tem, source_location_x_tem, source_location_y_tem, Wr_x_tem,
                                Wr_y_tem, w_speed_tem,
                                K_tem, H_tem, Phi_tem, sigma_epsilon_tem, lambda_1_tem, lambda_2_tem)
    # Set the initial guess of theta
    theta_esti_all_tem[0, :] = nmp.zeros((1, N_sources_tem))  # Here we start from zeros
    lr_inner = 0.5
    para_lr_inner = 2000000  # now we ignore this parameter by setting a large number
    # call the inner loop
    # theta_esti_all[0, :] = Inner_loop(q, C, D_T, N_sources, para_lr_inner, lr_inner, theta_true_all[0, :])
    theta_esti_all_tem[0, :] = cvxopt_solve_qp(C, D_T, -1 * nmp.eye(N_sources_tem), nmp.zeros(N_sources_tem))
    # theta_esti_all_tem[0, :] = cvxopt_solve_qp(C, D_T, -1 * nmp.zeros((N_sources_tem, N_sources_tem)), 1000+nmp.zeros(N_sources_tem))
    [Gradient_outerAll_x, Gradient_outerAll_y] = GradientOuterNew(x_sensor_tem, y_sensor_tem, source_location_x_tem,
                                                                  source_location_y_tem, Wr_x_tem, Wr_y_tem,
                                                                  w_speed_tem, K_tem, H_tem,
                                                                  Phi_tem, sigma_epsilon_tem, lambda_1_tem, theta_esti_all_tem[0, :])
    theta_error_all[0, :] = theta_esti_all_tem[0, :] - theta_true_tem
    Gradient_OuterSize_All_x_tem = nmp.matmul(Gradient_outerAll_x, theta_error_all[0, :])
    Gradient_OuterSize_All_y_tem = nmp.matmul(Gradient_outerAll_y, theta_error_all[0, :])
    return [Gradient_OuterSize_All_x_tem, Gradient_OuterSize_All_y_tem, theta_error_all[0, :], theta_esti_all_tem[0, :]]


# main
# start the local solver to fine-tune the results above
start = time.time()
cpu_count = os.cpu_count() + 1
# the domain of the concentration field
x = nmp.linspace(-25, 25, 200)
y = nmp.linspace(-25, 25, 200)

# start the global solver to get the initial solution
# bounds = nmp.array([[-25, 25], [-25, 25], [-25, 25], [-25, 25], [-25, 25], [-25, 25], [-25, 25], [-25, 25], [-25, 25], [-25, 25]])
# bounds = nmp.array([[-25, 25], [-25, 25]])
bounds = []
min_max = [-25, 25]
for ik in range(2*N_sensors):
    bounds.append(min_max)
result = optimize.dual_annealing(objective, nmp.array(bounds))
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('The initial solution by global solver: f(%s) = %.5f' % (solution, evaluation))

# the SGD-based Bi-level approximation method
# Initialize the locations of sensors
x_sensor = solution[0:N_sensors]
y_sensor = solution[N_sensors:]

# x_sensor = [0]
# y_sensor = [0]

n_k = Num_iteration_k  # the maximum iteration number of outer loop
batch_size = Num_SGD_BatchSize  # the size of mini_batch
num_batch = int(N_samples / batch_size)
q = 10000  # the maximum iteration number of inner loop
random_state = None  # the seed to control the shuffling. Here we consider randomness by 'None'

# fixed samplings
ws = nmp.random.uniform(ws_lower, ws_upper, N_samples)  # the wind speed distribution
wd = nmp.random.uniform(wd_lower, wd_upper, N_samples) * 360  # the wind angle distribution
Wr_x = nmp.cos((450. - wd) / 180. * nmp.pi)  # the x part of unit wind vector
Wr_y = nmp.sin((450. - wd) / 180. * nmp.pi)  # the y part of unit wind vector
w_speed = nmp.abs(ws)  # the wind speed
# theta_true_all = nmp.abs(nmp.random.multivariate_normal(mean, cov, N_samples))
theta_true_all = nmp.zeros((N_samples, N_sources))
X = nmp.array(mean)
D = sigma_pior_abs * nmp.ones(N_sources)
bound1 = nmp.zeros(N_sources)
bound2 = float('inf') * nmp.ones(N_sources)
for i in range(N_samples):
    theta_true_all[i, :] = stats.truncnorm.rvs((bound1 - X) / D, (bound2 - X) / D, loc=X,
                                               scale=D)  # sample from the truncated normal distribution
sensor_noise_all = nmp.random.normal(0, sigma_epsilon, size=(N_samples, N_sensors))

# define the random setting for shuffling
all_samplings = nmp.c_[Wr_x, Wr_y, w_speed, theta_true_all, sensor_noise_all]  # combine all data together
seed = None if random_state is None else int(random_state)
rng = nmp.random.default_rng(seed=seed)

Theta_error_norm_step_k = nmp.zeros(
    n_k * num_batch) + nmp.inf  # here we add 100 to the inital settings for while loop requirements
print(Theta_error_norm_step_k)
theta_esti_monitor = nmp.zeros((n_k * num_batch, N_sources))
step_alpha = nmp.zeros(n_k * num_batch)
step_all_x = nmp.zeros((n_k * num_batch, N_sensors))
step_all_y = nmp.zeros((n_k * num_batch, N_sensors))
all_sensor_x = nmp.zeros((n_k * num_batch, N_sensors))
all_sensor_y = nmp.zeros((n_k * num_batch, N_sensors))
stepsize_x = nmp.zeros((n_k * num_batch, N_sensors))
stepsize_y = nmp.zeros((n_k * num_batch, N_sensors))
theta_esti_all = nmp.zeros((batch_size, N_sources))
Gradient_OuterSize_All_x = nmp.zeros((batch_size, N_sensors))
Gradient_OuterSize_All_y = nmp.zeros((batch_size, N_sensors))
for k in range(n_k):
    # # re-samplings
    # ws = nmp.random.normal(ws_mean, ws_std, N_samples)  # the wind angle distribution
    # wd = nmp.random.uniform(wd_lower, wd_upper, N_samples) * 360  # the wind speed distribution
    # Wr_x = nmp.cos((450. - wd) / 180. * nmp.pi)  # the x part of unit wind vector
    # Wr_y = nmp.sin((450. - wd) / 180. * nmp.pi)  # the y part of unit wind vector
    # w_speed = nmp.abs(ws)  # the wind speed
    # theta_true_all = nmp.abs(nmp.random.multivariate_normal(mean, cov, N_samples))
    # sensor_noise_all = nmp.random.normal(0, sigma_epsilon, size=(N_samples, N_sensors))

    # shuffle all the samplings
    rng.shuffle(all_samplings)

    # performing minibatch moves
    count_temp = 0
    for start_batch in range(0, N_samples, batch_size):
        stop_batch = start_batch + batch_size
        Wr_x_batch, Wr_y_batch, w_speed_batch, theta_true_batch, sensor_noise_batch = \
            all_samplings[start_batch:stop_batch, 0], all_samplings[start_batch:stop_batch, 1],\
                all_samplings[start_batch:stop_batch, 2], all_samplings[start_batch:stop_batch, 3:(3 + N_sources)],\
                all_samplings[start_batch:stop_batch, (3 + N_sources):]
        # Generate the sensor readings
        Phi = nmp.zeros((N_sensors, batch_size))
        for mj in range(batch_size):
            for i in range(N_sensors):
                Phi[i, mj] = nmp.abs(
                    TwoDimenGauPlumeM_AllSource_Reading(x_sensor[i], y_sensor[i], source_location_x, source_location_y,
                                                        Wr_x_batch[mj], Wr_y_batch[mj], w_speed_batch[mj],
                                                        theta_true_batch[mj, :], K, H,
                                                        sensor_noise_batch[mj, i]))
        # Start the update
        lr_outer = lr_outer_initial*2
        if k > 0:
            count_while = 0
            while (Theta_error_norm_step_k[k * num_batch + count_temp] - Theta_error_norm_step_k[
                k * num_batch + count_temp - 1]) > 0:
                lr_outer = lr_outer / 2.
                temp_Gx = 2. * nmp.mean(Gradient_OuterSize_All_x, 0)
                temp_Gy = 2. * nmp.mean(Gradient_OuterSize_All_y, 0)
                if nmp.any(nmp.isnan(temp_Gx)) or nmp.any(nmp.isnan(temp_Gy)):  # to avoid abnormal gradients
                    lr_outer = 0
                # update
                x_sensor = all_sensor_x[k * num_batch + count_temp - 1, :] - lr_outer * temp_Gx
                y_sensor = all_sensor_y[k * num_batch + count_temp - 1, :] - lr_outer * temp_Gy
                for mk in range(N_sensors):
                    x_sensor[mk] = max(-25, x_sensor[mk])
                    x_sensor[mk] = min(25, x_sensor[mk])
                    y_sensor[mk] = max(-25, y_sensor[mk])
                    y_sensor[mk] = min(25, y_sensor[mk])

                # note that this for-loop is implemented in parallel
                parallel = Parallel(n_jobs=3, prefer="processes")
                tempdata = parallel(
                    delayed(Update_Inner_OuterStep)(x_sensor, y_sensor, source_location_x, source_location_y,
                                                    Wr_x_batch[i], Wr_y_batch[i],
                                                    w_speed_batch[i],
                                                    K, H, Phi[:, i], sigma_epsilon, lambda_1, lambda_2,
                                                    theta_true_batch[i, :], N_sources, q) for i in range(batch_size))
                # Split the data from parallel computing
                Theta_error_norm_step_k[k * num_batch + count_temp] = 0
                for jk in range(batch_size):
                    temp_xy = tempdata[jk]
                    Theta_error_norm_step_k[k * num_batch + count_temp] += (nmp.linalg.norm(
                        temp_xy[2])) ** 2 / batch_size
                if count_while == Num_Backtracking:
                    break
                count_while += 1
            all_sensor_x[k * num_batch + count_temp, :] = x_sensor
            all_sensor_y[k * num_batch + count_temp, :] = y_sensor
            stepsize_x[k * num_batch + count_temp, :] = temp_Gx  # check the step size of sensor locations
            stepsize_y[k * num_batch + count_temp, :] = temp_Gy

            # Split the data from parallel computing
            Theta_error_norm_step_k[k * num_batch + count_temp] = 0
            for jk in range(batch_size):
                temp_xy = tempdata[jk]
                Gradient_OuterSize_All_x[jk, :] = temp_xy[0]
                Gradient_OuterSize_All_y[jk, :] = temp_xy[1]
                # Theta_error_norm_step_k[k] += nmp.linalg.norm(temp_xy[2])/N_samples/nmp.linalg.norm(theta_true_all[jk, :])
                Theta_error_norm_step_k[k * num_batch + count_temp] += (nmp.linalg.norm(temp_xy[2])) ** 2 / batch_size
                theta_esti_all[jk, :] = temp_xy[3]
            theta_esti_monitor[k * num_batch + count_temp, :] = temp_xy[3]
            step_alpha[k * num_batch + count_temp] = lr_outer
            step_all_x[k * num_batch + count_temp] = - lr_outer * temp_Gx
            step_all_y[k * num_batch + count_temp] = - lr_outer * temp_Gy
            print('the step size:', lr_outer)

        if k == 0:
            lr_outer = 0.
            temp_Gx = 2. * nmp.mean(Gradient_OuterSize_All_x, 0)
            temp_Gy = 2. * nmp.mean(Gradient_OuterSize_All_y, 0)
            if nmp.any(nmp.isnan(temp_Gx)) or nmp.any(nmp.isnan(temp_Gy)):  # to avoid abnormal gradients
                lr_outer = 0
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

            # note that this for-loop is implemented in parallel
            parallel = Parallel(n_jobs=3, prefer="processes")
            tempdata = parallel(
                delayed(Update_Inner_OuterStep)(x_sensor, y_sensor, source_location_x, source_location_y,
                                                Wr_x_batch[i], Wr_y_batch[i],
                                                w_speed_batch[i],
                                                K, H, Phi[:, i], sigma_epsilon, lambda_1, lambda_2,
                                                theta_true_batch[i, :], N_sources, q) for i in range(batch_size))
            # Split the data from parallel computing
            Theta_error_norm_step_k[k * num_batch + count_temp] = 0
            for jk in range(batch_size):
                temp_xy = tempdata[jk]
                Gradient_OuterSize_All_x[jk, :] = temp_xy[0]
                Gradient_OuterSize_All_y[jk, :] = temp_xy[1]
                # Theta_error_norm_step_k[k] += nmp.linalg.norm(temp_xy[2])/N_samples/nmp.linalg.norm(theta_true_all[jk, :])
                Theta_error_norm_step_k[k * num_batch + count_temp] += (nmp.linalg.norm(temp_xy[2])) ** 2 / batch_size
                theta_esti_all[jk, :] = temp_xy[3]
            theta_esti_monitor[k * num_batch + count_temp, :] = temp_xy[3]
            step_alpha[k * num_batch + count_temp] = lr_outer
        count_temp += 1
    if (nmp.all(nmp.abs(nmp.array(temp_Gx)) < tol_G) and nmp.all(nmp.abs(nmp.array(temp_Gy)) < tol_G)) and k > 1:
        break
end = time.time()
print('{:.4f} s'.format(end - start))  # the computational time

# for tesing!
# Phi = nmp.zeros((N_sensors, 1))
# for i in range(N_sensors):
#     Phi[i] = TwoDimenGauPlumeM_AllSource_Reading(x_sensor[i], y_sensor[i], source_location_x, source_location_y, Wr_x[0], Wr_y[0], w_speed[0], theta_true_all[0, :], K, H, sensor_noise_all[0, i])
# [C_i, D_i_T] = GradientInnerNew(x_sensor, y_sensor, source_location_x, source_location_y, Wr_x[0], Wr_y[0], w_speed[0], K, H, Phi, sigma_epsilon, lambda_1, lambda_2)
#
#
# [xx, yy] = GradientOuterNew(x_sensor, y_sensor, source_location_x, source_location_y, Wr_x[0], Wr_y[0], w_speed[0], K, H, Phi, sigma_epsilon, lambda_1, mean)
#

# # plot the concentration field
# C = nmp.zeros((len(x), len(y)))
# mk = 0  # the index of the sample
# for i in range(len(x)):
#     for j in range(len(y)):
#         for k in range(N_sources):
#             C[i, j] += TwoDimenGauPlumeM(x[i], y[j], source_location_x[k], source_location_y[k], Wr_x[mk], Wr_y[mk],
#                                          w_speed[mk], theta_true_all[mk, k], K[k], H[k])
#
# xx, yy = nmp.meshgrid(x, y)
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

print(all_sensor_x[:, 0])
print(all_sensor_y[:, 0])
print(stepsize_x)
print(stepsize_y)
print(Theta_error_norm_step_k)
print(theta_esti_all)
print(theta_esti_monitor)
print(step_alpha)
print(temp_Gx)
print(temp_Gy)


l1 = plt.scatter(source_location_x, source_location_y, marker='x')
for i in range(N_sensors):
    plt.scatter(all_sensor_x[:, i], all_sensor_y[:, i], marker='.')
l2 = plt.scatter(x_sensor, y_sensor, marker='^')
l3 = plt.scatter(all_sensor_x[0, :], all_sensor_y[0, :], marker='*')
plt.xlim([-25, 25])
plt.ylim([-25, 25])
plt.legend((l1, l2, l3), ('emission sources', 'final sensor locations', 'initial sensor locations'),
           bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', ncols=3)
plt.show()

l1 = plt.scatter(source_location_x, source_location_y, marker='x')
plt.plot(all_sensor_x, all_sensor_y)
l2 = plt.scatter(x_sensor, y_sensor, marker='^')
l3 = plt.scatter(all_sensor_x[0, :], all_sensor_y[0, :], marker='*')
plt.xlim([-25, 25])
plt.ylim([-25, 25])
plt.legend((l1, l2, l3), ('emission sources', 'final sensor locations', 'initial sensor locations'),
           bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', ncols=3)
plt.show()

# plot without contour
l1 = plt.plot(source_location_x, source_location_y, marker='x', markersize=10, linestyle='None')
l2 = plt.plot(x_sensor, y_sensor, marker='^', markersize=10, linestyle='None')
plt.xlim([-25, 25])
plt.ylim([-25, 25])
# plt.legend((l1, l2), ('emission sources', 'final sensor locations'), bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', ncols=2)
plt.show()

# plot the step
for i in range(N_sensors):
    plt.plot(stepsize_x[:, i])
    plt.plot(stepsize_y[:, i])
plt.show()

for i in range(N_sensors):
    plt.plot(step_all_x[:, i])
    plt.plot(step_all_y[:, i])
plt.show()

# plot the objective value to show convergence
plt.plot(Theta_error_norm_step_k)
plt.show()

ws2 = nmp.random.uniform(ws_lower, ws_upper, 100000)  # the wind speed distribution
wd2 = nmp.random.uniform(wd_lower, wd_upper, 100000) * 360  # the wind speed distribution
ax = WindroseAxes.from_ax()
ax.box(wd2, ws2, bins=nmp.arange(0, 8, 1))
ax.set_legend()
plt.show()
