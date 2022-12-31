import numpy as nmp
import pylab as p
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import time
from joblib import Parallel, delayed
import os


def TwoDimenGauPlumeM(x, y, source_x, source_y, w_x, w_y, u, Q, K, H):
    x_new = nmp.sqrt(((1. - w_x ** 2.) * (x - source_x) - w_x * w_y * (y - source_y)) ** 2. + (
            -w_x * w_y * (x - source_x) + (1. - w_y ** 2.) * (y - source_y)) ** 2.)
    y_new = w_x * (x - source_x) + w_y * (y - source_y)
    if y_new > 0.:
        Pi = Q / (2. * nmp.pi * K * y_new) * nmp.exp(-u * (x_new ** 2. + H ** 2.) / (4. * K * y_new))
    else:
        Pi = 0.
    return Pi


def TwoDimenGauPlumeM_AllSource_Reading(x, y, source_x, source_y, w_x, w_y, u, q, K, H, noise):
    Pi = 0
    for k in range(len(source_location_x)):
        Pi += TwoDimenGauPlumeM(x, y, source_x[k], source_y[k], w_x, w_y, u, q[k], K[k], H[k])
    return Pi + noise


def GradientInnerNew(x_all, y_all, source_x, source_y, w_x, w_y, u, K, H, Phi, sigma_e, lambda_1, lambda_2):
    num_sensor = len(x_all)
    num_source = len(source_x)
    A_all = nmp.zeros((num_sensor, num_source))
    for i in range(num_sensor):
        temp_n = 0  # count th e number of zero A's
        for j in range(num_source):
            x_new = nmp.sqrt(((1. - w_x ** 2.) * (x_all[i] - source_x[j]) - w_x * w_y * (y_all[i] - source_y[j])) ** 2. + (
                    -w_x * w_y * (x_all[i] - source_x[j]) + (1. - w_y ** 2.) * (y_all[i] - source_y[j])) ** 2.)
            y_new = w_x * (x_all[i] - source_x[j]) + w_y * (y_all[i] - source_y[j])
            if y_new > 0.:
                A_all[i, j] = 1 / (2. * nmp.pi * K[j] * y_new) * nmp.exp(-u * (x_new ** 2. + H[j] ** 2.) / (4. * K[j] * y_new))
            else:
                A_all[i, j] = 0.
                temp_n += 1
        if temp_n == num_source:
            Phi[i] = 0
    C_coef = 1./(sigma_e**2)*(A_all.T@A_all) + lambda_1 * nmp.identity(num_source)
    D_coef_T = lambda_2*nmp.ones((num_source,1)) - 1./(sigma_e**2)*A_all.T@Phi
    return [C_coef, D_coef_T]


def Gradient_AA_x(x, y, source_x1, source_y1, source_x2, source_y2, w_x, w_y, u, K1, H1, K2, H2):
    r_per1 = nmp.sqrt(((1. - w_x ** 2.) * (x - source_x1) - w_x * w_y * (y - source_y1)) ** 2. + (
            -w_x * w_y * (x - source_x1) + (1. - w_y ** 2.) * (y - source_y1)) ** 2.)
    r_para1 = w_x * (x - source_x1) + w_y * (y - source_y1)
    r_per2 = nmp.sqrt(((1. - w_x ** 2.) * (x - source_x2) - w_x * w_y * (y - source_y2)) ** 2. + (
            -w_x * w_y * (x - source_x2) + (1. - w_y ** 2.) * (y - source_y2)) ** 2.)
    r_para2 = w_x * (x - source_x2) + w_y * (y - source_y2)
    if r_para1 > 0 and r_para2 > 0:
        G_AA_x = -(1/(4*nmp.pi**2*K1*K2*r_para1*r_para2)**2)*4*nmp.pi**2*K1*K2*(w_x*(r_para1+r_para2))*nmp.exp(-u*(r_per1**2+H1**2)/(4*K1*r_para1)-u*(r_per2**2+H2**2)/(4*K2*r_para2)) \
                +1/(4*nmp.pi**2*K1*K2*r_para1*r_para2)*nmp.exp(-u*(r_per1**2+H1**2)/(4*K1*r_para1)-u*(r_per2**2+H2**2)/(4*K2*r_para2)) \
                *((-u*(2*(1-w_x**2)-2*w_x*w_y)*4*K1*r_para1+u*(r_per1**2+H1**2)*4*K1*w_x)/(4*K1*r_para1)**2 + (-u*(2*(1-w_x**2)-2*w_x*w_y)*4*K2*r_para2+u*(r_per2**2+H2**2)*4*K2*w_x)/(4*K2*r_para2)**2)
    else:
        G_AA_x = 0
    return G_AA_x


def Gradient_AA_y(x, y, source_x1, source_y1, source_x2, source_y2, w_x, w_y, u, K1, H1, K2, H2):
    r_per1 = nmp.sqrt(((1. - w_x ** 2.) * (x - source_x1) - w_x * w_y * (y - source_y1)) ** 2. + (
            -w_x * w_y * (x - source_x1) + (1. - w_y ** 2.) * (y - source_y1)) ** 2.)
    r_para1 = w_x * (x - source_x1) + w_y * (y - source_y1)
    r_per2 = nmp.sqrt(((1. - w_x ** 2.) * (x - source_x2) - w_x * w_y * (y - source_y2)) ** 2. + (
            -w_x * w_y * (x - source_x2) + (1. - w_y ** 2.) * (y - source_y2)) ** 2.)
    r_para2 = w_x * (x - source_x2) + w_y * (y - source_y2)
    if r_para1 > 0 and r_para2 > 0:
        G_AA_y = -(1/(4*nmp.pi**2*K1*K2*r_para1*r_para2)**2)*4*nmp.pi**2*K1*K2*(w_y*(r_para1+r_para2))*nmp.exp(-u*(r_per1**2+H1**2)/(4*K1*r_para1)-u*(r_per2**2+H2**2)/(4*K2*r_para2)) \
                +1/(4*nmp.pi**2*K1*K2*r_para1*r_para2)*nmp.exp(-u*(r_per1**2+H1**2)/(4*K1*r_para1)-u*(r_per2**2+H2**2)/(4*K2*r_para2)) \
                *((-u*(2*(1-w_y**2)-2*w_x*w_y)*4*K1*r_para1+u*(r_per1**2+H1**2)*4*K1*w_y)/(4*K1*r_para1)**2 + (-u*(2*(1-w_y**2)-2*w_x*w_y)*4*K2*r_para2+u*(r_per2**2+H2**2)*4*K2*w_y)/(4*K2*r_para2)**2)
    else:
        G_AA_y = 0
    return G_AA_y


def Gradient_A_x(x, y, source_x1, source_y1, w_x, w_y, u, K, H):
    r_per1 = nmp.sqrt(((1. - w_x ** 2.) * (x - source_x1) - w_x * w_y * (y - source_y1)) ** 2. + (
            -w_x * w_y * (x - source_x1) + (1. - w_y ** 2.) * (y - source_y1)) ** 2.)
    r_para1 = w_x * (x - source_x1) + w_y * (y - source_y1)
    if r_para1 > 0:
        G_A_x = -(1/(2*nmp.pi*K*r_para1)**2)*2*nmp.pi*K*w_x*nmp.exp(-u*(r_per1**2+H**2)/(4*K*r_para1)) \
                +1/(2*nmp.pi*K*r_para1)*nmp.exp(-u*(r_per1**2+H**2)/(4*K*r_para1)) \
                *((-u*(2*(1-w_x**2)-2*w_x*w_y)*4*K*r_para1+u*(r_per1**2+H**2)*4*K*w_x)/(4*K*r_para1)**2)
    else:
        G_A_x = 0
    return G_A_x


def Gradient_A_y(x, y, source_x1, source_y1, w_x, w_y, u, K, H):
    r_per1 = nmp.sqrt(((1. - w_x ** 2.) * (x - source_x1) - w_x * w_y * (y - source_y1)) ** 2. + (
            -w_x * w_y * (x - source_x1) + (1. - w_y ** 2.) * (y - source_y1)) ** 2.)
    r_para1 = w_x * (x - source_x1) + w_y * (y - source_y1)
    if r_para1 > 0:
        G_A_y = -(1/(2*nmp.pi*K*r_para1)**2)*2*nmp.pi*K*w_y*nmp.exp(-u*(r_per1**2+H**2)/(4*K*r_para1)) \
                +1/(2*nmp.pi*K*r_para1)*nmp.exp(-u*(r_per1**2+H**2)/(4*K*r_para1)) \
                *((-u*(2*(1-w_y**2)-2*w_x*w_y)*4*K*r_para1+u*(r_per1**2+H**2)*4*K*w_y)/(4*K*r_para1)**2)
    else:
        G_A_y = 0
    return G_A_y


def GradientOuterNew(x_all, y_all, source_x, source_y, w_x, w_y, u, K, H, Phi, sigma_e, lambda_1, theta_curr):
    num_sensor = len(x_all)
    num_source = len(source_x)
    A_all = nmp.zeros((num_sensor, num_source))
    for i in range(num_sensor):
        for j in range(num_source):
            x_new = nmp.sqrt(((1. - w_x ** 2.) * (x_all[i] - source_x[j]) - w_x * w_y * (y_all[i] - source_y[j])) ** 2. + (
                    -w_x * w_y * (x_all[i] - source_x[j]) + (1. - w_y ** 2.) * (y_all[i] - source_y[j])) ** 2.)
            y_new = w_x * (x_all[i] - source_x[j]) + w_y * (y_all[i] - source_y[j])
            if y_new > 0.:
                A_all[i, j] = 1 / (2. * nmp.pi * K[j] * y_new) * nmp.exp(-u * (x_new ** 2. + H[j] ** 2.) / (4. * K[j] * y_new))
            else:
                A_all[i, j] = 0.
    C_coef = 1./(sigma_e**2)*(A_all.T@A_all) + lambda_1 * nmp.identity(num_source)
    G_x = nmp.zeros((num_sensor, num_source))
    G_y = nmp.zeros((num_sensor, num_source))
    for mk in range(num_sensor):
        for i in range(num_source):
            for j in range(num_source):
                G_x[mk, i] += Gradient_AA_x(x_all[mk], y_all[mk], source_x[i], source_y[i], source_x[j], source_y[j],
                                        w_x, w_y, u, K[i], H[i], K[j], H[j]) * theta_curr[j]
                G_y[mk, i] += Gradient_AA_y(x_all[mk], y_all[mk], source_x[i], source_y[i], source_x[j], source_y[j],
                                        w_x, w_y, u, K[i], H[i], K[j], H[j]) * theta_curr[j]
            G_x[mk, i] += Gradient_A_x(x_all[mk], y_all[mk], source_x[i], source_y[i], w_x, w_y, u, K[i], H[i]) * Phi[mk]
            G_y[mk, i] += Gradient_A_y(x_all[mk], y_all[mk], source_x[i], source_y[i], w_x, w_y, u, K[i], H[i]) * Phi[mk]
    # note that we used the linear regression package as the linear solver
    # it is better to develop our own linear solver, which should be efficient
    coef_x = nmp.zeros((num_sensor, num_source))
    coef_y = nmp.zeros((num_sensor, num_source))
    for i in range(num_sensor):
        G_x_T = nmp.array(G_x[i, :])
        G_x_T = G_x_T.T
        if (nmp.isnan(G_x_T).any()):
            coef_x[i, :] = nmp.zeros(num_source)
        else:
            reg1 = LinearRegression().fit(-nmp.array(C_coef), G_x_T)
            coef_x[i, :] = reg1.coef_
        G_y_T = nmp.array(G_y[i, :])
        G_y_T = G_y_T.T
        if (nmp.isnan(G_y_T).any()):
            coef_y[i, :] = nmp.zeros(num_source)
        else:
            reg2 = LinearRegression().fit(-nmp.array(C_coef), G_y_T)
            coef_y[i, :] = reg2.coef_
    #reg.coef_
    return [coef_x, coef_y]


def Update_Inner_OuterStep(x_sensor, y_sensor, source_location_x, source_location_y, Wr_x, Wr_y, w_speed,
                                K, H, Phi, sigma_epsilon, lambda_1, lambda_2, theta_true_all, N_sources, q, i):
    theta_esti_all = nmp.zeros((N_samples, N_sources))
    theta_error_all = nmp.zeros((N_samples, N_sources))
    Gradient_InerSize_All = nmp.zeros((q, N_sources))
    [C, D_T] = GradientInnerNew(x_sensor, y_sensor, source_location_x, source_location_y, Wr_x[i], Wr_y[i], w_speed[i],
                                K, H, Phi, sigma_epsilon, lambda_1, lambda_2)
    # Set the initial guess of theta
    theta_esti_all[i, :] = nmp.zeros((1, N_sources))  # Here we start from zeros
    lr_inner = 0.0001
    para_lr_inner = 200
    for j in range(q):
        Gradient_InerSize_All[j, :] = nmp.matmul(C, theta_esti_all[i, :]) + D_T.T
        # projected to be non-negative
        for mk in range(N_sources):
            theta_esti_all[i, mk] = max(0, theta_esti_all[i, mk])
        if j > para_lr_inner:
            theta_esti_all[i, :] = theta_esti_all[i, :] - lr_inner / nmp.sqrt(j + 1) * Gradient_InerSize_All[j, :]
        else:
            theta_esti_all[i, :] = theta_esti_all[i, :] - lr_inner * Gradient_InerSize_All[j, :]
    [Gradient_outerAll_x, Gradient_outerAll_y] = GradientOuterNew(x_sensor, y_sensor, source_location_x,
                                                                  source_location_y, Wr_x[i], Wr_y[i], w_speed[i], K, H,
                                                                  Phi, sigma_epsilon, lambda_1, theta_esti_all[i, :])
    theta_error_all[i, :] = theta_esti_all[i, :] - theta_true_all[i, :]
    Gradient_OuterSize_All_x = nmp.matmul(Gradient_outerAll_x, theta_error_all[i, :])
    Gradient_OuterSize_All_y = nmp.matmul(Gradient_outerAll_y, theta_error_all[i, :])
    return [Gradient_OuterSize_All_x, Gradient_OuterSize_All_y]


# main
start = time.time()
cpu_count=os.cpu_count()+1
# the domain of the concentration field
x = nmp.linspace(-25, 25, 200)
y = nmp.linspace(-25, 25, 200)

# # Define source location: the source locations are known;
# source_location_x = [-15., -10., -9., -5., 5., 5., 8., 10., 15., 20.]
# source_location_y = [17., -5., 22., 10., 18., 0., -10., 19., -10, 5.]
# N_sources = len(source_location_x)
# # Define the mean and variance of emission rates
# mean = [8., 10., 9., 8., 10., 9., 8., 10., 9., 10.]  # the mean of emission rates for the above sources
# cov = [[1., 0, 0, 0, 0, 0, 0, 0, 0, 0.], [0., 1, 0, 0, 0, 0, 0, 0, 0, 0.], [0., 0, 1, 0, 0, 0, 0, 0, 0, 0.], [0., 0, 0, 1, 0, 0, 0, 0, 0, 0.],
#        [0., 0, 0, 0, 1, 0, 0, 0, 0, 0.], [0., 0, 0, 0, 0, 1, 0, 0, 0, 0.], [0., 0, 0, 0, 0, 0, 1, 0, 0, 0.], [0., 0, 0, 0, 0, 0, 0, 1, 0, 0.],
#        [0., 0, 0, 0, 0, 0, 0, 0, 1, 0.], [0., 0, 0, 0, 0, 0, 0, 0, 0, 1.]]  # the covariance of these emission rates
# # Define the height of stacks and the eddy diffusion coefficient
# H = [2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]  # the height of stacks
# K = [0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.5]  # the eddy diffusion coefficient, which is simplified

# Define source location: the source locations are known;
source_location_x = [-15., 5., 10., 15., 20.]
source_location_y = [17., 10., 0., -10., 5.]
N_sources = len(source_location_x)
# Define the mean and variance of emission rates
mean = [8., 10., 9., 8., 10.]  # the mean of emission rates for the above sources
cov = [[1., 0, 0, 0, 0], [0., 1, 0, 0, 0], [0., 0, 1, 0, 0], [0., 0, 0, 1, 0],
       [0., 0, 0, 0, 1]]  # the covariance of these emission rates
# Define the height of stacks and the eddy diffusion coefficient
H = [2., 2., 2., 2., 2.]  # the height of stacks
K = [0.4, 0.5, 0.6, 0.4, 0.5]  # the eddy diffusion coefficient, which is simplified

# # Define source location: the source locations are known;
# # This setting is similar to the example we used, but not exactly the same
# source_location_x = [2., 10., -2.]
# source_location_y = [2., -5.5, -2.5]
# N_sources = len(source_location_x)
# # Define the mean and variance of emission rates
# mean = [8., 6., 4.]  # the mean of emission rates for the above sources
# cov = [[8, 0, 0], [0., 8, 0], [0., 0, 8]]  # the covariance of these emission rates
# # Define the height of stacks and the eddy diffusion coefficient
# H = [2., 2., 2.]  # the height of stacks
# K = [0.4, 0.4, 0.4]  # the eddy diffusion coefficient, which is simplified

# Define the number of Monte Carlo samples
N_samples = 100
# Define the number of sensors
N_sensors = 2
# Define the sensor noise level -> the standard deviation
sigma_epsilon = 0.01
# Sample from the distribution of the emission rates, the wind condition, and the sensor noise
ws_mean = 2.
ws_std = 0.5
wd_lower = 0.4
wd_upper = 0.6
ws = nmp.random.normal(ws_mean, ws_std, N_samples)  # the wind angle distribution
wd = nmp.random.uniform(wd_lower, wd_upper, N_samples) * 360  # the wind speed distribution
Wr_x = nmp.cos((450. - wd) / 180. * nmp.pi)  # the x part of unit wind vector
Wr_y = nmp.sin((450. - wd) / 180. * nmp.pi)  # the y part of unit wind vector
w_speed = nmp.abs(ws)  # the wind speed
theta_true_all = nmp.abs(nmp.random.multivariate_normal(mean, cov, N_samples))
sensor_noise_all = nmp.random.normal(0, sigma_epsilon, size=(N_samples, N_sensors))
# lambda
lambda_1 = 1./100.
lambda_2 = 1./100.


# the SGD-based Bi-level approximation method
# Initialize the locations of sensors
x_sensor = nmp.random.uniform(0, 10, N_sensors)
y_sensor = nmp.random.uniform(-20, 0, N_sensors)
n_k = 10000
q = 300
Gradient_OuterSize_All_x = nmp.zeros((N_samples, N_sensors))
Gradient_OuterSize_All_y = nmp.zeros((N_samples, N_sensors))
all_sensor_x = nmp.zeros((n_k, N_sensors))
all_sensor_y = nmp.zeros((n_k, N_sensors))
stepsize_x = nmp.zeros((n_k, N_sensors))
stepsize_y = nmp.zeros((n_k, N_sensors))
for k in range(n_k):
    theta_esti_all = nmp.zeros((N_samples, N_sources))
    theta_error_all = nmp.zeros((N_samples, N_sources))
    Gradient_outerAll_x = nmp.zeros((N_sensors, N_sources))
    Gradient_outerAll_y = nmp.zeros((N_sensors, N_sources))
    Gradient_InerSize_All = nmp.zeros((q, N_sources))
    # re-samplings
    ws = nmp.random.normal(ws_mean, ws_std, N_samples)  # the wind angle distribution
    wd = nmp.random.uniform(wd_lower, wd_upper, N_samples) * 360  # the wind speed distribution
    Wr_x = nmp.cos((450. - wd) / 180. * nmp.pi)  # the x part of unit wind vector
    Wr_y = nmp.sin((450. - wd) / 180. * nmp.pi)  # the y part of unit wind vector
    w_speed = nmp.abs(ws)  # the wind speed
    theta_true_all = nmp.abs(nmp.random.multivariate_normal(mean, cov, N_samples))
    sensor_noise_all = nmp.random.normal(0, sigma_epsilon, size=(N_samples, N_sensors))
    # Generate the sensor readings
    Phi = nmp.zeros((N_sensors, 1))
    for i in range(N_sensors):
        Phi[i] = nmp.abs(TwoDimenGauPlumeM_AllSource_Reading(x_sensor[i], y_sensor[i], source_location_x, source_location_y,
                                                     Wr_x[0], Wr_y[0], w_speed[0], theta_true_all[0, :], K, H,
                                                     sensor_noise_all[0, i]))
    # Start the update
    # note that this for-loop is implemented in parallel
    parallel = Parallel(n_jobs=3, prefer="processes")
    tempdata = parallel(delayed(Update_Inner_OuterStep)(x_sensor, y_sensor, source_location_x, source_location_y, Wr_x, Wr_y, w_speed,
                           K, H, Phi, sigma_epsilon, lambda_1, lambda_2,
                           theta_true_all, N_sources, q, i) for i in range(N_samples))
    # Split the data from parallel computing
    for jk in range(N_samples):
        temp_xy = tempdata[jk]
        Gradient_OuterSize_All_x[jk, :] = temp_xy[0]
        Gradient_OuterSize_All_y[jk, :] = temp_xy[1]
    lr_outer = 0.99
    para_lr_outer = n_k
    if k > para_lr_outer:
        x_sensor = x_sensor - lr_outer / nmp.sqrt(k + 1) * (2. * nmp.mean(Gradient_OuterSize_All_x, 0))
        y_sensor = y_sensor - lr_outer / nmp.sqrt(k + 1) * (2. * nmp.mean(Gradient_OuterSize_All_y, 0))
    else:
        x_sensor = x_sensor - lr_outer * (2. * nmp.mean(Gradient_OuterSize_All_x, 0))
        y_sensor = y_sensor - lr_outer * (2. * nmp.mean(Gradient_OuterSize_All_y, 0))
    for mk in range(N_sensors):
        x_sensor[mk] = max(-25, x_sensor[mk])
        x_sensor[mk] = min(25, x_sensor[mk])
        y_sensor[mk] = max(-25, y_sensor[mk])
        y_sensor[mk] = min(25, y_sensor[mk])
    all_sensor_x[k, :] = x_sensor
    all_sensor_y[k, :] = y_sensor
    stepsize_x[k, :] = 2. * nmp.mean(Gradient_OuterSize_All_x, 0)  # check the step size of sensor locations
    stepsize_y[k, :] = 2. * nmp.mean(Gradient_OuterSize_All_y, 0)

end = time.time()
print('{:.4f} s'.format(end-start))  # the computational time


# for tesing!
# Phi = nmp.zeros((N_sensors, 1))
# for i in range(N_sensors):
#     Phi[i] = TwoDimenGauPlumeM_AllSource_Reading(x_sensor[i], y_sensor[i], source_location_x, source_location_y, Wr_x[0], Wr_y[0], w_speed[0], theta_true_all[0, :], K, H, sensor_noise_all[0, i])
# [C_i, D_i_T] = GradientInnerNew(x_sensor, y_sensor, source_location_x, source_location_y, Wr_x[0], Wr_y[0], w_speed[0], K, H, Phi, sigma_epsilon, lambda_1, lambda_2)
#
#
# [xx, yy] = GradientOuterNew(x_sensor, y_sensor, source_location_x, source_location_y, Wr_x[0], Wr_y[0], w_speed[0], K, H, Phi, sigma_epsilon, lambda_1, mean)


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

l1 = plt.scatter(source_location_x, source_location_y, marker='x')
for i in range(N_sensors):
    plt.scatter(all_sensor_x[:, i], all_sensor_y[:, i], marker='.')
l2 = plt.scatter(x_sensor, y_sensor, marker='^')
l3 = plt.scatter(all_sensor_x[0, :], all_sensor_y[0, :], marker='*')
plt.xlim([-25, 25])
plt.ylim([-25, 25])
plt.legend((l1, l2, l3), ('emission sources', 'final sensor locations', 'initial sensor locations'), bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', mode='expand', ncols=3)
plt.show()

ws2 = nmp.random.normal(ws_mean, ws_std, 100000)  # the wind angle distribution
wd2 = nmp.random.uniform(wd_lower, wd_upper, 100000) * 360  # the wind speed distribution
ax = WindroseAxes.from_ax()
ax.box(wd2, ws2, bins=nmp.arange(0, 8, 1))
ax.set_legend()
plt.show()

