import numpy as nmp
import math
from sklearn.linear_model import LinearRegression


def SimpleGaussianPlumeM(w, YZ, sensor, source, YY, VZ, thetaAll, noise):
    A_1 = 1 / (2 * nmp.pi * w * YZ[0]) * math.exp(
        -((sensor - source[0]) * 2500.) ** 2. / (2. * YY[0])) * VZ[0] * 1e6
    A_2 = 1 / (2 * nmp.pi * w * YZ[1]) * math.exp(
        -((sensor - source[1]) * 2500.) ** 2. / (2. * YY[1])) * VZ[1] * 1e6
    A_3 = 1 / (2 * nmp.pi * w * YZ[2]) * math.exp(
        -((sensor - source[2]) * 2500.) ** 2. / (2. * YY[2])) * VZ[2] * 1e6
    A = nmp.array([A_1, A_2, A_3])
    theta = nmp.array([[thetaAll[0]], [thetaAll[1]], [thetaAll[2]]])
    reading = nmp.matmul(A, theta) + noise
    return reading

def GradientInner(w, YZ, sensor, source, YY, VZ, Phi,sigma_e):
    A_1 = 1 / (2 * nmp.pi * w * YZ[0]) * math.exp(
        -((sensor - source[0]) * 2500.) ** 2. / (2. * YY[0])) * VZ[0] * 1e6
    A_2 = 1 / (2 * nmp.pi * w * YZ[1]) * math.exp(
        -((sensor - source[1]) * 2500.) ** 2. / (2. * YY[1])) * VZ[1] * 1e6
    A_3 = 1 / (2 * nmp.pi * w * YZ[2]) * math.exp(
        -((sensor - source[2]) * 2500.) ** 2. / (2. * YY[2])) * VZ[2] * 1e6
    A = nmp.array([A_1, A_2, A_3])
    A_star = nmp.array([[A_1], [A_2], [A_3]])
    C = 1./(sigma_e**2)*nmp.multiply(A_star, A) + 1.0 / 400. * nmp.identity(3)
    D_T = -1./(sigma_e**2)*A_star*Phi
    return [C, D_T]

def GradientOuter(w, YZ, sensor, source, YY, VZ, Phi,sigma_e,theta_esti):
    A_0 = 1 / (2 * nmp.pi * w * YZ[0]) * math.exp(
        -((sensor - source[0]) * 2500.) ** 2. / (2. * YY[0])) * VZ[0] * 1e6
    A_1 = 1 / (2 * nmp.pi * w * YZ[1]) * math.exp(
        -((sensor - source[1]) * 2500.) ** 2. / (2. * YY[1])) * VZ[1] * 1e6
    A_2 = 1 / (2 * nmp.pi * w * YZ[2]) * math.exp(
        -((sensor - source[2]) * 2500.) ** 2. / (2. * YY[2])) * VZ[2] * 1e6
    A = nmp.array([A_0, A_1, A_2])
    A_star = nmp.array([[A_0], [A_1], [A_2]])
    C = 1./(sigma_e**2)*nmp.multiply(A_star, A) + 1.0 / 400. * nmp.identity(3)
    G_0 = A_0 * A_0 * theta_esti[0] * ((source[0] - sensor) / YY[0] + (source[0] - sensor) / YY[0]) \
          + A_0 * A_1 * theta_esti[1] * ((source[0] - sensor) / YY[0] + (source[1] - sensor) / YY[1]) \
          + A_0 * A_2 * theta_esti[2] * ((source[0] - sensor) / YY[0] + (source[2] - sensor) / YY[2]) \
          - A_0 * Phi * ((source[0] - sensor) / YY[0])
    G_1 = A_1 * A_0 * theta_esti[0] * ((source[1] - sensor) / YY[1] + (source[0] - sensor) / YY[0]) \
          + A_1 * A_1 * theta_esti[1] * ((source[1] - sensor) / YY[1] + (source[1] - sensor) / YY[1]) \
          + A_1 * A_2 * theta_esti[2] * ((source[1] - sensor) / YY[1] + (source[2] - sensor) / YY[2]) \
          - A_1 * Phi * ((source[1] - sensor) / YY[1])
    G_2 = A_2 * A_0 * theta_esti[0] * ((source[2] - sensor) / YY[2] + (source[0] - sensor) / YY[0]) \
          + A_2 * A_1 * theta_esti[1] * ((source[2] - sensor) / YY[2] + (source[1] - sensor) / YY[1]) \
          + A_2 * A_2 * theta_esti[2] * ((source[2] - sensor) / YY[2] + (source[2] - sensor) / YY[2]) \
          - A_2 * Phi * ((source[2] - sensor) / YY[2])
    G = [[1./(sigma_e**2) * G_0 * 2500.], [1./(sigma_e**2) * G_1 * 2500.], [1./(sigma_e**2) * G_2 * 2500.]]
    # note that we used the linear regression package as the linear solver
    reg = LinearRegression().fit(-nmp.array(C), nmp.array(G))
    #reg.coef_
    return reg.coef_



# Define source location: the source locations are known; we first test the case used for IEEE paper
source_location_x = [200. / 2500., 1000. / 2500., -200. / 2500.]
source_location_y = [200. / 2500., -550. / 2500., -250. / 2500.]
N_sources = len(source_location_x)

# Define the number of Monte Carlo samples
N_samples = 5

# Define the sensor noise level -> the standard deviation
sigma_epsilon = 1

# Sample from the distribution of \
# the emission rates, the wind condition (we only alter the wind direction here), and the sensor noise
wind_abstract = 5.
wind_angle = 0.
theta_true = [80., 60., 40.]
#theta_true_all = nmp.array(theta_true * N_samples).reshape(N_samples, N_sources)
#wind_velocity_all = nmp.array([wind_abstract] * N_sources * N_samples).reshape(N_samples, N_sources)
#wind_direction_all = nmp.array([wind_angle] * N_sources * N_samples).reshape(N_samples, N_sources)
#sensor_noise_all = nmp.random.normal(0, sigma_epsilon, size=(N_samples, 1))

# call the Gaussian Plume model. We first use the simple model from IEEE paper
SigmaY_SigmaZ = [1.0041e6, 2.8839e5, 5.0637e5]
SigmaY_SigmaY = [1.7385e5, 8.3761e4, 1.1649e5]
vertical_Z = [2., 2., 2.]
# s = 390. / 2500.
# reading_all = nmp.zeros(N_samples)
# for i in range(N_samples):
#     reading_all[i] = SimpleGaussianPlumeM(wind_abstract, SigmaY_SigmaZ, s, source_location_x, SigmaY_SigmaY, vertical_Z, theta_true, sensor_noise_all[i])

# the SGD-based Bilevel approximation method
s_0 = 700. / 2500.
theta_0 = [0., 0., 0.]
s = s_0
n_k = 300
q = 300
reading_all = nmp.zeros(N_samples)
Gradient_OuterSize_All = nmp.zeros(N_samples)
for k in range(n_k):
    theta_esti_all = nmp.zeros((N_samples, N_sources))
    theta_error_all = nmp.zeros((N_samples, N_sources))
    Gradient_outerAll = nmp.zeros((N_samples, N_sources))
    Gradient_InerSize_All = nmp.zeros((q, N_sources))
    # re-samplings
    sensor_noise_all = nmp.random.normal(0, sigma_epsilon, size=(N_samples, 1))
    for i in range(N_samples):
        reading_all[i] = SimpleGaussianPlumeM(wind_abstract, SigmaY_SigmaZ, s, source_location_x, SigmaY_SigmaY,
                                              vertical_Z, theta_true, sensor_noise_all[i])
    for i in range(N_samples):
        [C, D_T] = GradientInner(wind_abstract, SigmaY_SigmaZ, s, source_location_x, SigmaY_SigmaY, vertical_Z,
                                 reading_all[i], sigma_epsilon)
        theta_esti_all[i, :] = theta_0
        lr_inner = 0.99
        for j in range(q):
            Gradient_InerSize_All[j, :] = nmp.matmul(C, theta_esti_all[i, :]) + D_T.T
            # projected to be non-negative
            for mk in range(N_sources):
                theta_esti_all[i, mk] = max(0, theta_esti_all[i, mk])
            if j>200:
                theta_esti_all[i, :] = theta_esti_all[i, :] - lr_inner / nmp.sqrt(j+1) * Gradient_InerSize_All[j, :]
            else:
                theta_esti_all[i, :] = theta_esti_all[i, :] - lr_inner * Gradient_InerSize_All[j, :]
        Gradient_outerAll[i, :] = GradientOuter(wind_abstract, SigmaY_SigmaZ, s, source_location_x, SigmaY_SigmaY, vertical_Z, reading_all[i],
                        sigma_epsilon, theta_esti_all[i, :])
        theta_error_all[i, :] = theta_esti_all[i, :] - theta_true
        Gradient_OuterSize_All[i] = nmp.matmul(Gradient_outerAll[i, :], theta_error_all[i, :])
    lr_outer = 0.01
    if k > 100:
        s = s - lr_outer / nmp.sqrt(k+1) * (2. * nmp.mean(Gradient_OuterSize_All[:]))
    else:
        s = s - lr_outer * (2. * nmp.mean(Gradient_OuterSize_All[:]))

# print(reading_all)
# print(Gradient_InerSize_All)
# print(theta_esti_all)
# print(Gradient_OuterSize_All)
print("The optimal location is X =", s*2500., "m")

