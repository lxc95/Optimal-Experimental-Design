import numpy as nmp
import math
import cvxopt
import matplotlib.pyplot as plt
import quadprog

nmp.random.seed(0)


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


def GradientInner(w, YZ, sensor, source, YY, VZ, Phi_input, sigma_e, lambda_1_temp, lambda_2_temp):
    num_sensor = len(sensor)
    num_source = len(source)
    Phi_tem = nmp.zeros((num_sensor, 1))
    Phi_tem[:, 0] = Phi_input
    A_all = nmp.zeros((num_sensor, num_source))
    for i_ in range(num_sensor):
        for j_ in range(num_source):
            A_all[i_, j_] = 1 / (2 * nmp.pi * w * YZ[j_]) * math.exp(
                -((sensor[i_] - source[j_]) * 2500.) ** 2. / (2. * YY[j_])) * VZ[j_] * 1e6
    C = 1./(sigma_e**2)*(A_all.T @ A_all) + lambda_1_temp * nmp.identity(num_source)
    if num_sensor == 1:
        D_T = lambda_2_temp * nmp.ones((num_source, 1)) - 1. / (sigma_e ** 2) * A_all.T * Phi_tem
    else:
        D_T = lambda_2_temp * nmp.ones((num_source, 1)) - 1. / (sigma_e ** 2) * A_all.T @ Phi_tem
    return [C, D_T]


def GG_x(w, sensor_tem, YZ_i, YY_i, VZ_i, source_i, YZ_j, YY_j, VZ_j, source_j, theta_temp):
    A_i = 1 / (2 * nmp.pi * w * YZ_i) * math.exp(
        -((sensor_tem - source_i) * 2500.) ** 2. / (2. * YY_i)) * VZ_i * 1e6
    A_j = 1 / (2 * nmp.pi * w * YZ_j) * math.exp(
        -((sensor_tem - source_j) * 2500.) ** 2. / (2. * YY_j)) * VZ_j * 1e6
    GG = A_i * A_j * theta_temp * ((source_i - sensor_tem) / YY_i + (source_j - sensor_tem) / YY_j) * 2500
    return GG


def G_x(w, sensor_tem, YZ_i, YY_i, VZ_i, source_i, Phi_tem):
    A_i = 1 / (2 * nmp.pi * w * YZ_i) * math.exp(
        -((sensor_tem - source_i) * 2500.) ** 2. / (2. * YY_i)) * VZ_i * 1e6
    G = A_i * ((source_i - sensor_tem) / YY_i) * Phi_tem * 2500
    return G


def GradientOuter(w, YZ, sensor, source, YY, VZ, Phi_tem, sigma_e, theta_esti, lambda_1_temp):
    num_sensor = len(sensor)
    num_source = len(source)
    A_all = nmp.zeros((num_sensor, num_source))
    for i_ in range(num_sensor):
        for j_ in range(num_source):
            A_all[i_, j_] = 1 / (2 * nmp.pi * w * YZ[j_]) * math.exp(
                -((sensor[i_] - source[j_]) * 2500.) ** 2. / (2. * YY[j_])) * VZ[j_] * 1e6
    C = 1./(sigma_e**2)*(A_all.T @ A_all) + lambda_1_temp * nmp.identity(num_source)
    G_all = nmp.zeros((num_sensor, num_source))
    for mk in range(num_sensor):
        for i_ in range(num_source):
            for j_ in range(num_source):
                G_all[mk, i_] += 1. / (sigma_e ** 2) * GG_x(w, sensor[mk], YZ[i_], YY[i_], VZ[i_], source[i_], YZ[j_], YY[j_], VZ[j_], source[j_], theta_esti[j_])
            G_all[mk, i_] -= 1. / (sigma_e ** 2) * G_x(w, sensor[mk], YZ[i_], YY[i_], VZ[i_], source[i_], Phi_tem[mk])
    coef_x = nmp.zeros((num_sensor, num_source))
    for i in range(num_sensor):
        G_x_T = nmp.zeros(num_source)
        G_x_T[:] = nmp.array(G_all[i, :])
        if nmp.any(nmp.isnan(G_x_T)):
            coef_x[i, :] = nmp.zeros(num_source)
        else:
            reg1 = nmp.linalg.solve(-nmp.array(C), G_x_T)
            coef_x[i, :] = reg1
    return coef_x


def cvxopt_solve_qp(P, q_tem, G, h, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q_tem)]
    args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args, options={'show_progress': False})
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


# Define source location: the source locations are known; we first test the case used for IEEE paper
source_location_x = nmp.array([200. / 2500., 1000. / 2500., -200. / 2500.])
source_location_y = nmp.array([200. / 2500., -550. / 2500., -250. / 2500.])
N_sources = len(source_location_x)

# Define the number of Monte Carlo samples
N_samples = 50

# Define the number of sensors
N_sensors = 2

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

lambda_1 = 1/100.
lambda_2 = 1/100.

# the SGD-based Bilevel approximation method
s_0 = [-1000. / 2500., 1000. / 2500.]
theta_0 = [0., 0., 0.]
s = s_0
n_k = 2000
q = 300
reading_all = nmp.zeros((N_sensors, N_samples))
Gradient_OuterSize_All = nmp.zeros((N_samples, N_sensors))
Theta_error_norm_step_k = nmp.zeros(n_k) + nmp.inf
all_solution = nmp.zeros((n_k, N_sensors))
# re-samplings
sensor_noise_all = nmp.random.normal(0, sigma_epsilon, size=(N_samples, N_sensors))
for k in range(n_k):
    theta_esti_all = nmp.zeros((N_samples, N_sources))
    theta_error_all = nmp.zeros((N_samples, N_sources))
    # # re-samplings
    # sensor_noise_all = nmp.random.normal(0, sigma_epsilon, size=(N_samples, N_sensors))
    for pk in range(N_sensors):
        for i in range(N_samples):
            reading_all[pk, i] = SimpleGaussianPlumeM(wind_abstract, SigmaY_SigmaZ, s[pk], source_location_x, SigmaY_SigmaY,
                                                  vertical_Z, theta_true, sensor_noise_all[i, pk])
    for i in range(N_samples):
        [C, D_T] = GradientInner(wind_abstract, SigmaY_SigmaZ, s, source_location_x, SigmaY_SigmaY, vertical_Z,
                                 reading_all[:, i], sigma_epsilon, lambda_1, lambda_2)
        theta_esti_all[i, :] = theta_0
        # the inverse model
        theta_esti_all[i, :] = cvxopt_solve_qp(C, D_T, -1 * nmp.eye(N_sources), nmp.zeros(N_sources))
        # theta_esti_all[i, :] = quadprog_solve_qp(C, D_T, -1 * nmp.eye(N_sources), nmp.zeros(N_sources))
        # print(theta_esti_all[i, :])
        Gradient_outerAll = GradientOuter(wind_abstract, SigmaY_SigmaZ, s, source_location_x, SigmaY_SigmaY, vertical_Z, reading_all[:, i],
                        sigma_epsilon, theta_esti_all[i, :], lambda_1)
        theta_error_all[i, :] = theta_esti_all[i, :] - theta_true
        # print(Gradient_outerAll)
        Gradient_OuterSize_All[i, :] = nmp.matmul(Gradient_outerAll, theta_error_all[i, :])
        # print(theta_error_all[i, :])
    Theta_error_norm_step_k[k] = 0
    for jk in range(N_samples):
        if nmp.any(nmp.isinf(theta_error_all[jk, :])) or nmp.any(
                nmp.isnan(theta_error_all[jk, :])):  # avoid the Inf and NaN value from the inner solver
            Theta_error_norm_step_k[k] += (nmp.linalg.norm(theta_true)) ** 2 / N_samples
        else:
            Theta_error_norm_step_k[k] += (nmp.linalg.norm(theta_error_all[jk, :])) ** 2 / N_samples

    lr_outer = 0.001
    if k > n_k:
        s = s - lr_outer / nmp.sqrt(k+1) * 2. * nmp.mean(Gradient_OuterSize_All, 0)
    else:
        # print(nmp.mean(Gradient_OuterSize_All, 0))
        s = s - lr_outer * 2. * nmp.mean(Gradient_OuterSize_All, 0)
    all_solution[k, :] = s
    print('s:', s*2500, '; k = ', k)

# print(reading_all)
# print(Gradient_InerSize_All)
print(theta_esti_all)
# print(Gradient_OuterSize_All)
print("The optimal location is X =", s*2500., "m")
print(all_solution*2500)

plt.scatter(source_location_x*2500., source_location_y*2500., marker='x')
plt.plot(all_solution[:, 0]*2500., -nmp.ones(n_k)*2000., marker='.')
plt.plot(all_solution[:, 1]*2500., -nmp.ones(n_k)*2000., marker='.')
plt.plot(all_solution[0, 0]*2500., -2000., marker='*')
plt.plot(all_solution[0, 1]*2500., -2000., marker='*')
plt.plot(s[0]*2500., -2000, marker='^')
plt.plot(s[1]*2500., -2000, marker='^')
plt.xlim([-2500, 2500])
plt.ylim([-2500, 2500])
plt.show()

# plot
plt.plot(all_solution[:, 0]*2500., all_solution[:, 1]*2500., marker='.')
plt.plot(all_solution[0, 0]*2500., all_solution[0, 1]*2500., marker='x')
plt.plot(s[0]*2500., s[1]*2500., marker='^')
plt.xlim([-2500, 2500])
plt.ylim([-2500, 2500])
plt.show()

# plot the objective value to show convergence
plt.plot(Theta_error_norm_step_k)
plt.xlabel('the index of iteration')
plt.ylabel('the objective value')
plt.show()

# plot the objective value to show convergence
plt.plot(all_solution*2500)
plt.xlabel('the index of iteration')
plt.ylabel('the sensor location')
plt.show()

