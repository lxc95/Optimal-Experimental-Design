# Optimal-Experimental-Design
Title: Optimal Sensor Placement for Atmospheric Inverse Modeling Using Bilevel Optimization

Write-up is available on Overleaf.

The "OptiSensorPlace_AnySource_AnySensor_miniBatch_20230123_AnyWind.py" file is the implementation of the Algorithm 1.

You can input any source locations, wind condition distributions, emission rate distributions and the number of sensors you would like to place, then the solver will output the final designs.

 
 Important notes:
 
 
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
        initial guesses of snesor locations.
"""
