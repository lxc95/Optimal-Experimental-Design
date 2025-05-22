# Optimal-Experimental-Design
Title: **Optimal Sensor Allocation with Multiple Linear Dispersion Processes**

The "OptiSensorPlace_Torch_AnySource_AnySensor_miniBatch_AnyWind_newWindAngle_20241202_SGD_evaluation_10sensor_20source_3000iter_random_1e5_SGD10000_Aoptimal20_priorSTD20_lambda0.01_smallq_q1_known_lr5e-7.py" file is the implementation of the SBA algorithm without GUI. You can input any source locations, wind condition distributions, emission rate distributions and the number of sensors you would like to place, then the solver will output the final designs.

Important notes:
  1. Please read the detailed descriptions at the beginnings of the "OptiSensorPlace_AnySource_AnySensor_miniBatch_20230123_AnyWind.py" file.

The 'GUI.py' is developed based on the above codes and you can play with it for your projects.

## Example 1: a specific concentration field
### - Initial guess of sensor locations
<img src="https://github.com/user-attachments/assets/e9b07830-a72b-4cf0-abd4-9e49b8ec70ab" height="200"/><img src="https://github.com/user-attachments/assets/83bfbe4e-f4e4-43a3-a8a1-38cf6c268638" height="200"/><img src="https://github.com/user-attachments/assets/e415971d-4b7d-474f-b09b-d56b04ad68a6" height="200"/>

<img src="https://github.com/user-attachments/assets/5dbd52fb-ce1d-4491-973a-1f5d0e4a8e21" height="200"/><img src="https://github.com/user-attachments/assets/cb2eaed0-d46e-4bd6-9007-b9cbf29816dd" height="200"/><img src="https://github.com/user-attachments/assets/8cb072ed-c779-4f5f-92c8-80575928d9e7" height="200"/>

### - Initial design + bilevel optimization
<img src="https://github.com/user-attachments/assets/b6bdddb1-fee1-43b4-ab06-7e2c3ef891ec" height="250"/><img src="https://github.com/user-attachments/assets/32b268d6-4486-470a-a86b-5d1c2662dc66" height="250"/>

## Example 2:  allocate 6 sensors for 10 sources
### - Objective value decreases
<img src="https://github.com/user-attachments/assets/7337f65f-5be7-4678-84c9-baf21aa2a9ed" height="230"/><img src="https://github.com/user-attachments/assets/dadf49f8-5420-4ff5-8bbe-2e8738b502c1" height="210"/>

## Example 3:  allocate 50 sensors for 100 sources
### - Scalable sensor allocation
<img src="https://github.com/user-attachments/assets/941ad77b-4f5c-4f04-a855-cefc10af317c" height="330"/><img src="https://github.com/user-attachments/assets/0d4c48ab-09c0-4dc9-970a-cc686ef8b20b" height="300"/>
