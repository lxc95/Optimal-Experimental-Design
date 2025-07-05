# Optimal-Experimental-Design
Title: **Optimal Sensor Allocation with Multiple Linear Dispersion Processes**/**Sparse Sensor Allocation for Inverse Problems of Detecting Sparse Leaking Emission Sources**

Please cite:
**Liu, X., Phan, D., Hwang, Y., Klein, L., Liu, X., & Yeo, K. (2024). Optimal Sensor Allocation with Multiple Linear Dispersion Processes. arXiv preprint arXiv:2401.10437.**

You can input any source locations, wind condition distributions, emission rate distributions and the number of sensors you would like to place, then the solver will output the final designs. The 'GUI_GPU.py' is developed based on the above codes and you can play with it for your projects.

Important notes:
  1. Please read the detailed descriptions at the beginnings of the "OptiSensorPlace_Torch_AnySource_AnySensor_miniBatch_AnyWind_...known_lr5e-7.py" file.


## Example 1: a specific concentration field
### - Initial guess of sensor locations

<img src="https://github.com/user-attachments/assets/e9b07830-a72b-4cf0-abd4-9e49b8ec70ab" style="width:30%;"><img src="https://github.com/user-attachments/assets/83bfbe4e-f4e4-43a3-a8a1-38cf6c268638" style="width:30%;"><img src="https://github.com/user-attachments/assets/e415971d-4b7d-474f-b09b-d56b04ad68a6" style="width:30%;">

<img src="https://github.com/user-attachments/assets/5dbd52fb-ce1d-4491-973a-1f5d0e4a8e21" style="width:30%;"><img src="https://github.com/user-attachments/assets/cb2eaed0-d46e-4bd6-9007-b9cbf29816dd" style="width:30%;"><img src="https://github.com/user-attachments/assets/8cb072ed-c779-4f5f-92c8-80575928d9e7" style="width:30%;">

### - Initial design + bilevel optimization
<img src="https://github.com/user-attachments/assets/b6bdddb1-fee1-43b4-ab06-7e2c3ef891ec" style="width:40%;"><img src="https://github.com/user-attachments/assets/32b268d6-4486-470a-a86b-5d1c2662dc66" style="width:40%;">

## Example 2:  allocate 6 sensors for 10 sources
### - Objective value decreases

<p align="center">
<img src="https://github.com/user-attachments/assets/7337f65f-5be7-4678-84c9-baf21aa2a9ed" width="300"/><img src="https://github.com/user-attachments/assets/dadf49f8-5420-4ff5-8bbe-2e8738b502c1" width="450"/>
</p>

## Example 3:  allocate 50 sensors for 100 sources
### - Scalable sensor allocation

<p align="center">
<img src="https://github.com/user-attachments/assets/941ad77b-4f5c-4f04-a855-cefc10af317c" width="300""><img src="https://github.com/user-attachments/assets/0d4c48ab-09c0-4dc9-970a-cc686ef8b20b" width="230"">
</p>

## Example 4:  the software GUI
### - starting with K-means design (using GPU)

<p align="center">
<img src="https://github.com/user-attachments/assets/46c58485-b3c9-4c26-9672-0e6af0d10f35" style="width:80%;">
</p>
