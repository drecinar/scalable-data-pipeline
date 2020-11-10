import os
import numpy as np
import requests

#Constant values for ADXL345 sensor
sample_rate_Hz = 5
length_s = 10

#Read acceleration data from ADXL345 and put the results in out.csv file
os.system(f'sudo ../adxl345spi/adxl345spi -t {length_s} -f {sample_rate_Hz} -s out.csv')

#Parse out.csv file and read the acceleration data
acc_data = np.genfromtxt('out.csv', delimiter=',', names=True)

#Further parse the acc. data into x,y,z axis
x_axis_acceleration = []
y_axis_acceleration = []
z_axis_acceleration = []

for val in acc_data:
    x_axis_acceleration.append(val[0])
    y_axis_acceleration.append(val[1])
    z_axis_acceleration.append(val[2])

#Function to calculate Root Min Square Error
def rootMinSquare(actual_values,expected_values):
    if len(actual_values) != len(expected_values):
        return
    return np.square(np.subtract(actual_values,expected_values)).mean() 

#Function to calculate average of values
def average(values):
    if len(values) < 1:
        return 
    np.mean(values)

'''
Average X,Y,Z axis accelerations
'''
AVG_x = np.mean(x_axis_acceleration)
AVG_y = np.mean(y_axis_acceleration)
AVG_z = np.mean(z_axis_acceleration)

print('Average x axis acceleration', AVG_x)
print('Average y axis acceleration', AVG_y)
print('Average z axis acceleration', AVG_z)

'''
Root Min Square Error Calculations
'''
x_axis_acceleration_expected = 0.7
y_axis_acceleration_expected = 0.2
z_axis_acceleration_expected = 0.9

RMS_x = rootMinSquare(x_axis_acceleration, np.full(len(x_axis_acceleration),x_axis_acceleration_expected))
RMS_y = rootMinSquare(y_axis_acceleration, np.full(len(y_axis_acceleration),y_axis_acceleration_expected))
RMS_z = rootMinSquare(z_axis_acceleration, np.full(len(z_axis_acceleration),z_axis_acceleration_expected))

print('RMS_x', RMS_x)
print('RMS_y', RMS_y)
print('RMS_z', RMS_z)

'''
Send data to Fiware via HTTP POST request
'''
POST_ENDPOINT = 'http://4053a0ee7547.ngrok.io/iot/d?k=4jggokgpepnvsb2uv4s40d59ov&i=vibration001'

def postData():
    r = requests.post(url = POST_ENDPOINT, 
    json = {
        'AVG_x': AVG_x,
        'AVG_y': AVG_y,
        'AVG_z': AVG_z,
        'RMS_x': RMS_x, 
        'RMS_y': RMS_y,
        'RMS_z': RMS_z,
    })
    return r

postData()