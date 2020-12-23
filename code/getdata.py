import os
import sys
import numpy as np
import requests
from scipy.stats import kurtosis, skew
import configparser
from numbers import Number
import sqlite3
from sqlite3 import Error
import time
import json

#Further parse the acc. data into x,y,z axis
x_axis_acceleration = []
y_axis_acceleration = []
z_axis_acceleration = []


'''
Send data to Fiware via HTTP POST request
'''
POST_ENDPOINT =  'http://f733c3fc722f.ngrok.io/iot/json?k=4jggokgpepnvsb2uv4s40d59ov&i=vibrationSensor001'

def getVibrationData(length_second, rate_hz):
    #Read acceleration data from ADXL345 and put the results in out.csv file
    os.system(f'sudo ../adxl345spi/adxl345spi -t {length_second} -f {rate_hz} -s out.csv')

def parseVibrationData():
    #Parse out.csv file and read the acceleration data
    acc_data = np.genfromtxt('out.csv', delimiter=',', names=True)
    for val in acc_data:
        x_axis_acceleration.append(val[1])
        y_axis_acceleration.append(val[2])
        z_axis_acceleration.append(val[3])

#Function to calculate Root Min Square Error
def rootMinSquare(actual_values,expected_values):
    if len(actual_values) != len(expected_values):
        return
    return np.square(np.subtract(actual_values,expected_values)).mean() 

#Function to calculate average of values
def average(values):
    if len(values) < 1:
        return 
    return np.mean(values)


def createStatistics():
    '''
    Min Values
    '''
    MIN_x = np.amin(x_axis_acceleration)
    MIN_y = np.amin(y_axis_acceleration)
    MIN_z = np.amin(z_axis_acceleration)

    print('MIN_x', MIN_x)
    print('MIN_y', MIN_y)
    print('MIN_z', MIN_z)

    '''
    Max Values
    '''
    MAX_x = np.amax(x_axis_acceleration)
    MAX_y = np.amax(y_axis_acceleration)
    MAX_z = np.amax(z_axis_acceleration)

    print('MAX_x', MAX_x)
    print('MAX_y', MAX_y)
    print('MAX_z', MAX_z)

    '''
    MEAN X,Y,Z axis accelerations
    '''
    MEAN_x = np.mean(x_axis_acceleration)
    MEAN_y = np.mean(y_axis_acceleration)
    MEAN_z = np.mean(z_axis_acceleration)

    print('MEAN_x', MEAN_x)
    print('MEAN_y', MEAN_y)
    print('MEAN_z', MEAN_z)

    '''
    Root Min Square Calculations
    '''
    RMS_x = np.sqrt(abs(np.mean(x_axis_acceleration)))
    RMS_y = np.sqrt(abs(np.mean(y_axis_acceleration)))
    RMS_z = np.sqrt(abs(np.mean(z_axis_acceleration)))

    print('RMS_x', RMS_x)
    print('RMS_y', RMS_y)
    print('RMS_z', RMS_z)

    '''
    Standard Deviation Calculations
    The standard deviation is the square root of the average of the squared deviations from the mean, i.e., std = sqrt(mean(abs(x - x.mean())**2)).
    The average squared deviation is normally calculated as x.sum()/N, where N = len(x). If, however, ddof is specified in std(), the divisor N - ddof is used instead
    '''
    STD_x = np.std(x_axis_acceleration, dtype=np.float64, ddof=1)
    STD_y = np.std(y_axis_acceleration, dtype=np.float64, ddof=1)
    STD_z = np.std(z_axis_acceleration, dtype=np.float64, ddof=1 )

    print('STD_x', STD_x)
    print('STD_y', STD_y)
    print('STD_z', STD_z)

    '''
    Skewness Calculations
    '''
    SKEW_x = skew(x_axis_acceleration)
    SKEW_y = skew(y_axis_acceleration)
    SKEW_z = skew(z_axis_acceleration)

    print('SKEW_x', SKEW_x)
    print('SKEW_y', SKEW_y)
    print('SKEW_z', SKEW_z)

    '''
    Kurtosis Calculations
    '''
    KURT_x = kurtosis(x_axis_acceleration)
    KURT_y = kurtosis(x_axis_acceleration)
    KURT_z = kurtosis(x_axis_acceleration)

    print('KURT_x', KURT_x)
    print('KURT_y', KURT_y)
    print('KURT_z', KURT_z)

    json = {
    'MIN_x': MIN_x,
    'MIN_y': MIN_y,
    'MIN_z': MIN_z,
    'MAX_x': MAX_x,
    'MAX_y': MAX_y,
    'MAX_z': MAX_z,
    'MEAN_x': MEAN_x,
    'MEAN_y': MEAN_y,
    'MEAN_z': MEAN_z,
    'RMS_x': RMS_x, 
    'RMS_y': RMS_y,
    'RMS_z': RMS_z,
    'STD_x': STD_x,
    'STD_y': STD_y,
    'STD_z': STD_z,
    'SKEW_x': SKEW_x,
    'SKEW_y': SKEW_y,
    'SKEW_z': SKEW_z,
    'KURT_x': KURT_x,
    'KURT_y': KURT_y,
    'KURT_z': KURT_z,
    }
    return json

def createDatabaseConnection(fileName):
    conn = None
    try:
        conn = sqlite3.connect(fileName)
    except Error as e:
        print(e)
    
    return conn

    
def saveToDatabase(conn,startTime,jsonStat):
    c = conn.cursor()
    
    c.execute("INSERT INTO vibration(START_TIME,SENT,MIN_x,MIN_y,MIN_z,MAX_x,MAX_y,MAX_z,MEAN_x,MEAN_y,MEAN_z,RMS_x,RMS_y,RMS_z,STD_x,STD_y,STD_z,SKEW_x,SKEW_y,SKEW_z,KURT_x,KURT_y,KURT_z) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (1000,0,jsonStat["MIN_x"],jsonStat["MIN_y"],jsonStat["MIN_z"],jsonStat["MAX_x"],jsonStat["MAX_y"],jsonStat["MAX_z"],jsonStat["MEAN_x"],jsonStat["MEAN_y"],jsonStat["MEAN_z"],jsonStat["RMS_x"],jsonStat["RMS_y"],jsonStat["RMS_z"],
               jsonStat["STD_x"],jsonStat["STD_y"],jsonStat["STD_z"], jsonStat["SKEW_x"], jsonStat["SKEW_y"],jsonStat["SKEW_z"],jsonStat["KURT_x"],jsonStat["KURT_y"],jsonStat["KURT_z"]))
    
    conn.commit()
    

def main():    
    configParser = configparser.RawConfigParser()   
    configFilePath = r'config.txt'
    configParser.read(configFilePath)
    
    lengthSecond = int(configParser.get('scalable-data-pipeline', 'dataGatheringWindowAsSeconds'))
    rateHz = int(configParser.get('scalable-data-pipeline', 'dataGatheringRateAsHertz'))
    sqliteFileName = configParser.get('scalable-data-pipeline', 'sqliteFileName')
    if (lengthSecond) <= 0:
        lengthSecond = 10

    if int(rateHz)<=0:
        rateHz = 5
    
    conn = createDatabaseConnection(sqliteFileName)
    
    while 1:
        startTimeInEpoch = int(time.time())
        getVibrationData(lengthSecond,rateHz)
        parseVibrationData()
        jsonStat = createStatistics()
        print("jsonStat",jsonStat)
        saveToDatabase(conn,startTimeInEpoch,jsonStat)
    
if __name__ == '__main__':
    main()
