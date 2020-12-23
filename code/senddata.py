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

def postData(url, row): 
    #row has following values: ID, START_TIME, SENT, STATISTICS[MIN_x,MIN_y,...]
    #we send only the statistics to fiware, so we convert them into a json object 
    json = {
    'MIN_x': row[3],
    'MIN_y': row[4],
    'MIN_z': row[5],
    'MAX_x': row[6],
    'MAX_y': row[7],
    'MAX_z': row[8],
    'MEAN_x': row[9],
    'MEAN_y': row[10],
    'MEAN_z': row[11],
    'RMS_x': row[12], 
    'RMS_y': row[13],
    'RMS_z': row[14],
    'STD_x': row[15],
    'STD_y': row[16],
    'STD_z': row[17],
    'SKEW_x': row[18],
    'SKEW_y': row[19],
    'SKEW_z': row[20],
    'KURT_x': row[21],
    'KURT_y': row[22],
    'KURT_z': row[23],
    }
    r = requests.post(url = url, json=json)
    print("r",r)
    return r

def createDatabaseConnection(fileName):
    conn = None
    try:
        conn = sqlite3.connect(fileName)
    except Error as e:
        print(e)
    
    return conn

    
def saveToDatabase(conn,startTime,jsonStat):
    c = conn.cursor()
    print("jsonStat[MIN_x]",jsonStat["MIN_x"])
    
    c.execute("INSERT INTO vibration(START_TIME,SENT,MIN_x,MIN_y,MIN_z,MAX_x,MAX_y,MAX_z,MEAN_x,MEAN_y,MEAN_z,RMS_x,RMS_y,RMS_z,STD_x,STD_y,STD_z,SKEW_x,SKEW_y,SKEW_z,KURT_x,KURT_y,KURT_z) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
              (1000,0,jsonStat["MIN_x"],jsonStat["MIN_y"],jsonStat["MIN_z"],jsonStat["MAX_x"],jsonStat["MAX_y"],jsonStat["MAX_z"],jsonStat["MEAN_x"],jsonStat["MEAN_y"],jsonStat["MEAN_z"],jsonStat["RMS_x"],jsonStat["RMS_y"],jsonStat["RMS_z"],
               jsonStat["STD_x"],jsonStat["STD_y"],jsonStat["STD_z"], jsonStat["SKEW_x"], jsonStat["SKEW_y"],jsonStat["SKEW_z"],jsonStat["KURT_x"],jsonStat["KURT_y"],jsonStat["KURT_z"]))
    conn.commit()

def getUnsentData(conn):
    c = conn.cursor()
    c.execute("SELECT * FROM vibration WHERE SENT=0 ORDER BY START_TIME ASC")
    row = c.fetchone()
    print(row)
    return row

    return json
    
def updateDatabase(conn,id):
    c = conn.cursor()
    c.execute("UPDATE vibration SET SENT=1 WHERE id=" + str(id))
    conn.commit()
    return

def main():
    #Read configurations
    configParser = configparser.RawConfigParser()   
    configFilePath = r'config.txt'
    configParser.read(configFilePath)
    lengthSecond = int(configParser.get('scalable-data-pipeline', 'dataSendingWindowsAsSeconds'))
    url = configParser.get('scalable-data-pipeline', 'fiwareEndpoint')
    sqliteFileName = configParser.get('scalable-data-pipeline', 'sqliteFileName')
    if (int(lengthSecond)) <= 0:
        lengthSecond = 10    
    #Create database connection
    conn = createDatabaseConnection(sqliteFileName)
    
    
    while 1:
        #Get data from database 
        dataToBeSent = getUnsentData(conn) # ID, START_TIME, SENT, STATISTICS[MIN_x,MIN_y,...]
        print("dataToBeSent[0]", dataToBeSent[0])
        #Post data to Fiware
        postResult = postData(url,dataToBeSent)
        print(postResult)
        #If post is successful update that row's SENT=1
        if postResult:
            print('Success')
            updateDatabase(conn,dataToBeSent[0])
        else:
            print('Error sending data to Fiware')
        time.sleep(lengthSecond)
        
if __name__ == '__main__':
    main()
