import pandas as pd
import json
import numpy as np
# Normalize the data attributes for the Iris dataset.
from sklearn import preprocessing
import sys
import requests
import os


############# - PART 1 - #############
########### IMPORTING DATA ###########

#Taking data from AWS EC2 machine.
url = "http://" + "3.125.115.202" + ":4200/_sql"
#url = "http://" + str(sys.argv[1]) + ":4200/_sql"
headers = {'Content-Type' : 'application/json'}
data = '{"stmt":"SELECT * FROM mtopeniot.etvibrationsensor"}'

response = requests.post(url, headers=headers, data=data)
#print(response.text)
data = json.loads(response.text)

############# - PART 2 - #############
########### EXPLATORY DATA ANALYSIS & DATA PREPARATION  ###########

df = pd.DataFrame(np.array(data["rows"][::]), columns=data["cols"])
df = df.sort_values(by=['time_index'])
df = df.reset_index(drop=True)
#Eliminating NULL values
df = df.replace(np.nan, 0)

#Dropping unnecessary columns.
df = df.drop(columns=["entity_type","fiware_servicepath","__original_ngsi_entity__","entity_id"])

#Turning miliseconds to timestamp
df["time_index"] = pd.to_datetime(df.time_index, unit='ms')

#Normalizing data
normalised_data = preprocessing.normalize(df.loc[:, "min_x":"kurt_z"],axis = 0)
normalised_df = pd.DataFrame(df["time_index"])

tmp_df = pd.DataFrame(normalised_data, index=range(normalised_data.shape[0]),
                          columns=['min_x', 'min_y', 'min_z', 'max_x', 'max_y', 'max_z',
       'mean_x', 'mean_y', 'mean_z', 'rms_x', 'rms_y', 'rms_z', 'std_x',
       'std_y', 'std_z', 'skew_x', 'skew_y', 'skew_z', 'kurt_x', 'kurt_y',
       'kurt_z'])

normalised_df = pd.concat([normalised_df,tmp_df], axis=1)


############# - PART 3 - #############
########### K-MEANS IMPLEMENTATION WITH PYSPARK ########### 

import findspark
findspark.init("/opt/spark")

from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import *
from pyspark.sql.functions import *
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

#Converting NumPy dataframe to Spark dataframe
sdf = spark.createDataFrame(df)

sdf2 = sdf.withColumn("time_index",col("time_index").cast("float")).withColumn("min_x",col("min_x").cast("float")).withColumn("min_y",col("min_y").cast("float")).withColumn("min_z",col("min_z").cast("float")).withColumn("max_x",col("max_x").cast("float")).withColumn("max_y",col("max_y").cast("float")).withColumn("max_z",col("max_z").cast("float")).withColumn("mean_x",col("mean_x").cast("float")).withColumn("mean_y",col("mean_y").cast("float")).withColumn("mean_z",col("mean_z").cast("float")).withColumn("rms_x",col("rms_x").cast("float")).withColumn("rms_y",col("rms_y").cast("float")).withColumn("rms_z",col("rms_z").cast("float")).withColumn("std_x",col("std_x").cast("float")).withColumn("std_y",col("std_y").cast("float")).withColumn("std_z",col("std_z").cast("float")).withColumn("skew_x",col("skew_x").cast("float")).withColumn("skew_y",col("skew_y").cast("float")).withColumn("skew_z",col("skew_z").cast("float")).withColumn("kurt_x",col("kurt_x").cast("float")).withColumn("kurt_y",col("kurt_y").cast("float")).withColumn("kurt_z",col("kurt_z").cast("float"))

#Spark ML requires your input features to be gathered in a single column of your dataframe, usually named features;
#and it provides a specific method for doing this, VectorAssembler:
vecAssembler = VectorAssembler(inputCols=['time_index', 'min_x', 'min_y', 'min_z', 'max_x', 'max_y', 'max_z',
       'mean_x', 'mean_y', 'mean_z', 'rms_x', 'rms_y', 'rms_z', 'std_x',
       'std_y', 'std_z', 'skew_x', 'skew_y', 'skew_z', 'kurt_x', 'kurt_y',
       'kurt_z'], outputCol="features")

spark_df = vecAssembler.transform(sdf2)

kmeans = KMeans(k=2, seed=1)  # 2 clusters here
model = kmeans.fit(spark_df.select('features'))
transformed = model.transform(spark_df)
transformed.show()

#Printing centroids
centers = model.clusterCenters()
print(centers)

'''
#Saving the model
kmeans_path = "/kmeans"
kmeans.save(kmeans_path)
'''

df_predicted=transformed.toPandas()

############# - PART 4 - #############
########### VISUALISING RESULTS ########### 

import matplotlib.pyplot as plt
import seaborn as sns


#Correlation between normalized data
plt.figure(figsize=(10, 10))
cor = normalised_df.corr() #Calculate the correlation of the above variables
sns.heatmap(cor, square = True) #Plot the correlation as heat map
plt.title("Correlation between normalized data", fontsize =20)


'''
'time_index', 'min_x', 'min_y', 'min_z', 'max_x', 'max_y', 'max_z',
       'mean_x', 'mean_y', 'mean_z', 'rms_x', 'rms_y', 'rms_z', 'std_x',
       'std_y', 'std_z', 'skew_x', 'skew_y', 'skew_z', 'kurt_x', 'kurt_y',
       'kurt_z'
threedee = plt.figure(figsize=(12,10)).gca(projection='3d')
threedee.scatter(df_predicted.x, df_predicted.y, df_predicted.z, c=df_predicted.prediction)
threedee.set_xlabel('x')
threedee.set_ylabel('y')
threedee.set_zlabel('z')
plt.show()
'''

plt.figure(figsize=(10, 10))
plt.plot (df[:]["min_x"], label="x")
plt.plot (df[:]["min_y"], label="y")
plt.plot (df[:]["min_z"], label="z")
plt.title("Minimum x,y,z values", fontsize =20)
plt.legend(loc='upper left')
plt.show()

plt.figure(figsize=(10, 10))
plt.plot (df[:]["min_x"], label="x")
plt.plot (df[:]["min_y"], label="y")
plt.plot (df[:]["min_z"], label="z")
plt.title("Minimum x,y,z values", fontsize =20)
plt.legend(loc='upper left')
plt.show()















