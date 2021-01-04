import pandas as pd
import json
import numpy as np
 
#Used for converting NULL values as "0"s in given file. 
def null2zero(inFile):
  with open(inFile, 'r') as file :
    filedata = file.read()

  # Replace the target string
  filedata = filedata.replace('null', "0")

  # Write the file out again
  with open(inFile, 'w') as file:
    file.write(filedata)

############# - PART 1 - #############
########### IMPORTING DATA ###########

file = "edge.json"
with open(file) as f:
  data = json.load(f)

df = pd.DataFrame(np.array(data["rows"][::]), columns=data["cols"])

############# - PART 2 - #############
########### EXPLATORY DATA ANALYSIS & DATA PREPARATION  ###########
df.shape
df.head()

#Dropping unnecessary columns.
df = df.drop(columns=["entity_type","fiware_servicepath","__original_ngsi_entity__","entity_id"])

df.shape
df.head()

############# - PART 3 - #############
########### K-MEANS ALGORITHM WITH PYSPARK ########### 
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

lastDF=transformed.toPandas()









