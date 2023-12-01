# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import col, column
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql import Row
import csv
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, IndexToString, StandardScaler, PCA
from pyspark.ml.classification import DecisionTreeClassifier
from decision_tree_plot.decision_tree_parser import decision_tree_parse
from decision_tree_plot.decision_tree_plot import plot_trees
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ss = SparkSession.builder.appName("Project").getOrCreate()

schema = StructType([ StructField("ID", IntegerType(), False ), \
                        StructField("Case Number", StringType(), False), \
                        StructField("Date", StringType(), False ), \
                        StructField("Block", StringType(), False ), \
                        StructField("IUCR", StringType(), False), \
                        StructField("Primary Type", StringType(), False), \
                        StructField("Description", StringType(), False),\
                        StructField("Location Description", StringType(), False), \
                        StructField("Arrest", StringType(), False), \
                        StructField("Domestic", StringType(), False), \
                        StructField("District", StringType(), False) ,\
                        StructField("Ward", StringType(), False ), \
                        StructField("Community Area", StringType(), False ), \
                        StructField("FBI Code", StringType(), False), \
                        StructField("Year", StringType(), False), \
                        StructField("Latitude", StringType(), False),\
                        StructField("Longitude", StringType(), False)
                           ])

data = ss.read.csv("new_file3.csv",schema=schema, header=True, inferSchema=False)

df = data.drop(*['Case Number', 'Block', 'Description', 'District', 'Community Area'])

df1 = df
df1.persist()

# List of categorical columns to convert to numerical
categorical_cols = ["Date", "IUCR", "Location Description", "Arrest", "Domestic", "FBI Code", "Year", "Ward", 'Latitude', 'Longitude']

# StringIndexer to convert categorical strings to numerical indices
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index").fit(df1) for col in categorical_cols]
pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(df).transform(df)

indexed_df = StringIndexer(inputCol='Primary Type', outputCol="i_type").fit(df_r).transform(df_r)

selected_columns = [f"{col}_index" for col in categorical_cols]
assembler = VectorAssembler(inputCols=selected_columns, outputCol="features")
assembled_df = assembler.transform(indexed_df)

# Scale the features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scaled_df = scaler.fit(assembled_df).transform(assembled_df)

df1 = scaled_df.select("scaledFeatures","i_type")
df1.persist()

model_path="/storage/home/cpw5598/MiniProj/DT_HPT_cluster"

trainingData, testingData = df1.randomSplit([0.8, 0.2], seed=1234)

# Chosen hyperparameters
max_depth = 5
minInstancesPerNode = 4

# Train the model with chosen hyperparameters
dt = DecisionTreeClassifier(labelCol="i_type", featuresCol="scaledFeatures", maxDepth=max_depth, minInstancesPerNode=minInstancesPerNode)
model = dt.fit(trainingData)

# Make predictions on training and testing data
training_predictions = model.transform(trainingData)
testing_predictions = model.transform(testingData)

# Evaluate model performance
evaluator = MulticlassClassificationEvaluator(labelCol="i_type", predictionCol="prediction", metricName="f1")
training_f1 = evaluator.evaluate(training_predictions)
testing_f1 = evaluator.evaluate(testing_predictions)


trainingData, testingData= df1.randomSplit([0.8, 0.2], seed=1234)

## Initialize a Pandas DataFrame to store evaluation results of all combination of hyper-parameter settings
hyperparams_eval_df = pd.DataFrame( columns = ['max_depth', 'minInstancesPerNode', 'training f1', 'testing f1', 'Best Model'] )
# initialize index to the hyperparam_eval_df to 0
index =0
# initialize lowest_error
highest_testing_f1 = 0
# Set up the possible hyperparameter values to be evaluated
max_depth_list = [15, 16, 17, 18, 19, 20, 25]
minInstancesPerNode_list = [2, 3, 4, 5]
trainingData.persist()
testingData.persist()
for max_depth in max_depth_list:
    for minInsPN in minInstancesPerNode_list:
        seed = 37
        # Construct a DT model using a set of hyper-parameter values and training data
        dt= DecisionTreeClassifier(labelCol="i_type", featuresCol="scaledFeatures", maxDepth=max_depth, minInstancesPerNode=minInsPN)
        model = dt.fit(trainingData)
        training_predictions = model.transform(trainingData)
        testing_predictions = model.transform(testingData)
        evaluator = MulticlassClassificationEvaluator(labelCol="i_type", predictionCol="prediction", metricName="f1")
        training_f1 = evaluator.evaluate(training_predictions)
        testing_f1 = evaluator.evaluate(testing_predictions)
        # We use 0 as default value of the 'Best Model' column in the Pandas DataFrame.
        # The best model will have a value 1000
        hyperparams_eval_df.loc[index] = [max_depth, minInsPN, training_f1, testing_f1, 0]
        index = index +1
        if testing_f1 > highest_testing_f1 :
            best_max_depth = max_depth
            best_minInsPN = minInsPN
            best_index = index -1
            best_parameters_training_f1 = training_f1
            best_DTmodel= model
            best_tree = decision_tree_parse(best_DTmodel, ss, model_path)
            column = dict( [ (str(idx), i) for idx, i in enumerate(selected_columns) ])
            highest_testing_f1 = testing_f1
print('The best max_depth is ', best_max_depth, ', best minInstancesPerNode = ', \
      best_minInsPN, ', testing f1 = ', highest_testing_f1)
column = dict([(str(idx), i) for idx, i in enumerate(selected_columns)])

# Store the Testing RMS in the DataFrame
hyperparams_eval_df.loc[best_index]=[best_max_depth, best_minInsPN, best_parameters_training_f1, highest_testing_f1, 1000]

output_path = "/storage/home/cpw5598/MiniProj/ward_predictionHP.csv"
hyperparams_eval_df.to_csv(output_path)

