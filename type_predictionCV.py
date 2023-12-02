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
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


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


# Split the data into training and testing sets
trainingData, testingData = df1.randomSplit([0.8, 0.2], seed=1234)

# Set up the possible hyperparameter values to be evaluated
max_depth_list = [30, 25, 20, 15]
minInstancesPerNode_list = [3, 5, 7]

# Create an empty DataFrame to store cross-validation results
cv_results = pd.DataFrame(columns=['max_depth', 'minInstancesPerNode', 'avg_f1', 'bestModel'])

best_avg_f1 = float('-inf')  # Initialize best average f1

for max_depth in max_depth_list:
    for minInsPN in minInstancesPerNode_list:
        # Construct a DT model using a set of hyper-parameter values
        dt = DecisionTreeClassifier(labelCol="i_type", featuresCol="scaledFeatures", maxDepth=max_depth, minInstancesPerNode=minInsPN)
        
        # Define the parameter grid
        paramGrid = ParamGridBuilder() \
            .addGrid(dt.maxDepth, [max_depth]) \
            .addGrid(dt.minInstancesPerNode, [minInsPN]) \
            .build()
        
        # Create a CrossValidator with k-fold (e.g., k=5) cross-validation
        evaluator = MulticlassClassificationEvaluator(labelCol="i_type", predictionCol="prediction", metricName="f1")
        crossval = CrossValidator(estimator=dt,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=5, parallelism=2)  # Set the number of folds
        
        # Perform cross-validation and get the average f1 score
        cv_model = crossval.fit(trainingData)
        avg_f1 = cv_model.avgMetrics[0]
        
        # Store results in the DataFrame
        cv_results = cv_results.append({'max_depth': max_depth, 'minInstancesPerNode': minInsPN, 'avg_f1': avg_f1, 'bestModel': 0}, ignore_index=True)
        
        # Update best average f1 and mark the best model
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_max_depth = max_depth
            best_minInstancesPerNode = minInsPN
            best_index = len(cv_results) - 1  # Index of the best model in the DataFrame

# Mark the best model in the DataFrame
cv_results.loc[best_index, 'bestModel'] = 1000

# Save the results to a CSV file
output_path = "/storage/home/cpw5598/MiniProj/type_predictionHPCV.csv"
cv_results.to_csv(output_path, index=False)  # Set index=False to exclude the DataFrame index from the CSV