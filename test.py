# Install necessary packages in Colab 
!pip install pyspark
!pip install findspark

# Spark environment
import findspark
findspark.init()

# Import libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import col
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName('WineQualityPrediction') \
    .master('local') \
    .getOrCreate()

#   Load and Inspect Validation Dataset 
# Specify the path of the validation dataset 
file_path = "ValidationDataset.csv"  
val = spark.read.csv(file_path, header=True, sep=";")
val.printSchema()  
val.show(5) 

#   Data Preprocessing 
for col_name in val.columns[:-1] + ['""""quality"""""']:
    val = val.withColumn(col_name, col(col_name).cast('float'))
val = val.withColumnRenamed('""""quality"""""', "label") 

# Data transformation
val.printSchema()
val.show(5)

#  Assemble Features into a Single Feature Vector
assembler = VectorAssembler(inputCols=val.columns[:-1], outputCol="features")
val = assembler.transform(val)

# Load Pre-Trained Random Forest Model
model_path = "/content/rf_wine_quality_model"  

if os.path.exists(model_path):
    print("Model path exists!")
else:
    print("Model path does not exist!")

# Attempt to load the pre-trained RandomForest model
try:
    RFModel = RandomForestClassificationModel.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

#   Make Predictions
predictions = RFModel.transform(val)
predictions.show(5)  # Display a few predictions

#  Evaluate Model Performance

label_pred_df = predictions.select("label", "prediction").toPandas()

# Calculate F1-score
f1 = f1_score(label_pred_df['label'], label_pred_df['prediction'], average='micro')
print(f"F1-Score: {f1:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(label_pred_df['label'], label_pred_df['prediction'])
print("Confusion Matrix:\n", conf_matrix)

# Classification report
class_report = classification_report(label_pred_df['label'], label_pred_df['prediction'])
print("Classification Report:\n", class_report)

# Accuracy
accuracy = accuracy_score(label_pred_df['label'], label_pred_df['prediction'])
print(f"Accuracy: {accuracy:.4f}")

# Calculate test error
test_error = label_pred_df[label_pred_df['label'] != label_pred_df['prediction']].shape[0] / float(label_pred_df.shape[0])
print(f"Test Error: {test_error:.4f}")
