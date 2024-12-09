# Install necessary packages in Colab
!pip install pyspark
!pip install findspark

# Configure Spark
import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

# Start Spark session
spark = SparkSession.builder \
    .appName("Wine Quality Classification") \
    .master("local") \
    .getOrCreate()

# Load the dataset
file_path = "TrainingDataset.csv" 
df = spark.read.csv(file_path, header=True, sep=";")

# Debug: Print original column names
print("Original column names:", df.columns)

# Standardize column names by removing quotes and spaces
df = df.toDF(*[c.replace('"', '').strip() for c in df.columns])
print("Cleaned column names:", df.columns)

# Rename 'quality' column to 'label'
df = df.withColumnRenamed("quality", "label")

# Cast feature columns to float
for column in df.columns[:-1]: 
    df = df.withColumn(column, df[column].cast("float"))
df = df.withColumn("label", df["label"].cast("float"))

# cleaned and processed data
df.show(5)

# Assemble features into a single vector
feature_columns = df.columns[:-1] 
vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = vector_assembler.transform(df).select("features", "label")

# sample of the transformed data
data.show(5)

# Split data into training and testing sets
train_data, test_data = data.randomSplit([0.7, 0.3], seed=42)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=20, maxDepth=10)
rf_model = rf_classifier.fit(train_data)

# Make predictions
predictions = rf_model.transform(test_data)

# Evaluate the model using evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy:.4f}")

# Evaluate using additional metrics
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator_f1.evaluate(predictions)
print(f"F1 Score: {f1_score:.4f}")

# Show confusion matrix using Pandas
predictions_pd = predictions.select("label", "prediction").toPandas()
confusion_matrix = pd.crosstab(predictions_pd['label'], predictions_pd['prediction'], rownames=['Actual'], colnames=['Predicted'])
print("\nConfusion Matrix:")
print(confusion_matrix)

# Save the model locally
model_path = "rf_wine_quality_model"
rf_model.save(model_path)
print(f"Model saved at: {model_path}")
