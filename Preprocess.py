from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, pandas_udf
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, FloatType, IntegerType, BooleanType, LongType
import re
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')


spark = SparkSession.builder \
    .appName("ManualLinearRegression") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.executor.instances", "4") \
    .config("spark.executor.memory", "8g") \
    .config("spark.executor.cores", "2") \
    .config("spark.driver.memory", "4g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .getOrCreate()

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

# Define the schema for the review data based on provided fields
review_schema = StructType([
    StructField("rating", FloatType()),
    StructField("title", StringType()),
    StructField("text", StringType()),
    StructField("images", ArrayType(StringType())),
    StructField("asin", StringType()),
    StructField("parent_asin", StringType()),
    StructField("user_id", StringType()),
    StructField("timestamp", LongType()),
    StructField("verified_purchase", BooleanType()),
    StructField("helpful_vote", IntegerType())
])

meta_schema = StructType([
    StructField("main_category", StringType()),
    StructField("title", StringType()),
    StructField("average_rating", FloatType()),
    StructField("rating_number", IntegerType()),
    StructField("features", ArrayType(StringType())),
    StructField("description", ArrayType(StringType())),
    StructField("price", FloatType()),
    StructField("images", ArrayType(StringType())),
    StructField("videos", ArrayType(StringType())),
    StructField("store", StringType()),
    StructField("categories", ArrayType(StringType())),
    StructField("details", StringType()),
    StructField("parent_asin", StringType()),
    StructField("bought_together", ArrayType(StringType()))
])


# Load data with the defined schemas
reviews_df = spark.read.schema(review_schema).json(
    'Data/Review/All_Beauty.jsonl')
meta_data_df = spark.read.schema(meta_schema).json(
    'PreprocessedData/Meta/meta_All_Beauty.jsonl')

# Get the list of column names for each DataFrame
review_columns = set(reviews_df.columns)
meta_columns = set(meta_data_df.columns)

# Find the intersection of both sets to get the duplicate column names
conflicting_columns = review_columns.intersection(meta_columns)
print("Conflicting columns:", conflicting_columns)

meta_data_df = meta_data_df.withColumnRenamed("images", "meta_images") \
                           .withColumnRenamed("title", "meta_title")
reviews_df = reviews_df.withColumnRenamed("title", "review_title")


def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
    text = re.sub(r'\d', '', text)  # Remove digits
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()  # Remove spaces at the beginning and end of the text
    return text


@pandas_udf(StringType())
def preprocess_text_udf_pandas(text_series: pd.Series) -> pd.Series:
    return text_series.apply(preprocess_text)


# Applying text preprocessing UDF
reviews_df = reviews_df.withColumn("review_title", preprocess_text_udf_pandas(col("review_title"))) \
                       .withColumn("text", preprocess_text_udf_pandas(col("text")))

# Joining data on parent_asin without using broadcast
joined_df = reviews_df.join(meta_data_df, "parent_asin")

# Show some of the processed data
joined_df.show()

# Save the joined data
joined_df.write.json('./PreprocessedData/BeautyProcessed-2')

spark.stop()
