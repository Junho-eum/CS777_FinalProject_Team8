import os
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, to_timestamp
from pyspark.sql.types import FloatType, TimestampType, StructType, StructField

pd.set_option('display.max_rows', None)

# Initialize NLTK and VADER sentiment analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Amazon Review Sentiment Analysis") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

# Define a UDF to calculate sentiment score using VADER


def sentiment_score(review_text):
    if review_text:
        return sia.polarity_scores(review_text)['compound']
    return 0.0


# Register UDF with Spark
sentiment_udf = udf(sentiment_score, FloatType())

# Load the preprocessed JSON data
df = spark.read.json(
    "/Users/junhoeum/Desktop/CS777/FinalProject/PreprocessedData/VideoGamePreprocessed")

# Convert timestamps from UNIX time to TimestampType
df = df.withColumn("timestamp", to_timestamp(col("timestamp") / 1000))

# Apply the sentiment UDF to calculate sentiment scores
df = df.withColumn("sentiment_score", sentiment_udf(col("text")))

# Select relevant columns for plotting
df = df.select("timestamp", "sentiment_score")

# Collect data to Pandas DataFrame for plotting
pdf = df.toPandas()

# Convert timestamp to the correct datetime format for monthly aggregation
pdf['month'] = pdf['timestamp'].dt.to_period('M').apply(lambda r: r.start_time)

# Group by month and calculate mean sentiment score
monthly_sentiment = pdf.groupby(
    'month')['sentiment_score'].mean().reset_index()


# Convert timestamp to the correct datetime format for quarterly aggregation
pdf['quarter'] = pdf['timestamp'].dt.to_period(
    'Q').apply(lambda r: r.start_time)

# Group by quarter and calculate mean sentiment score
quarterly_sentiment = pdf.groupby(
    'quarter')['sentiment_score'].mean().reset_index()

# Print the quarterly average sentiment scores
print(quarterly_sentiment)
# Plotting the sentiment scores over time using Plotly, aggregated by month
fig = px.line(
    monthly_sentiment,
    x='month',
    y='sentiment_score',
    title='Monthly Sentiment Trend of Video Game Products Over Time',
    labels={'month': 'Month', 'sentiment_score': 'Average Sentiment Score'},
    markers=True  # Add markers to each data point
)

# Update layout for better readability and to make the plot responsive
fig.update_layout(
    autosize=True,  # Make the plot responsive
    plot_bgcolor='white',
    xaxis_title='Month',
    yaxis_title='Average Sentiment Score',
    xaxis=dict(
        showline=True,
        showgrid=False,
        linecolor='black',
    ),
    yaxis=dict(
        showline=True,
        showgrid=True,
        gridcolor='lightgrey',
        linecolor='black',
    ),
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="black"
    ),
    hovermode='x'  # Improves hover interaction
)

# Optionally, add hover template for detailed hover info
fig.update_traces(
    line=dict(color='red'),
    hovertemplate="Month: %{x} <br>Average Sentiment Score: %{y:.2f}"
)

# Make the rangeslider and other interactive components responsive as well
fig.update_xaxes(rangeslider=dict(
    bgcolor="lightgray",
    autorange=True
))

# Show the figure
fig.show()
