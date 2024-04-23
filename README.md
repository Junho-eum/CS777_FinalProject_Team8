# Sentiment Analysis of Amazon Video Game Reviews: Predicting Customer Satisfaction

## Introduction
Our project analyzes over 1.7 million Amazon video game reviews to build a model that predicts customer satisfaction based on review sentiments. We categorize reviews as negative (1 to 3 stars) or positive (4 to 5 stars) and apply text analysis techniques such as tokenization and TF-IDF.

We use logistic regression and naive Bayes models, chosen for their efficiency with binary outcomes. These models are trained and tested using metrics like precision, recall, F1 score, and AUC, aiming to accurately reflect and predict customer sentiment. The goal is to link these sentiments to actual product ratings, providing insights that can help sellers adjust their strategies and better engage with customers. This work aims to demonstrate the tangible impact of sentiment on customer satisfaction in e-commerce.

## Data Source
For our sentiment analysis, we utilized the Amazon Reviews dataset available from the UCSD Datasets page. Specifically, we focused on the Video Games category, which features a comprehensive collection of user reviews and associated product metadata. This dataset includes data up to the year 2023 and contains approximately 1.7 million user reviews.

The dataset is organized into two primary types: user reviews and product metadata, which are linked using the ASIN product identifier. By merging these two datasets, we created a unified dataset that is ideal for conducting detailed sentiment analysis and big data tasks.

You can access the dataset and learn more about it here: [UCSD Datasets - Amazon Reviews](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews)

#### Prerequisites
- **Python**: Version 3.6 or newer.
- **Apache Spark**: Spark 3.0 or newer is required to handle large-scale data processing.
- **Java**: Java 8 or 11, required for running Spark.
- **NLTK**: The Natural Language Toolkit for Python, used for text processing.
- **Pandas**: For handling data manipulation in Python UDFs.
- **Additional Python Libraries**: `re` for regular expressions, `pandas_udf` from PySpark.

#### Installation Steps
1. **Install Python**: Ensure Python 3.6+ is installed on your system.
2. **Install Java**: Java is needed to run Spark, install either version 8 or 11.
3. **Download and Set Up Apache Spark**: Download Spark from the Apache website and configure it on your machine.
4. **Set up Python Libraries**: Install the required Python libraries using pip:
```bash
pip install nltk pandas pyspark
```

## Data Preprocessing

The primary goal of preprocessing in our project was to refine the data from Amazon's reviews for sentiment analysis, specifically focusing on video games and beauty products. This step is crucial to enhance the quality and effectiveness of our analysis by cleaning and transforming the text data.

#### Process Overview
After loading user reviews and product metadata from JSONL files into separate Spark DataFrames, we first addressed conflicts in column names (e.g., both DataFrames had 'title' and 'images' columns). To resolve these, we renamed them to 'review_title', 'meta_title', and 'meta_images' for clarity.

We merged the reviews DataFrame with the metadata DataFrame using the 'parent_asin' column, enriching the review data with corresponding product metadata. This merging is essential for creating a richer feature set for our models.

The preprocessing steps included converting text to lowercase, removing non-alphanumeric characters, tokenizing the text, and removing common stop words. These steps are implemented in a custom script, `Preprocess.py`, which is designed to run efficiently on large datasets using Apache Spark.

#### Usage
- **Script Execution**: To preprocess the data, run the Preprocess.py script. This script will perform all necessary preprocessing steps on the dataset and prepare it for sentiment analysis.
#### Example
- **Running the Preprocessing Script**:
```bash
spark-submit --executor-memory 8G --driver-memory 4G Preprocess.py
```
This command will execute the Preprocess.py script using Spark, which preprocesses the Amazon reviews data according to the defined schema and text processing functions described in the script.













