##
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import RegexTokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# Show feature vectors and their non-zero counts
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, when
from pyspark.ml.feature import RegexTokenizer, CountVectorizer, IDF, StopWordsRemover
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import MulticlassMetrics
import sys
##
if len(sys.argv) != 3:
    print("Usage: METCS777-term-project.py <data_path> <output_dir_1>",file=sys.stderr)
    exit(-1)
##
spark = SparkSession.builder \
    .appName("Amazon Review Classification with Naive Bayes") \
    .getOrCreate()

##
# Load all JSON data from the directory
data = sys.argv[1]
df = spark.read.json(data)

#Label based on ratings
df = df.withColumn('label', when(col('rating') < 4, 0).otherwise(1))

# Tokenize text
tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
df = tokenizer.transform(df)

# Apply HashingTF
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
featurized_data = hashingTF.transform(df)

##
def non_zero_count(vector):
    return vector.numNonzeros()


non_zero_count_udf = udf(non_zero_count, IntegerType())

#featurized_data.withColumn(
#    "nonZeroCount", non_zero_count_udf(col("rawFeatures"))).show()

##
# Apply IDF
# Use a different column name for IDF output
idf = IDF(inputCol="rawFeatures", outputCol="idfFeatures")
idf_model = idf.fit(featurized_data)
rescaled_data = idf_model.transform(featurized_data)

# Prepare data for modeling
model_data = rescaled_data.select('idfFeatures', 'label')
##
model_data.cache()

major_df = model_data.filter(col("label") == 1)
minor_df = model_data.filter(col("label") == 0)
# Count the number of instances in each class
major_df_count = major_df.count()
minor_df_count = minor_df.count()
# Calculate the fraction to which the minority class needs to be oversampled
oversample_ratio = major_df_count / minor_df_count
# Perform oversampling
oversampled_minor_df = minor_df.sample(withReplacement=True, fraction=oversample_ratio, seed=123)
# Combine the majority class DataFrame and the oversampled minority class DataFrame
balanced_data = major_df.unionAll(oversampled_minor_df)

##
# Split data into training and testing sets
(training_data, test_data) = balanced_data.randomSplit([0.8, 0.2], seed=1234)

# Train Naive Bayes model
nb = NaiveBayes(featuresCol='idfFeatures', labelCol='label', modelType="multinomial")
nb_model = nb.fit(training_data)

# Predict and evaluate the model
predictions = nb_model.transform(test_data)
##
predictions = predictions.withColumn("prediction", col("prediction").cast("double"))
predictions.cache()
# Prepare the prediction and label for MulticlassMetrics
predictionAndLabels = predictions.select("prediction", "label").rdd.map(lambda r: (float(r[0]), float(r[1])))
# Instantiate metrics object
metrics = MulticlassMetrics(predictionAndLabels)
# Collecting metrics into a list of strings
metrics_list = [
    f"Accuracy: {metrics.accuracy}",
    f"Precision (class 0.0): {metrics.precision(0.0)}",
    f"Precision (class 1.0): {metrics.precision(1.0)}",
    f"Weighted Precision: {metrics.weightedPrecision}",
    f"Recall (class 0.0): {metrics.recall(0.0)}",
    f"Recall (class 1.0): {metrics.recall(1.0)}",
    f"Weighted Recall: {metrics.weightedRecall}",
    f"F-measure (class 0.0): {metrics.fMeasure(0.0)}",
    f"F-measure (class 1.0): {metrics.fMeasure(1.0)}",
    f"Weighted F-measure: {metrics.weightedFMeasure()}"
]
# Save to a text file in Spark
spark.sparkContext.parallelize(metrics_list).coalesce(1).saveAsTextFile(sys.argv[2])
##
spark.stop()
##

