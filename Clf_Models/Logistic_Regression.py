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
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
import sys
##
if len(sys.argv) != 3:
    print("Usage: METCS777-term-project.py <data_path> <output_dir_1>",file=sys.stderr)
    exit(-1)
##
spark = SparkSession.builder \
    .appName("Amazon Review Classification with Log Reg") \
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

# Train Logistic Regression model
lr = LogisticRegression(featuresCol='idfFeatures', labelCol='label')
lr_model = lr.fit(training_data)

# Predict and evaluate the model
predictions = lr_model.transform(test_data)
##
predictions.cache()

# Setup Binary and Multiclass evaluators
binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
evaluator_precision = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall")
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")

# Calculate metrics
auc_score = binary_evaluator.evaluate(predictions)
precision = evaluator_precision.evaluate(predictions)
recall = evaluator_recall.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

metrics = [
    f"Precision: {precision}",
    f"Recall: {recall}",
    f"F1 Score: {f1_score}",
    f"AUC: {auc_score}"
]

# Save to a text file in Spark
spark.sparkContext.parallelize(metrics).coalesce(1).saveAsTextFile(sys.argv[2])
##
spark.stop()
##

