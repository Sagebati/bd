# -*- coding: utf-8 -*-

import time

from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.feature import Word2Vec, StopWordsRemover, StringIndexer, CountVectorizer
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession


def calculate_time(f):
    start = time.time()
    model = f()
    end = time.time()
    return model, end - start


# init
sc = SparkContext(conf=SparkConf())
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("OFF")

# Lecture des données
df = spark.read.csv("Sentiment Analysis Dataset.csv", header=True)

tokenizer = Tokenizer(inputCol="SentimentText", outputCol="tokens")
stop_remover = StopWordsRemover(inputCol="tokens", outputCol="words")
hash_tf = HashingTF(numFeatures=2 ** 16, inputCol="words", outputCol='tf')  # Term frequencies
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol="Sentiment", outputCol="label")

rf = RandomForestRegressor()

pipeline = Pipeline(stages=[tokenizer, stop_remover, hash_tf, idf, label_stringIdx, rf])

for sample_size in (0.001, 0.002, 0.005, 0.01, 0.015, 0.020, 0.025):
    sample = df.sample(withReplacement=False, fraction=sample_size)

    model, duration = calculate_time(lambda: pipeline.fit(sample))
    predictionAndLabels = model.transform(df).select("prediction", "label")

    evaluator = RegressionEvaluator().setMetricName("rmse")
    print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)), " time: ", duration
          , " sample : ", sample_size)
