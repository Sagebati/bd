# -*- coding: utf-8 -*-

import time

from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import Word2Vec, StopWordsRemover, StringIndexer, CountVectorizer
from pyspark.ml.pipeline import Pipeline
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
stop_remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
countVectors = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=140, minDF=5)
label_stringIdx = StringIndexer(inputCol="Sentiment", outputCol="label")

word2vec = Word2Vec(inputCol="filtered", outputCol="features", maxSentenceLength=140)

# mlp = LogisticRegression(maxIter=10, regParam=0.01, featuresCol ="SentimentTextTransform")
mlp = MultilayerPerceptronClassifier(maxIter=100, layers=[140, 70, 50, 2],
                                     blockSize=64, seed=123,
                                     predictionCol="prediction")

pipeline = Pipeline(stages=[tokenizer, stop_remover, countVectors, label_stringIdx, mlp])

for sample_size in (0.001, 0.002, 0.005, 0.01, 0.015, 0.020, 0.025):
    sample = df.sample(withReplacement=False, fraction=sample_size)

    model, duration = calculate_time(lambda: pipeline.fit(sample))
    predictionAndLabels = model.transform(df).select("prediction", "label")

    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)), " time: ", duration
          , " sample : ", sample_size)
