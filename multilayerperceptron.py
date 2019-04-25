# -*- coding: utf-8 -*-

from pyspark.ml.classification import MultilayerPerceptronClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Word2Vec, Word2VecModel, StopWordsRemover, HashingTF, IDF, StringIndexer, CountVectorizer
from pyspark.ml.pipeline import Pipeline, PipelineModel
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import numpy as np
from pyspark.ml.feature import Tokenizer
import os
from pyspark.sql.functions import lpad

# init
sc = SparkContext(conf=SparkConf())
spark = SparkSession(sc)

# Lecture des données

df = spark.read.csv("Sentiment Analysis Dataset.csv", header=True)

(train, test) = df.randomSplit([0.9, 0.1])
train.show(20)
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
model = pipeline.fit(train) if os.path.exists("mlp") else PipelineModel.load("mlp")

result = model.transform(test)
result.show()

model.write().overwrite().save("mlp")

predictionAndLabels = result.select("prediction", "label")

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
