# -*- coding: utf-8 -*-

from pyspark.ml.classification import MultilayerPerceptronClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Word2Vec, Word2VecModel, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.pipeline import Pipeline
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import numpy as np
from pyspark.ml.feature import Tokenizer
import os

# init
sc = SparkContext(conf=SparkConf())
spark = SparkSession(sc)

# Lecture des données

df = spark.read.csv("Sentiment Analysis Dataset.csv", header=True)
(train, test) = df.randomSplit([0.7, 0.3])
train.show(20)
tokenizer = Tokenizer(inputCol="SentimentText", outputCol="tokens")
stop_remover = StopWordsRemover(inputCol="tokens", outputCol="words")
label_stringIdx = StringIndexer(inputCol="Sentiment", outputCol="label")

# Utilisation de Word2vec
file = "tokenSave2"
if not os.path.exists(file):
    word2Vec = Word2Vec(inputCol="words", outputCol="SentimentTextTransform")
    word2Vec = word2Vec.fit(train)
    word2Vec.write().overwrite().save(file)
else:
    word2Vec = Word2VecModel.load(file)

#mlp = LogisticRegression(maxIter=10, regParam=0.01, featuresCol ="SentimentTextTransform")
mlp = MultilayerPerceptronClassifier(maxIter=100, layers=[2, 3, 3, 2],
                                     blockSize=1, seed=123, featuresCol="SentimentTextTransform",
                                     predictionCol="prediction")

pipeline = Pipeline(stages=[tokenizer, stop_remover, word2Vec, label_stringIdx, mlp])
model = pipeline.fit(train)

result = model.transform(test)
result.show()
model.write().overwrite().save("mpc")

predictionAndLabels = result.select("prediction", "label")

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
