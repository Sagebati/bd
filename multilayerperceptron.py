# -*- coding: utf-8 -*-

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Word2Vec, Word2VecModel

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
#text_word = (i.split(" ") for i in df["SentimentText"].collect())
#df.withColumn("SentimentText", df.select("SentimentText").split(" "))
#text_word = df.rdd.map(lambda x: ("SentimentText").split(" ")).show()

tokenizer = Tokenizer(inputCol="SentimentText", outputCol="words")
tok = tokenizer.transform(df)
df.show(20)
tok.select("SentimentText", "words").show(20)
train_texts = train.select("SentimentText").rdd.flatMap(lambda x: x).collect()
train_labels = train.select("Sentiment").rdd.flatMap(lambda x: x).collect()

test_texts = test.select("SentimentText").rdd.flatMap(lambda x: x).collect()
test_labels = test.select("Sentiment").rdd.flatMap(lambda x: x).collect()

# Utilisation de Word2vec
file = "tokenSave2"
if not os.path.exists(file):
    word2Vec = Word2Vec(inputCol="words", outputCol="SentimentTextTransform")
    word2Vec = word2Vec.fit(tok.select("words"))
    word2Vec.write().overwrite().save(file)
else:
    word2Vec = Word2VecModel.load(file)

result = word2Vec.transform(tok.select("words"))
result.show(20)
word2Vec.getVectors().show()
split = result.select("words").randomSplit([0.6, 0.4])
train = split[0]
test = split[1]
mlp = MultilayerPerceptronClassifier(maxIter=100, layers=[256, 128, 2], blockSize=1, seed=123)
train = train.withColumnRenamed('words', 'features').collect()
model = mlp.fit(train)

test = test.withColumnRenamed('words', 'features').collect()

result = model.transform(test)
predictionAndLabels = result.select("prediction", "Sentiment")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))