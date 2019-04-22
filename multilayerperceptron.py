# -*- coding: utf-8 -*-

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Word2Vec
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import numpy as np
from pyspark.ml.feature import Tokenizer
import os

# init
sc = SparkContext(conf=SparkConf())
spark = SparkSession(sc)

# Lecture des données

df = spark.read.csv("csv.csv", header=True)
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
if not os.path.isfile("tokenSave"):
    word2Vec = Word2Vec(inputCol="words", outputCol="SentimentTextTransform")
    model = word2Vec.fit(tok.select("words"))
    tokenizer.save("tokenSave")
tokenizer = Tokenizer.load("/Users/tanguyherserant/Desktop/bdd/saves/tokenSave")
result = model.transform(tok.select("words"))

result.show()
# for row in result.collect():
 #   text, vector = row
 #   print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))
