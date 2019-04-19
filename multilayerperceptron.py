from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Word2Vec
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import numpy as np

# init
sc = SparkContext(conf=SparkConf())
spark = SparkSession(sc)

# Lecture des données ;

df = spark.read.csv("Sentiment Analysis Dataset.csv", header=True)
(train, test) = df.randomSplit([0.7, 0.3])

df.withColumn("SentimentText", np.array(df.select("SentimentText")).split(" "))


train_texts = train.select("SentimentText").rdd.flatMap(lambda x: x).collect()
train_labels = train.select("Sentiment").rdd.flatMap(lambda x: x).collect()

test_texts = test.select("SentimentText").rdd.flatMap(lambda x: x).collect()
test_labels = test.select("Sentiment").rdd.flatMap(lambda x: x).collect()

# Utilisation de Word2vec

word2Vec = Word2Vec(inputCol="SentimentText", outputCol="SentimentTextTransform")
model = word2Vec.fit(df.select("SentimentText").split(" "))
result = model.transform(' '.join(train_texts))
for row in result.collect():
    text, vector = row
    print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))