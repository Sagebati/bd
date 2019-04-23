from elephas.spark_model import SparkModel
from elephas.utils import to_simple_rdd
from keras_preprocessing.sequence import pad_sequences
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from tensorflow.python.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import numpy as np

sc = SparkContext(conf=SparkConf())
spark = SparkSession(sc)

df = spark.read.csv("csv.csv", header=True)
(train, test) = df.randomSplit([0.7, 0.3])

train_texts = train.select("SentimentText").rdd.flatMap(lambda x: x).collect()
train_labels = train.select("Sentiment").rdd.flatMap(lambda x: x).collect()

test_texts = test.select("SentimentText").rdd.flatMap(lambda x: x).collect()
test_labels = test.select("Sentiment").rdd.flatMap(lambda x: x).collect()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts + test_texts)

train_texts_tokens = tokenizer.texts_to_sequences(train_texts)
test_texts_tokens = tokenizer.texts_to_sequences(test_texts)

num_tokens = [len(tokens) for tokens in train_texts_tokens]
num_tokens = np.array(num_tokens)

max_tokens = 280

pad = 'pre'

x_train_pad = pad_sequences(train_texts_tokens, maxlen=max_tokens, padding=pad, truncating=pad)
x_test_pad = pad_sequences(test_texts_tokens, maxlen=max_tokens, padding=pad, truncating=pad)

rdd = to_simple_rdd(sc, x_train_pad, train_labels)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(len(train_texts), 16))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

sparkmodel = SparkModel(model, frequency='epoch', mode='asynchronous')
model_fited = sparkmodel.fit(rdd, epochs=20, batch_size=32, verbose=0, validation_split=0.1)

rdd_test = to_simple_rdd(sc, x_test_pad, test_labels)
model_fited.predict(rdd)
