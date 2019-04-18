import tensorflow as tf
from pyspark import SparkContext, SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer
from pyspark.sql import SparkSession
from sparkflow.graph_utils import build_graph
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sparkflow.tensorflow_async import SparkAsyncDL

ctx = SparkContext(conf=SparkConf())
spark = SparkSession(ctx)


def model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(280, 16))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    # model.compile(optimizer='adam',
    # loss = 'binary_crossentropy',
    # metrics = ['accuracy'])
    # pad = 'pre'

    # x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
    #                            padding=pad, truncating=pad)

    # x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
    #                           padding=pad, truncating=pad)
    return model


df = spark.read.csv('Sentiment Analysis Dataset.csv', header=True)
mg = build_graph(model)

# Assemble and one hot encode

spark_model = SparkAsyncDL(
    inputCol='features',
    tensorflowGraph=mg,
    tfInput='x:0',
    tfLabel='y:0',
    tfOutput='out:0',
    tfLearningRate=.001,
    iters=2,
    predictionCol='prediction',
    labelCol='label',
    verbose=2
)

tokenizer = Tokenizer(inputCol="SentimentText", outputCol="feature")
(trainingData, test_data) = df.randomSplit([0.7, 0.3])
pipeline_model = Pipeline(stages=[tokenizer, spark_model]).fit(trainingData)

pipeline_model.write().overwrite().save("nn")
