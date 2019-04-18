import tensorflow as tf
from pyspark import SparkContext, SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer
from pyspark.sql import SparkSession
from systemml.mllearn import Keras2DML
from sparkflow.tensorflow_async import SparkAsyncDL
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

ctx = SparkContext(conf=SparkConf())
spark = SparkSession(ctx)


def model():
    m = tf.keras.Sequential()
    m.add(tf.keras.layers.Embedding(280, 16))
    m.add(tf.keras.layers.GlobalAveragePooling1D())
    m.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
    m.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m


pad = 'pre'

x_train_pad = pad_sequences(, maxlen=max_tokens, padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens, padding=pad, truncating=pad)

sysml_model = Keras2DML(spark, model(), input_shape=(280,))
df = spark.read.csv('Sentiment Analysis Dataset.csv', header=True)

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
