import sys

from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import Tokenizer, IDF, HashingTF, StopWordsRemover
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.session import SparkSession
from pyspark.sql.types import Row

sc = SparkContext(conf=SparkConf())
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("OFF")

lrModel = LogisticRegressionModel.load("lr")

tokenizer = Tokenizer(inputCol="_1", outputCol="tokens")
stop_remover = StopWordsRemover(inputCol="tokens", outputCol="words")
hash_tf = HashingTF(numFeatures=2 ** 16, inputCol="words", outputCol='tf')  # Term frequencies
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms

pipeline = Pipeline(stages=[tokenizer, stop_remover, hash_tf, idf])

# for line in sys.stdin:
df = spark.createDataFrame([("I'll kill you",)])
pipelineModel = pipeline.fit(df)
prediction = lrModel.transform(pipelineModel.transform(df))
prediction.show()
