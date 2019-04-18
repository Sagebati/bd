import os

from pyspark import SparkConf
from pyspark.context import SparkContext
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, DecisionTreeClassificationModel, \
    NaiveBayes, LogisticRegressionModel, \
    NaiveBayesModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import Tokenizer, IDF, HashingTF, StringIndexer, StopWordsRemover
from pyspark.ml.pipeline import Pipeline
from pyspark.sql.session import SparkSession

sc = SparkContext(conf=SparkConf())
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("OFF")
df = spark.read.csv("Sentiment Analysis Dataset.csv", header=True)

tokenizer = Tokenizer(inputCol="SentimentText", outputCol="tokens")
stop_remover = StopWordsRemover(inputCol="tokens", outputCol="words")
hash_tf = HashingTF(numFeatures=2 ** 16, inputCol="words", outputCol='tf')  # Term frequencies
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol="Sentiment", outputCol="label")

pipeline = Pipeline(stages=[tokenizer, stop_remover, hash_tf, idf, label_stringIdx])

pipelineFit = pipeline.fit(df)
df_fitted = pipelineFit.transform(df)

# df.show(n=10) for debugging purposes

# Data
(trainingData, test_data) = df_fitted.randomSplit([0.7, 0.3])
trainingData.cache()
test_data.cache()

# Naive Bayes
nb = NaiveBayes(featuresCol="features", labelCol="label")

# Learn if the ml doesn't exist
nb_model = NaiveBayesModel.load("bayes") if os.path.exists("bayes") else nb.fit(trainingData)
predictions_nb = nb_model.transform(test_data)

nb_model.write().overwrite().save('bayes')
predictions_nb.show(10)

# Compute raw scores on the test set
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print('Bayes Test Area Under ROC', evaluator.evaluate(predictions_nb))

# Logistic regression
lr = LogisticRegression(maxIter=100)

lrModel = LogisticRegressionModel.load("lr") if os.path.exists("lr") else lr.fit(trainingData)
predictions_lr = lrModel.transform(test_data)  # test
predictions_lr.show(10)

lrModel.write().overwrite().save('lr')

print('LR Test Area Under ROC', evaluator.evaluate(predictions_lr))

# Decision tree
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

dt_model = DecisionTreeClassificationModel.load("dt") if os.path.exists("dt") else dt.fit(trainingData)
predictions_dt = dt_model.transform(test_data)
predictions_dt.show(5)

dt_model.write().overwrite().save('dt')

print('DT Test Area Under ROC', evaluator.evaluate(predictions_dt))

# Reseau neurones

