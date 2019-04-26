# -*- enconding: utf-8 -*-

from pyspark import SparkContext, SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, StopWordsRemover, Tokenizer, HashingTF, IDF
from pyspark.sql import SparkSession

sc = SparkContext(conf=SparkConf())
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("OFF")
# Lecture des données

df = spark.read.csv("Sentiment Analysis Dataset.csv", header=True)

tokenizer = Tokenizer(inputCol="SentimentText", outputCol="tokens")
stop_remover = StopWordsRemover(inputCol="tokens", outputCol="words")
hash_tf = HashingTF(numFeatures=2 ** 16, inputCol="words", outputCol='tf')  # Term frequencies
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol="Sentiment", outputCol="label")

pipeline = Pipeline(stages=[tokenizer, stop_remover, hash_tf, idf, label_stringIdx])

pipelineFit = pipeline.fit(df)
df_fitted = pipelineFit.transform(df)

for sample_size in (0.001, 0.002, 0.005, 0.01, 0.015, 0.020, 0.025):
    sample = df_fitted.sample(withReplacement=False, fraction=sample_size)
    bayes_model = NaiveBayes().fit(sample)
    lr_model = LogisticRegression().fit(sample)
    dt_model = DecisionTreeClassifier().fit(sample)

    bayes_pr = bayes_model.transform(sample)
    lr_pr = lr_model.transform(sample)
    dt_pr = dt_model.transform(sample)

    evaluator_lr = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    evaluator_dt = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    evaluator_bayes = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
    print('Bayes Test Area Under ROC sample size : ', sample_size, "score : ", evaluator_bayes.evaluate(bayes_pr))
    print('Lr Test Area Under ROC sample size : ', sample_size, "score : ", evaluator_lr.evaluate(lr_pr))
    print('Dt Test Area Under ROC sample size : ', sample_size, "score : ", evaluator_dt.evaluate(dt_pr))