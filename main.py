from pyspark.context import SparkContext
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, NaiveBayes
from pyspark.ml.feature import Tokenizer, IDF, HashingTF, StringIndexer, StopWordsRemover
from pyspark.ml.pipeline import Pipeline
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.session import SparkSession


def metrics(test_df, model):
    prediction_and_labels = test_df.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    return BinaryClassificationMetrics(prediction_and_labels)


sc = SparkContext('local')
spark = SparkSession(sc)

df = spark.read.csv('Sentiment Analysis Dataset.csv', header=True)

tokenizer = Tokenizer(inputCol="SentimentText", outputCol="tokens")
stop_remover = StopWordsRemover(inputCol="tokens", outputCol="words")
hash_tf = HashingTF(numFeatures=2 ** 16, inputCol="words", outputCol='tf')
idf = IDF(inputCol='tf', outputCol="features", minDocFreq=5)  # minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol="Sentiment", outputCol="label")

pipeline = Pipeline(stages=[tokenizer, stop_remover, hash_tf, idf, label_stringIdx])

pipelineFit = pipeline.fit(df)
df_fitted = pipelineFit.transform(df)

# df.show(n=10) for debugging purposes

# Data
(trainingData, test_data) = df_fitted.randomSplit([0.7, 0.3])

# Naive Bayes
nb = NaiveBayes(featuresCol="features", labelCol="label")

nb_model = nb.fit(trainingData)
predictions_nb = nb_model.transform(test_data)


# Logistic regression
lr = LogisticRegression(maxIter=100)

lrModel = lr.fit(trainingData)  # learn
predictions_lr = lrModel.transform(test_data)  # test
predictions_lr.show(10)


# Decision tree
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

dt_model = dt.fit(trainingData)
predictions_dt = dt_model.transform(test_data)
predictions_dt.show(5)


