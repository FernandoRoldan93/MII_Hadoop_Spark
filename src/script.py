# -*- coding: utf-8 -*-
# Autor: Fernando Roldán Zafra
# Clasificación con spark
from __future__ import print_function
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, StringType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC

#Creation of the spark context
def init_Context():
	conf = SparkConf().setAppName("Practica 4 - Fernando Roldan")
	sc = SparkContext(conf=conf)
	#Do not show warnings on logs
	sc.setLogLevel('ERROR')
	return sc

def read_csv(sc):
	sql = SQLContext(sc)
	df = sql.read.csv("./filteredC.small.training", header=True, inferSchema=True)
	return df

def preprocess(df):
	df = under_sampling(df)
	indexer = StringIndexer(inputCol="PredSS_central_1", outputCol="PredSS_central_1_indexed")

	assembler = VectorAssembler(inputCols=["PSSM_r1_1_N", "PredSS_central_1_indexed", "AA_freq_central_A", 
		"AA_freq_global_H", "PSSM_r1_1_S", "PSSM_r2_-3_Y"], outputCol='features')
	
	pipeline = Pipeline(stages=[indexer, assembler])
	df_1 = pipeline.fit(df).transform(df).select('features', 'class')
	#df = assembler.transform(df).select('features', 'class')
	#df = df.select('features', 'labels')
	scale = StandardScaler(withMean=True, withStd=True, inputCol='features', outputCol='scaled_features')
	scale = scale.fit(df_1)
	df_1 = scale.transform(df_1)
	return df_1


def under_sampling(df):
	df1 = df.filter("class=0")
	df2 = df.filter("class=1")
	ratio = float(df2.count())/float(df1.count())
	sample = df1.sample(withReplacement=False, fraction=ratio, seed=27)
	df = sample.union(df2)
	return df


def tuning(classifier, paramGrid, train):
	tvs = TrainValidationSplit(estimator=classifier,
						   estimatorParamMaps=paramGrid,
						   evaluator=BinaryClassificationEvaluator(),
						   # 80% of the data will be used for training, 20% for validation.
						   trainRatio=0.8)

	# Run TrainValidationSplit, and choose the best set of parameters.
	model = tvs.fit(train)

	ParamMaps = model.getEstimatorParamMaps()
	for i, params in enumerate(ParamMaps):
		print("---------_", str(i), "_---------", " AUC: ", str(model.validationMetrics[i]))
		for param, value in params.items():
			print(param.name, ": ", str(value), "; ", end='')
		print("\n")

	return model.bestModel


def validate(model, test):
	#model = estimator.fit(train)
	eval = BinaryClassificationEvaluator()
	score = eval.evaluate(model.transform(test))
	return score

def train_logistic_regresion(train, test):
	lr = LogisticRegression(labelCol="class", featuresCol="scaled_features")
	
	paramGrid = ParamGridBuilder() \
	.addGrid(lr.maxIter, [5, 10, 20]) \
	.addGrid(lr.regParam, [0.1, 0.01]) \
	.addGrid(lr.elasticNetParam, [0.5, 1]) \
	.build()

	best_model = tuning(lr, paramGrid, train)
	AUC = validate(best_model, test)
	print("LogisticRegression best model AUC", AUC)
	best_model.write().overwrite().save("./logisticRegression_model")

def train_random_forest(train, test):
	rf = RandomForestClassifier(labelCol="class", featuresCol="scaled_features")
	paramGrid = ParamGridBuilder() \
	.addGrid(rf.numTrees, [5, 10, 15]) \
	.addGrid(rf.maxDepth, [3, 5, 8]) \
	.build()
	
	best_model = tuning(rf, paramGrid, train)
	AUC = validate(best_model, test)
	print("Random forest best model AUC", AUC)
	best_model.write().overwrite().save("./random_forest_model")


def train_linearSVC(train, test):
	svc = LinearSVC(labelCol="class", featuresCol="scaled_features")
	
	paramGrid = ParamGridBuilder() \
	.addGrid(svc.maxIter, [3, 5, 8]) \
	.addGrid(svc.regParam, [0.1, 0.05, 0.15]) \
	.build()

	best_model = tuning(svc, paramGrid, train)
	AUC = validate(best_model, test)
	print("Linear SVC best model AUC", AUC)
	best_model.write().overwrite().save("./svc_model")  	
	
if __name__=="__main__":
	sc = init_Context()
	df = read_csv(sc)
	df = preprocess(df)
	df = df.selectExpr("scaled_features","class as label", "class")
	train, test = df.randomSplit([0.8, 0.2], seed=27) 

	print("Starting logistic Regression train...")
	train_logistic_regresion(train, test)
	
	#print("Starting Random Forest train...")
	#train_random_forest(train, test)

	#print("Starting linear SVC train...")
	#train_linearSVC(train, test)
	sc.stop()
