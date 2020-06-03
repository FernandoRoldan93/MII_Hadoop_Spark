# -*- coding: utf-8 -*-
# Autor: Fernando Roldán Zafra

from pyspark import SparkContext, SparkConf, SQLContext

#Creation of the spark context
def init_Context():
	conf = SparkConf().setAppName("Practica 4 - Fernando Roldan")
	sc = SparkContext(conf=conf)
	#Do not show warnings on logs
	sc.setLogLevel('ERROR')
	return sc


def create_dataset(sc):
	# Creation of the dataframe that will be used for the model. It will generate a .csv file containing all the data.
	headers = sc.textFile("/user/datasets/ecbdl14/ECBDL14_IR2.header").collect()
	headers = list(filter(lambda x:"@inputs" in x, headers))[0]
	headers = headers.replace(",", "").strip().split()
	del headers[0]
	headers.append("class")

	sqlc = SQLContext(sc)
	df = sqlc.read.csv(
		"/user/datasets/ecbdl14/ECBDL14_IR2.data", header=False, inferSchema=True)

	for i, col in enumerate(df.columns):
		df = df.withColumnRenamed(col, headers[i])
	
	df = df.select(
		"PSSM_r1_1_N", "PredSS_central_1", "AA_freq_central_A", 
		"AA_freq_global_H", "PSSM_r1_1_S", "PSSM_r2_-3_Y", "class")
	
	df.write.csv('./filteredC.small.training', header=True, mode="overwrite")    
	print("Done")
	
if __name__=="__main__":
	sc = init_Context()
	create_dataset(sc)
	sc.stop()