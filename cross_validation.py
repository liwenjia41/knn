import numpy as np
import KNN

np.set_printoptions(suppress=True)
#导入数据集
def loadDataSet(filename):
	with open(filename, 'rt', encoding='UTF-8') as data_raw:
		data = np.loadtxt(data_raw, delimiter=',')
	return data

#交叉验证
def crossValidation(dataSet, labelSet, K):
	n, d = dataSet.shape[0], dataSet.shape[1]
	m = int(n/K)
	dataGroup = np.zeros((K, m, d))
	labelGroup = np.zeros((K, m))

	start = 0
	for i in range(K):
		dataGroup[i] = dataSet[start:start+m]
		labelGroup[i] = labelSet[start:start+m]
		start += m
	
	accuracy = []
	for i in range(K):
		A = dataSet
		B = labelSet
		samples = np.delete(A, [j for j in range(i*m,(i+1)*m)], axis=0)
		labels = np.delete(B, [j for j in range(i*m,(i+1)*m)], axis=0)
		#print(dataGroup[i].shape)
		#print(samples.shape)
		res = KNN.accuracy(dataGroup[i], samples, labelGroup[i], labels)
		accuracy.append(res)
	return sum(accuracy) / float(len(accuracy))

if __name__ == '__main__':
	TrainSample = loadDataSet('TrainSamples.csv')
	TrainLabel = loadDataSet('TrainLabels.csv')
	print(crossValidation(TrainSample, TrainLabel, 10))

	
