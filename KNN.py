import numpy as np
import operator

np.set_printoptions(suppress=True)
#导入数据集
def loadDataSet(filename):
	with open(filename, 'rt', encoding='UTF-8') as data_raw:
		data = np.loadtxt(data_raw, delimiter=',')
	return data

#K近邻算法
def KNN(testSamples, trainSamples, trainLabels, K):
	n = testSamples.shape[0]
	labels = np.zeros(n)

	for i in range(n):
		#计算测试样本与每一个训练样本的距离
		all_dist = np.sqrt(np.sum(np.square(testSamples[i] - trainSamples), axis=1))
		maxValue = max(all_dist)
		minValue = min(all_dist)
		weight = (maxValue - all_dist)/(maxValue - minValue)
		#将all_dist按升序排列并保存索引列表
		sort_dist_index = all_dist.argsort()

		classCount = {}
		for j in range(K):
			votelabel = trainLabels[sort_dist_index[j]]
			classCount[votelabel] = classCount.get(votelabel,0) + weight[sort_dist_index[j]]
		#找众数
		sortedclassCount = sorted(classCount.items(),
			key=operator.itemgetter(1), reverse=True)
		labels[i] = sortedclassCount[0][0]

	return labels
'''
def accuracy(testSamples, trainSamples, testLabels, trainLabels):

	labels = KNN(testSamples, trainSamples, trainLabels, 3)
	m = testLabels.shape[0]
	errorCount = 0.0
	for i in range(m):
		if labels[i] != testLabels[i]:
			print('The classifier came back with: %d ,the real answer is: %d'
				% (labels[i], testLabels[i]))
			errorCount += 1
	#print('the error count is %d' % int(errorCount))
	#print('The error rate is %f' % (errorCount/float(m)))

	return 1 - errorCount/float(m)
'''

if __name__ == '__main__':
	trainSamples = loadDataSet('TrainSamples.csv')
	trainLabels = loadDataSet('TrainLabels.csv')
	testSamples = loadDataSet('TestSamples.csv')

	labels = KNN(testSamples, trainSamples,trainLabels, 3)
	np.savetxt('Result.csv', labels, fmt='%d', delimiter=',')


	

