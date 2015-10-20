import numpy as np
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *
from run_knn import *
import matplotlib.pyplot as plt

def correctPredictRate(target, predict):
	dimension = len(target)
	if dimension != len(predict):
		print ('wrong dimension')
		return
	correctNum = 0
	for i in range(dimension):
		if (target[i] == predict[i]):
			correctNum += 1 
	return (float(correctNum)/dimension)

def run():
	train_inputs, train_targets = load_train()
	valid_inputs, valid_targets = load_valid()

	predict_label_dict = {}
	for i in range(10):
		k = 1 + 2*i
		# predict_label_dict[k] = run_knn(k, train_inputs, train_targets, valid_inputs)
		predict_label_dict[k] = run_knn(k, train_inputs, train_targets, train_inputs)
		

	# plot config
	area = np.pi*(3)**2

	for k, predict_label in predict_label_dict.iteritems():
		# rate = correctPredictRate(valid_targets, predict_label)
		rate = correctPredictRate(train_targets, predict_label)
		plt.scatter(k, rate, s=area, alpha=0.8)
	plt.show()

if __name__ == '__main__':
	run()