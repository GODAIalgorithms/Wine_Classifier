#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission.
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function

import re
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_data, print_features, print_predictions
from scipy import stats

from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
from sklearn.decomposition import PCA
from sklearn import preprocessing

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

features_3 = [6,9,12]
features_2 = [6,9]

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']

train_set, train_labels, test_set, test_labels = load_data()



def calculate_confusion_matrix(gt_labels, pred_labels):
	labels = list(set(gt_labels))
	class_count = len(labels)
	confusion = np.zeros((class_count, class_count), dtype=float)
	for i in range(class_count):
		counts = 0.0 + (gt_labels == i+1).sum()
		for j in range(class_count):
			number = 0.0
			for x in range(len(gt_labels)):
				if gt_labels[x] == labels[i] and pred_labels[x] == labels[j]:
					number = number + 1
			result=number/counts
			print(number, "/", counts)
			confusion[i,j] = number/counts
	return confusion

def plot_matrix( matrix, k, ax=None, title='Confusion Matrix', xlabel='', ylabel=''):
	if ax is None:
		ax = plt.gca()
	cmap = plt.get_cmap('Blues')
	handle = plt.imshow(matrix, cmap=cmap)
	plt.colorbar(handle)
	for i in range(len(matrix)):
		for j in range(len(matrix)):
			plt.text(j,i,"{0:.2f}".format(matrix[i,j]), horizontalalignment='center')
	plt.xticks(np.arange(0, len(matrix), step=1))
	plt.yticks(np.arange(0, len(matrix), step=1))
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title("Confusion Matrix for k = {}".format(k))
	# plt.savefig("Confusion Matrix for k = {}.png".format(k))
	plt.show()

def feature_selection(train_set, test_set, train_labels, **kwargs):
	# write your code here and make sure you return the features at the end of
	# the function
#	n_features = train_set.shape[1]
#	fig, ax = plt.subplots(n_features, n_features, figsize=(100,100))
#	plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=1.2)
#	colours = np.zeros_like(train_labels, dtype=np.object)
#	colours[train_labels == 1] = CLASS_1_C
#	colours[train_labels == 2] = CLASS_2_C
#	colours[train_labels == 3] = CLASS_3_C
#	for x in range(n_features):
#		for y in range(n_features):
#			ax[x,y].scatter(train_set[:,x], train_set[:,y], c = colours, s=3)
#			ax[x,y].tick_params(axis='both', which='major', labelsize=7)
#			ax[x,y].tick_params(axis='both', which='minor', labelsize=7)
#			ax[x,y].set_title('{} vs {}'.format(x+1,y+1))
#	plt.savefig("Feature Selection.png")
#	plt.show()
	# 	# THIS SECTION IS JUST TO PLOT OUR TWO SELECTED FEATURES
#	train_set_selected = np.column_stack((train_set[:,6], train_set[:,9]))
#	fig, ax = plt.subplots()
#	ax.scatter(train_set_selected[:, 0], train_set_selected[:, 1], c=colours)
#	ax.set_title('Features 7 vs 10')
	#plt.savefig("Feature Selected.png")

	return [6,9]

def plot_3_features(train_set, test_set, train_labels, **kwargs):
#########graph needs resizing but otherwise works for 12 vs n and 9 vs n
	train_set_reduced = np.column_stack((train_set[:,6], train_set[:,9]))
	test_set_reduced = np.column_stack((test_set[:,6], test_set[:,9]))

	n_features = train_set.shape[1]
	fig, ax = plt.subplots(n_features, 2)
	   # plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.2, hspace=0.4)
	colours = np.zeros_like(train_labels, dtype=np.object)
	colours[train_labels == 1] = CLASS_1_C
	colours[train_labels == 2] = CLASS_2_C
	colours[train_labels == 3] = CLASS_3_C

	for x in range(n_features):
		for y in range(2):
			if y == 0:
				fea = 6
			elif y == 1:
				fea = 9
			ax[x,y].scatter(train_set[:,x], train_set[:,fea], c = colours, s = 3)
			ax[x,y].tick_params(axis='both', which='major', labelsize=7)
  #          ax[x,y].set_title('Feature {} vs Feature {}'.format(x+1, fea+1))
	plt.savefig("Feature Selection for 3D with 2D graphs.png")
	plt.show()

def plot_3d_graphs(train_set, test_set, train_labels, **kwargs):
############ works but needs resizing as not readable
	n_features = train_set.shape[1]
	colours = np.zeros_like(train_labels, dtype=np.object)
	colours[train_labels == 1] = CLASS_1_C
	colours[train_labels == 2] = CLASS_2_C
	colours[train_labels == 3] = CLASS_3_C
	fig = plt.figure(figsize=(50,100))
	for x in range(n_features-1):
		ax = fig.add_subplot(3, 4, x+1, projection='3d')
		ax.scatter(train_set[:,x], train_set[:,9], train_set[:,12], c =colours, s=20)
		ax.set_xlabel('{}'.format(x+1))
		ax.set_ylabel('10')
		ax.set_zlabel('13')
		ax.set_title('{} v {} v {}'.format(x+1, 10, 13))
	plt.savefig("Feature Selection for 3D with 3D graphs")
	plt.show()


def nearest_nodes(train_set, test_set, train_labels, k):
	dist = lambda x, y: np.sqrt(np.sum((x-y)**2))
	train_dist = lambda x : [dist(x, point) for point in train_set]
	predictions = np.zeros((len(test_set),1))
	original_k = k
	for x in range(len(test_set)):
			success = 0
			k = original_k
			while ( success == 0 ):
				result = knn_predictions(test_set, train_labels, train_dist, x, k)
				if (result == -1 ):
					k = k-1
				else:
					predictions[x] = result
					success = 1
	return predictions

def knn_predictions(test_set, train_labels, train_dist, x, k):
#	dist = lambda x, y: np.sqrt(np.sum((x-y)**2))
#	train_dist = lambda x : [dist(x, point) for point in train_set]
	temp = train_dist(test_set[x])
	temp_with_labels = np.column_stack((temp,train_labels))
	temp_with_labels = temp_with_labels[temp_with_labels[:,0].argsort()]
	knns = temp_with_labels[0:k,:]
	votes = np.zeros((3,1))
	for y in range(k):
		index = int(knns[y,1]-1)
		votes[index] += 1
	a = votes.max()
	count = 0
	for i in range(3):
		if (votes[i] == a):
			count += 1
   # if( count > 1 ):
	#    return -1;
   # else:
	for y in range(3):
			if ( votes[y] == a ):
				return y+1


def knn(train_set, train_labels, test_set, test_labels, k, **kwargs):
	# write your code here and make sure you return the predictions at the end of
	# the function

	train_set_selected = np.column_stack((train_set[:,6], train_set[:,9]))
	test_set_selected = np.column_stack((test_set[:,6], test_set[:,9]))
	result_labels = nearest_nodes(train_set_selected, test_set_selected, train_labels, k)
	accuracy = 0.0
	for x in range(len(test_set)):
		if (result_labels[x] == test_labels[x]):
				accuracy += 1
	accuracy = accuracy / len(test_set)
#	print(k, " accuracy ", accuracy)
#	confusion_matrix = calculate_confusion_matrix(test_labels, result_labels)
#	plot_matrix(confusion_matrix, k)
		#print(result_labels)

	return result_labels


##################################################################
##############################################Naive Bayes Analysis
def mean(numbers):
	return np.mean(number, axis=0)

def cov(numbers):
	return np.cov(numbers, rowvar=False)
def stdev(numbers):
	return np.sqrt(var(number))

#prior pdf
def calculate_prior_probability(means, train_set, stdev, test_labels):
	classes = np.unique(test_labels)
	num= np.array([np.sum(test_labels==c)for c in classes])
	num_1= num[0]/len(train_set)
	num_2 =num[1]/len(train_set)
	num_3 =num[2]/len(train_set)
#	print('num is ',num)

	return num_1, num_2, num_3

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
#likelihood P(x|theta)
def calculateLikelihood(means,test_set, stdev, test_labels):
	results = []
	temp = []
	for i in range(len(test_labels)):
		# print('test_set is ', np.shape(test_set))
		# print('test_set[i][0] is',test_set[i][0])
		prob_3 = calculateProbability(test_set[i][0], means, stdev)
		prob_4= calculateProbability(test_set[i][1], means, stdev)
		# print('prob_3 is', prob_3)
		# print('prob_4 is', prob_4)
		temp.append([prob_3*prob_4])
	temp = np.array(temp)
	return temp

def alternative_classifier(train_set, train_labels, test_set, **kwargs):
	train_set_selected = np.column_stack((train_set[:,6], train_set[:,9]))
	test_set_selected = np.column_stack((test_set[:,6], test_set[:,9]))
#	print('test_set_selected is  ',np.shape(test_set_selected))
#1) Summarize Training Dataset
#1.1 Seperate Data by Class
# def separateByClass(train_set, train_labels, test_set):
	classes = [[],[],[]]
	for i in range(len(train_labels)):
		if train_labels[i] == 1:
			classes[0].append(train_set_selected[i])
		elif train_labels[i] == 2:
			classes[1].append(train_set_selected[i])
		elif train_labels[i] == 3:
			 classes[2].append(train_set_selected[i])
			#class_3.append(train_set_selected[i])
	# print()

	means = []
	covs = []
	stdev=[]
	for i in range(0,3):
		means.append(np.mean(classes[i]))
		# covs.append(np.cov(classes[i]))
		stdev.append(np.std(classes[i]))
	likelihood1 = calculateLikelihood(means[0], test_set_selected, stdev[0], test_labels)
	likelihood2 = calculateLikelihood(means[1], test_set_selected, stdev[1], test_labels)
	likelihood3 = calculateLikelihood(means[2], test_set_selected, stdev[2], test_labels)
	prior_class = calculate_prior_probability(0, train_set_selected, 0, train_labels)


	# print('likelihood is' , likelihood1, likelihood2, likelihood3)
	#posterior probability
	likelihood1 = np.array(likelihood1)
	likelihood2 = np.array(likelihood2)
	likelihood3 = np.array(likelihood3)
	likelihood = []
	posterior1=[] #result of posterior probability
	posterior2=[]
	posterior3=[]
	posterior=[]

	for i in range(len(likelihood1)):
		for j in range(3):
			locals()['posterior'+str(j+1)].append(locals()['likelihood'+str(j+1)][i]*prior_class[j])

		blah = np.argsort((float(posterior1[i]),float(posterior2[i]),float(posterior3[i])))[2]+1
		posterior.append(blah)  #SELECETING POSTERIOR in order

#2.5 accuracy calculation
# def getAccuracy(test_set, predictions):
	numOfCorrect = 0.0
	for x in range(len(test_set)):
		if (posterior[x] == test_labels[x]):
			numOfCorrect +=1
	accuracy = numOfCorrect / len(test_set)
#	print((numOfCorrect/float(len(test_set)) * 100.0))
	return posterior

############################################################
########################################Naive Bayes Analysis
def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
	# write your code here and make sure you return the predictions at the end of
	# the function

#	plot_3_features(train_set, test_set, train_labels)
	# plot_3d_graphs(train_set, test_set, train_labels)

	train_set_selected = np.column_stack((train_set[:,6], train_set[:,9], train_set[:,12]))
	test_set_selected = np.column_stack((test_set[:,6], test_set[:,9], test_set[:,12]))

	result_labels = nearest_nodes_3d(train_set_selected, test_set_selected, train_labels, k)
	accuracy = 0.0
	for x in range(len(test_set)):
		if (result_labels[x] == test_labels[x]):
				accuracy += 1
	accuracy = accuracy / len(test_set)
#	print(k, " accuracy ", accuracy)
#        confusion_matrix = calculate_confusion_matrix2(test_labels, result_labels)
#        plot_matrix(confusion_matrix, k)

	return result_labels
#	return []



def nearest_nodes_3d(train_set, test_set, train_labels, k):
	dist = lambda x, y: np.sqrt(np.sum([(a-b)**2 for a,b in zip(x,y)]))
	train_dist = lambda x : [dist(x, point) for point in train_set]

	predictions = np.zeros((len(test_set),1))
	original_k = k
	for x in range(len(test_set)):
		success = 0
		k = original_k
		while ( success == 0 ):
			result = knn_predictions(test_set, train_labels, train_dist, x, k)
			if (result == -1 ):
				k = k-1
			else:
				predictions[x] = result
				success = 1
	return predictions


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
	# write your code here and make sure you return the predictions at the end of
	# the function

	   # train_set = preprocessing.normalize(train_set)
	  #  test_set = preprocessing.normalize(test_set)

	pca = PCA(n_components=n_components)
	pca.fit(train_set)
	scipy_train = pca.transform(train_set)
	scipy_test = pca.transform(test_set)
	scipy_train[:,1] *= -1
	scipy_test[:,1] *= -1

		#scipy_train = preprocessing.normalize(scipy_train)

	colours = np.zeros_like(train_labels, dtype=np.object)
	colours[train_labels == 1] = CLASS_1_C
	colours[train_labels == 2] = CLASS_2_C
	colours[train_labels == 3] = CLASS_3_C

	new_train = np.column_stack((train_set[:,6], train_set[:,9]))

	fig, ax = plt.subplots(1,2)
	ax[0].scatter(new_train[:,0], new_train[:,1], c = colours)
	ax[1].scatter(scipy_train[:,0], scipy_train[:,1], c = colours)
	ax[0].set_title("Manual Feature Selection")
	ax[1].set_title("PCA Feature Selection")
#	plt.savefig("PCA vs manual select.png")
#	plt.show()


	scipy_knn = nearest_nodes(scipy_train, scipy_test, train_labels, k)
	accuracy = 0.0
	for x in range(len(test_set)):
		if (scipy_knn[x] == test_labels[x]):
				accuracy += 1
	accuracy = accuracy / len(test_set)
#	print(k, " accuracy ", accuracy)
	# confusion_matrix = calculate_confusion_matrix(test_labels, scipy_knn)
	# plot_matrix(confusion_matrix, k)
        
	return scipy_knn, scipy_train


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('mode', nargs=1, type=str, help='Running mode. Must be one of the following modes: {}'.format(MODES))
	parser.add_argument('--k', nargs='?', type=int, default=1, help='Number of neighbours for knn')
	parser.add_argument('--train_set_path', nargs='?', type=str, default='data/wine_train.csv', help='Path to the training set csv')
	parser.add_argument('--train_labels_path', nargs='?', type=str, default='data/wine_train_labels.csv', help='Path to training labels')
	parser.add_argument('--test_set_path', nargs='?', type=str, default='data/wine_test.csv', help='Path to the test set csv')
	parser.add_argument('--test_labels_path', nargs='?', type=str, default='data/wine_test_labels.csv', help='Path to the test labels csv')

	args = parser.parse_args()
	mode = args.mode[0]

	return args, mode


if __name__ == '__main__':
	args, mode = parse_args() # get argument from the command line

	# load the data
	train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
																	   train_labels_path=args.train_labels_path,
																	   test_set_path=args.test_set_path,
																	   test_labels_path=args.test_labels_path)
	if mode == 'feature_sel':
		selected_features = feature_selection(train_set, test_set, train_labels)
		print_features(selected_features)
	elif mode == 'knn':
		predictions = knn(train_set, train_labels, test_set, test_labels, args.k)
		print_predictions(predictions)
	elif mode == 'alt':
		predictions = alternative_classifier(train_set, train_labels, test_set)
		print_predictions([float(i) for i in predictions])
	elif mode == 'knn_3d':
		predictions = knn_three_features(train_set, train_labels, test_set, args.k)
		print_predictions(predictions)
           #     print(labels)
	elif mode == 'knn_pca':
		prediction, scipy_train = knn_pca(train_set, train_labels, test_set, args.k)
		print_predictions(prediction)
                #print(scipy_train)
                #return confusion_matrix

	elif mode == 'knn_confusion':

		result_labels = knn(train_set, train_labels, test_set,test_labels, args.k)
		confusion_matrix = calculate_confusion_matrix(test_labels, result_labels)
		print(confusion_matrix)
		plot_matrix(confusion_matrix)
	elif mode == 'knn_3d_confusion':
		#result_labels = knn_three_features(train_set, train_labels, test_set, args.k)
		confusion_matrix = calculate_confusion_matrix(test_labels, result_labels)
		print(confusion_matrix)
		plot_matrix(confusion_matrix)

	else:
		raise Exception('Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
