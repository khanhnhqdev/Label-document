# SVM: data loaded from sparse matrix saved in npz file
# Kmean: data loaded from text file saved from session 1


import numpy as np 
import pandas as pd 
from scipy.sparse import coo_matrix
from utils import *
from SVM import *
from Kmeans import *

############################# SVM ############################################ 
# Load data

X_train, Y_train = scipy.sparse.load_npz('./data_set/X_train.npz'), np.load('./data_set/Y_train.npy')
X_test, Y_test = scipy.sparse.load_npz('./data_set/X_test.npz'), np.load('./data_set/Y_test.npy')

print(X_train.toarray().shape)
print(Y_train.shape)
print(X_test.toarray().shape)
print(Y_test.shape)


# SVM model to classification
clustering_with_linear_SVM_sklearn(X_train, X_test, Y_train, Y_test)


############################# Kmean ######################################
with open('./data_set/words_idfs.txt') as f:
	vocab_size = len(f.read().splitlines())

num_cluster = 20

Kmean = Kmeans(num_clusters =  num_cluster, num_word_vocab = vocab_size) 
print(Kmean._num_clusters)
print(Kmean._num_word_vocab)

# Load data
Kmean.load_data('./data_set/train_tf_idf.txt')

max_purity = -1
max_NMI = -1
choose_seed = 0

# Run and choose the best seed

for i in range(10):
	Kmean.run(seed_value = i + 1, criterion = 'centroid', threshold = 0)
	print(Kmean.compute_purity())
	print(Kmean.compute_NMI())
	if(Kmean.compute_purity() > max_purity):
		max_purity = Kmean.compute_purity()
		max_NMI = Kmean.compute_NMI()
		choose_seed = i
print()
print(' Best compute_purity is ' + str(max_purity) + 'with seed ' + str(choose_seed))
print(' When seed is ' + str(choose_seed) +' NMI = ' + str(max_NMI))
