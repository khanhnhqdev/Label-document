import numpy as np 
import pandas as pd 
from scipy.sparse import coo_matrix
from utils import *
from SVM import *
from Kmeans import *

# Test select_cluster
member1 = Member(label = 1, doc_id = 1, r_d = [1,0])
member2 = Member(label = 1, doc_id = 1, r_d = [0,0])
member3 = Member(label = 1, doc_id = 1, r_d = [0,1])
member4 = Member(label = 1, doc_id = 1, r_d = [1,1])
Cluster = Cluster()
Cluster.add_member(member1)
Cluster.add_member(member2)
Cluster.add_member(member3)
Cluster.add_member(member4)

Kmean = Kmeans(num_clusters =  3, num_word_vocab = 2) 
Kmean.update_centroid_of(Cluster)
print(Cluster._centroid) #[0.5,0.5]

# Test random init ###
Kmean = Kmeans(num_clusters =  3, num_word_vocab = 2) 
Kmean.random_init(1)
for cluster in Kmean._clusters:
	print(cluster._centroid)

# Test select_cluster_for
Kmean = Kmeans(num_clusters =  3, num_word_vocab = 2) 
Cluster1 = Cluster()
Cluster1._centroid = [0,0]
Cluster2 = Cluster()
Cluster2._centroid = [2,0]
Kmean._clusters = [Cluster1, Cluster2]
member1 = Member(label = 1, doc_id = 1, r_d = [-1,0])
Kmean.select_cluster_for(member1) # member1 thuoc Cluster1
print(Cluster1._members)
print(Cluster2._members)


# Test run
member1 = Member(label = 1, doc_id = 1, r_d = [0,0])
member2 = Member(label = 1, doc_id = 1, r_d = [1,0])
member3 = Member(label = 1, doc_id = 1, r_d = [0,1])
member4 = Member(label = 0, doc_id = 1, r_d = [5,0])
member5 = Member(label = 0, doc_id = 1, r_d = [6,0])
member6 = Member(label = 0, doc_id = 1, r_d = [5,1])
Kmean = Kmeans(num_clusters =  2, num_word_vocab = 2) 
Kmean._data.append(member1)
Kmean._data.append(member2)
Kmean._data.append(member3)
Kmean._data.append(member4)
Kmean._data.append(member5)
Kmean._data.append(member6)
Kmean._label_count = {0 : 3, 1 : 3}
Kmean.run(seed_value = 1, criterion = 'centroid', threshold = 0)
print(Kmean.compute_purity())
print(Kmean.compute_NMI())
# 1




###################################################################################################3
Test SVM
Load data

X_train, Y_train = scipy.sparse.load_npz('./data_set/X_train.npz'), np.load('./data_set/Y_train.npy')
X_test, Y_test = scipy.sparse.load_npz('./data_set/X_test.npz'), np.load('./data_set/Y_test.npy')

print(X_train.toarray().shape)
print(Y_train.shape)
print(X_train.toarray())
print(Y_train)


# SVM model to classification
clustering_with_linear_SVM_sklearn(X_train, X_test, Y_train, Y_test)



