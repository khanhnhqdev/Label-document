import numpy as np 
import pandas as pd 
from scipy.sparse import coo_matrix
import scipy.sparse


def txt_to_matrix_sparse(data_path):
	'''
	read data saved in txt file(in session 1) and convert to sparse matrix save in npz file(for SVM)
	'''
	with open('./data_set/words_idfs.txt') as f:
		vocab_size = len(f.read().splitlines())
	with open(data_path) as f:
		d_lines = f.read().splitlines()
	
	col = []
	row = []
	data = []
	num_row = len(d_lines)
	num_col = vocab_size
	y =[]
	print(num_row)
	for data_id, d in enumerate(d_lines):
		features = d.split('<fff>')
		label, doc_id = int(features[0]), int(features[1])      
		y.append(label)
		vector_text = features[2].split()
		for index_tfidf in vector_text:
			index = int(index_tfidf.split(':')[0])
			tfidf = float(index_tfidf.split(':')[1])
			row.append(data_id)
			col.append(index)
			data.append(tfidf)

	scipy.sparse.save_npz('./data_set/X_test.npz', coo_matrix((data, (row, col)), shape = (num_row, num_col)))
	np.save('./data_set/Y_test.npy', np.array(y))
	return coo_matrix((data, (row, col)), shape = (num_row, num_col)), np.array(y)

# txt_to_matrix_sparse('./data_set/test_tf_idf.txt')
# txt_to_matrix_sparse('./data_set/train_tf_idf.txt')
