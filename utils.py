import numpy as np 
import pandas as pd 
from scipy.sparse import coo_matrix
import scipy.sparse
class Member:
	def __init__(self, r_d, label=None, doc_id=None):
		self._r_d = r_d
		self.label = label
		self.doc_id = doc_id

class Cluster:
	def __init__(self):
		self._centroid = None
		self._members = []

	def reset_members(self):
		self._members = []

	def add_member(self, member):
		self._members.append(member)


def txt_to_matrix_sparse(data_path):
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