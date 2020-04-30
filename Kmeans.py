import numpy as np 
import pandas as pd 
from collections import defaultdict
from utils import *
class Kmeans:
	def __init__(self, num_clusters, num_word_vocab):
		self._num_clusters = num_clusters
		self._num_word_vocab = num_word_vocab
		self._clusters = [Cluster() for _ in range(self._num_clusters)]
		self._E = []
		self._S = 0 #similarity
		self._data = []
		self._label_count = defaultdict(int)
	def load_data(self, data_path):
		def sparse_to_dense(sparse_r_d, vocab_size):
			r_d = [0.0 for _ in range(vocab_size)]
			indices_tfidf = sparse_r_d.split()
			for index_tfidf in indices_tfidf:
				index = int(index_tfidf.split(':')[0])
				tfidf = float(index_tfidf.split(':')[1])
				r_d[index] = tfidf
			return np.array(r_d)

		with open(data_path) as f:
			d_lines = f.read().splitlines()
		with open('./data_set/words_idfs.txt') as f:
			vocab_size = len(f.read().splitlines())

		self._data = []
		self._label_count = defaultdict(int)

		for data_id, d in enumerate(d_lines):
			features = d.split('<fff>')
			label, doc_id = int(features[0]), int(features[1])
			self._label_count[label] += 1 #self._label_count chua phan bo so luong cac label
			r_d = sparse_to_dense(sparse_r_d = features[2], vocab_size = vocab_size)
			self._data.append(Member(r_d = r_d, label = label, doc_id = doc_id)) #self._data chua tat ca cac member(gom doc_id, label va vector tuong ung)
		## check sum ######################
		# print(len(self._data))
		# ch_sum = 0
		# for i in self._label_count:
		#  	print(self._label_count[i])
		#	ch_sum += self._label_count[i]
		# print(ch_sum)
		##################################
	def random_init(self, seed_value):
		cnt = 0
		for cluster in self._clusters:
			cnt += 1
			np.random.seed(seed_value + cnt)
			tmp = np.random.randint(len(self._data))
			cluster._centroid = self._data[tmp]._r_d
			# print(cluster._centroid)
			# print(cluster._centroid.shape)

	def compute_similarity(self, member, centroid):
		# khoang cach cang nho thi similarity cang lon
		distance = np.sum((np.array(member._r_d) - np.array(centroid)) ** 2)
		return 1. / (distance + 1e-12)

	def select_cluster_for(self, member):
		best_fit_cluster = None
		max_similarity = -1
		for cluster in self._clusters:
			similarity = self.compute_similarity(member, cluster._centroid)
			if similarity > max_similarity:
				best_fit_cluster = cluster
				max_similarity = similarity

		best_fit_cluster.add_member(member)
		# print(best_fit_cluster._centroid)
		return max_similarity

	def update_centroid_of(self, cluster):
		member_list = [member._r_d for member in cluster._members] # chua cac vector cua cac member
		aver_r_d = np.mean(member_list ,axis = 0)
		tmp = np.sqrt(np.sum(aver_r_d ** 2))
		# new_centroid = np.array([value / tmp for value in aver_r_d])
		new_centroid = np.array(aver_r_d)
		cluster._centroid = new_centroid

	def stopping_condition(self, criterion, threshold):
		criteria = ['centroid', 'similarity', 'max_iter']
		assert criterion in criteria
		if criterion == 'max_iter':
			if self._iterration >= threshold:
				return True
			else:
				return False
		elif criterion == 'centroid':
			E_new = [list(cluster._centroid) for cluster in self._clusters]
			E_new_minus_E = [centroid for centroid in E_new if centroid not in self._E]
			self._E = E_new
			print(len(E_new_minus_E))
			if len(E_new_minus_E) <= threshold:
				return True
			else:
				return False
		else:
			new_S_minus_S = self._new_S - self._S
			self._S = self._new_S
			if new_S_minus_S <= threshold:
				return True
			else:
				return False

	def run(self, seed_value, criterion, threshold):
		self.random_init(seed_value)
		self._iterration = 0
		while True:
			for cluster in self._clusters:
				cluster.reset_members()
			self._new_S = 0
			
			for member in self._data:
				max_s = self.select_cluster_for(member)
				self._new_S += max_s
			for cluster in self._clusters:
				self.update_centroid_of(cluster)
			self._iterration += 1
			if self.stopping_condition(criterion, threshold):
				break
			print(str(self._iterration) + ' iteration')

	def compute_purity(self):
		majority_sum = 0
		for cluster in self._clusters:
			member_labels = [member.label for member in cluster._members]
			max_count = max([member_labels.count(label) for label in range(20)])
			majority_sum += max_count
		return majority_sum * 1. / len(self._data)


	def compute_NMI(self):
		I_value, H_C, H_omega, N = 0.0, 0.0, 0.0, len(self._data)
		for cluster in self._clusters:
			wk = len(cluster._members) * 1.
			H_omega += -wk / N * np.log10(wk / N)
			member_labels = [member.label for member in cluster._members]
			for label in range(self._num_clusters):
				wk_cj = member_labels.count(label) * 1.
				cj = self._label_count[label]
				I_value += wk_cj / N * np.log10(N * wk_cj / (wk * cj) + 1e-12)
		for label in range(self._num_clusters):
			cj = self._label_count[label] * 1.
			H_C += -cj / N * np.log10(cj / N)
		return I_value * 2. / (H_omega + H_C)