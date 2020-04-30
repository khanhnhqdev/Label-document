from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
def clustering_with_linear_SVM_sklearn(X_train, X_test, Y_train, Y_test):
	
	# SVM classification
	classifier = LinearSVC(
				 C = 5.0,
				 tol = 0.001,
				 verbose = False)

	classifier.fit(X_train, Y_train)
	Y_pred = classifier.predict(X_test)

	# Classification report and confusion matrix
	# labels = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 
	#           'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 
	#           'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 
	#           'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
	print(classification_report(Y_test, Y_pred))
	plt.rcParams["figure.figsize"] = (10, 20)
	plot_confusion_matrix(confusion_matrix(Y_test, Y_pred),
						  show_normed=True,
						  show_absolute=False)
						  # class_names=labels)
	plt.show()