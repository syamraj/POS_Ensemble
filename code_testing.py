# import opennlp
#
# pos = opennlp.OpenNLP("/home/devil/Thesis/apache-opennlp-1.8.0", "POSTagger", "en-pos-maxent.bin")
#
#
# list = []
#
#
# with open('/home/devil/Thesis/testdata.txt','rU') as fp:
#     for line in fp:
#         line2 = pos.parse(line[:-1])
#         list.append(line2)
#
# print list

# line = "hi_xx this_uu is_gb the_tt devil_ff speaking_ll"
# for words in line.split(' '):
#     # new_words = words.split(words.index('_'))
#     print words.split('_')[0]

# import matplotlib.pyplot as plt
# plt([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
# plt([1,2,3], [1,4,9], 'rs',  label='line 2')
# plt.axis([0, 4, 0, 10])
# plt.legend()
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Simple data to display in various forms
# x = np.linspace(0, 2 * np.pi, 400)
# y = np.sin(x ** 2)
# f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
#
# ax1.plot(y)
# ax1.set_title('Sharing both axes')
# ax2.plot(y)
# f.subplots_adjust(hspace=0)
# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
# plt.show()


# training_data_size = [100, 200, 300, 400, 500]
#
# for i in range(len(training_data_size)):
#     print i, training_data_size[i], type(training_data_size[i])
#
# def initialize():
#     global i
#     i= 0
#     i = i + 1
#     print i
#
#
# def somefun():
#     global i
#     i= i+1
#     print i
#
#
# initialize()
# somefun()
# initialize()
# somefun()


import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
# y_true = [1, 0, 1, 2]
# y = label_binarize(y_true, classes=[0, 1, 2])
# print y
# y_scores = np.array([0.1, 0.4, 0.35, 0.8])
# precision, recall, thresholds = precision_recall_curve(y[:, 1], y_scores)
# print precision
# print recall


# something = "Hi this is devil speaking. "
# line = something[:-1]
# # for words in something.split(' '):
# print len(something.split(' '))
# print something.split(' ')[:-1]


# class test1:
#     def func1(self):
#         print 'test1_func1'
#     def func2(self):
#         print 'test2_func2'
#
# class test2:
#     def func9(self):
#         print 'test2_func9'
#     def fun8(self):
#         print 'test2_func8'
#
# # test1.func1()
# t = test2()
# t.func9()

# class Employee:
#     'Common base class for all employees'
#     empCount = 0
#
#     # def __init__(self, name, salary):
#     #     self.name = name
#     #     self.salary = salary
#     #     Employee.empCount += 1
#
#     def displayCount(self):
#         print "Total Employee %d" % Employee.empCount
#
#     def displayEmployee(self):
#         print "Name : "
#
#
#
# emp1 = Employee()
# # emp2 = Employee("Manni", 5000)
# emp1.displayEmployee()
# # emp2.displayEmployee()


# file = open('/home/devil/Thesis/train_meta.txt', 'wt')
# line = 'original_feature' + ' ' + 'predicted_value'
# file.write(line+'\n')
# line2 = 'zxcvbnm'+ ' ' + 'NN'
# file.write(line2+'\n')
# # for processedwords in testdata_line_chunks:
# #     line = line + processedwords + ' '
# #     print '@@@@@@@@@@', line
# #     file.write(line+'\n')
# file.close()


# list1 = np.array([])
# list2 = np.array(['a', 'b'])
# print type(list1)
# list1 = np.append(list1, list2)
# print list1

from scipy.sparse import csr_matrix


# X_train = [['qwe', 'ert', 'sdf', 'fff'], ['vb', 'gg', 'rt', 'sd']]
# X_train = [['as', 'ddf', 'rrt', 'vvg'], ['tyu', 'hjk', 'qwerty', 'fghj']]
#
# indptr = [0]
# indices = []
# data = []
# vocabulary = {}
# for d in X_train:
#     for term in d:
#        index = vocabulary.setdefault(term, len(vocabulary))
#        indices.append(index)
#        data.append(1)
#        indptr.append(len(indices))
# X_train_new = csr_matrix((data, indices, indptr), dtype=int).toarray()
#
# # X_train_1 = ['qwe', 'ert', 'sdf', 'fff']
# # X_train_2 = ['vb', 'gg', 'rt', 'sd']
# y = ['nn', 'gg']
# X_test = ['as', 'ddf', 'rrt', 'vvg']
#
# indptr = [0]
# indices = []
# data = []
# vocabulary = {}
# for d in X_test:
#     for term in d:
#        index = vocabulary.setdefault(term, len(vocabulary))
#        indices.append(index)
#        data.append(1)
#        indptr.append(len(indices))
# X_test_new = csr_matrix((data, indices, indptr), dtype=int).toarray()
#
#
# clf_S = OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
# # count_vect = CountVectorizer()
# # tfidf_transformer = TfidfTransformer()
#
# # X_train_counts_s = count_vect.fit_transform(X_train[0])
# # X_tfidf = tfidf_transformer.fit_transform(X_train_counts_s)
# # X_train_counts_s_1 = count_vect.fit_transform(X_train[1])
# # X_tfidf_1 = tfidf_transformer.fit_transform(X_train_counts_s_1)
# # X_train_new = np.array([X_tfidf, X_tfidf_1])
# clf_S.fit(X_train_new, y)
# # X_train_counts_stack_test_meta_s = count_vect.transform(X_test)
# # X_tfidf_stack_test_meta = tfidf_transformer.transform(X_train_counts_stack_test_meta_s)
# temp = clf_S.predict(X_test_new)
#
#
# # from scipy.sparse import csr_matrix
# # docs = [["hello", "world", "hello"], ["goodbye", "cruel", "world"]]
# # indptr = [0]
# # indices = []
# # data = []
# # vocabulary = {}
# # for d in docs:
# #     for term in d:
# #        index = vocabulary.setdefault(term, len(vocabulary))
# #        indices.append(index)
# #        data.append(1)
# #        indptr.append(len(indices))
# # print csr_matrix((data, indices, indptr), dtype=int)

temp = "Ex"
print temp.upper()