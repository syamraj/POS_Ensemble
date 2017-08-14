from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from itertools import cycle

import nltk
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

import opennlp

pos = opennlp.OpenNLP("/home/devil/Thesis/apache-opennlp-1.8.0", "POSTagger", "en-pos-maxent.bin")

f1_overall_before_correction_list = []
f1_overall_after_correction_list = []
map_list = []
predicted_class = []
predicted_class_listfeature = []
class_list = ['LS', ':', 'NN', 'CC', 'IN', 'VBZ', 'JJ', 'DT', 'VBG', 'VBN', '(', ')', 'NNS', 'PRP', 'VBP', 'TO', 'WDT',
              '.', 'VBD', 'VB', ',', 'PRP$', 'RB', 'MD', 'CD', 'NNP', 'FW',
              'JJS', 'RBS', 'WRB', 'RP', 'EX', 'JJR', 'RBR', 'POS', 'WP$', 'WP', 'PDT',
              'SYM', '``', "''", '-', 'NNS|FW', 'JJ|NN', 'E2A', '',
              'VBN|JJ', 'JJ|NNS', 'JJ|VBN', 'VBG|JJ', 'JJ|VBG', 'IN|PRP$', 'NN|NNS', 'N', 'CT', 'XT',
              'NNPS', 'NN|CD', 'NN|DT', 'VBD|VBN', 'JJ|RB', 'PP']


def initialize():
    global accracy_list_before_correction
    global accuracy_list_after_correction

    global overall_tpos_before_correction
    global overall_fpos_before_correction
    global overall_fn_before_correction

    global overall_tpos_after_correction
    global overall_fpos_after_correction
    global overall_fn_after_correction

    global f1_overall_before_correction
    global f1_overall_after_correction

    global predicted_class
    global predicted_class_listfeature
    global predicted_class_unique

    accracy_list_before_correction = []
    accuracy_list_after_correction = []

    overall_tpos_before_correction = 0
    overall_fpos_before_correction = 0
    overall_fn_before_correction = 0

    overall_tpos_after_correction = 0
    overall_fpos_after_correction = 0
    overall_fn_after_correction = 0

    f1_overall_before_correction = 0
    f1_overall_after_correction = 0

    predicted_class = []
    predicted_class_listfeature = []
    predicted_class_unique = []


def accuracy_pos(testdata_completelist, testdata_gold_completelist):
    for i in range(len(testdata_complete)):
        global overall_tpos_before_correction
        global overall_fpos_before_correction
        global overall_fn_before_correction

        tp = 0
        fpos = 0
        fn = 0
        testdata_line_chunks = testdata_completelist[i].split(' ')
        testdata_gold_line_chunks = testdata_gold_completelist[i].split(' ')
        print len(testdata_line_chunks)
        print len(testdata_gold_line_chunks)
        if len(testdata_line_chunks) != len(testdata_gold_line_chunks):
            print "line number ", i, "in the file not in sync"
        else:
            for j in range(len(testdata_line_chunks)):
                if testdata_line_chunks[j].split('_')[0] == testdata_gold_line_chunks[j].split('/')[0]:
                    print testdata_line_chunks[j].split('_')[0] + '............' + \
                          testdata_gold_line_chunks[j].split('/')[0]
                    print testdata_line_chunks[j].split('_')[1]
                    print testdata_gold_line_chunks[j].split('/')[1]
                    if testdata_line_chunks[j].split('_')[1] == testdata_gold_line_chunks[j].split('/')[1]:
                        tp = tp + 1
                        overall_tpos_before_correction = overall_tpos_before_correction + 1
                        print "tp incremented"
                    else:
                        fpos = fpos + 1
                        overall_fpos_before_correction = overall_fpos_before_correction + 1
                        fn = fn + 1
                        overall_fn_before_correction = overall_fn_before_correction + 1
            print "tp = ", tp
            print "fn = ", fn
            accuracy = float(tp * 2) / float(2 * tp + fn + fpos)
            accracy_list_before_correction.append(accuracy)
    return accuracy


def accuracy_pos_withcorrection(testdata_completelist, testdata_gold_completelist, map_list):
    # file = open('/home/devil/Thesis/Processed_testdata.txt', 'wt')
    for i in range(len(testdata_complete)):
        tp = 0
        fpos = 0
        fn = 0
        global overall_tpos_after_correction
        global overall_fpos_after_correction
        global overall_fn_after_correction
        global predicted_class_listfeature
        global predicted_class
        global predicted_class_unique

        testdata_line_chunks = testdata_completelist[i].split(' ')
        testdata_gold_line_chunks = testdata_gold_completelist[i].split(' ')
        print len(testdata_line_chunks)
        print len(testdata_gold_line_chunks)
        if len(testdata_line_chunks) != len(testdata_gold_line_chunks):
            print "line number ", i, "in the file not in sync"
        else:
            for j in range(len(testdata_line_chunks)):
                if testdata_line_chunks[j].split('_')[0] == testdata_gold_line_chunks[j].split('/')[0]:
                    print testdata_line_chunks[j].split('_')[0] + '............' + \
                          testdata_gold_line_chunks[j].split('/')[0]
                    print testdata_line_chunks[j].split('_')[1]
                    print "gold", testdata_gold_line_chunks[j].split('/')[1]
                    if testdata_line_chunks[j].split('_')[1] == testdata_gold_line_chunks[j].split('/')[1]:
                        tp = tp + 1
                        overall_tpos_after_correction = overall_tpos_after_correction + 1
                        print "tp incremented"
                    else:
                        text_for_correction = testdata_line_chunks[j].split('_')[0][-3:] + \
                                              testdata_line_chunks[j - 1].split('_')[0] + \
                                              testdata_line_chunks[j - 1].split('_')[1]
                        text_for_correction_list = [text_for_correction]
                        X_train_counts1 = count_vect.transform(text_for_correction_list)
                        X_tfidf1 = tfidf_transformer.transform(X_train_counts1)
                        new_tag = clf.predict(X_tfidf1)
                        predicted_class_listfeature.append(text_for_correction)
                        pos1 = 0
                        count = 0
                        flag = 0
                        for iter in new_tag[0]:
                            if iter == 1:
                                pos1 = count
                                predicted_class.append(class_list[pos1])
                                flag = 1
                            else:
                                count = count + 1
                                if count == len(new_tag[0]) and flag == 0:
                                    predicted_class.append(class_list[pos1])
                        if not predicted_class_unique.__contains__(class_list[pos1]):
                            predicted_class_unique.append(class_list[pos1])
                        # new_wrd_tag_pair = testdata_line_chunks[j].split('_')[0] + '_' + new_tag[0]
                        new_wrd_tag_pair = testdata_line_chunks[j].split('_')[0] + '_' + class_list[pos1]
                        testdata_completelist[i].replace(testdata_line_chunks[j], new_wrd_tag_pair)
                        testdata_line_chunks[j] = new_wrd_tag_pair
                        for line in map_list:
                            if testdata_line_chunks[j].split('_')[0] == line.split('_')[0]:
                                testdata_line_chunks[j] = line[:-1]
                        print "############", testdata_line_chunks[j]
                        if testdata_line_chunks[j].split('_')[1] != testdata_gold_line_chunks[j].split('/')[1]:
                            fpos = fpos + 1
                            overall_fpos_after_correction = overall_fpos_after_correction + 1
                            fn = fn + 1
                            overall_fn_after_correction = overall_fn_after_correction + 1
                        if testdata_line_chunks[j].split('_')[1] == testdata_gold_line_chunks[j].split('/')[1]:
                            tp = tp + 1
                            overall_tpos_after_correction = overall_tpos_after_correction
                            print "tp incremented"
            print "tp = ", tp
            print "fn = ", fn
            accuracy = float(tp) * 2 / float(2 * tp + fn + fpos)
            accuracy_list_after_correction.append(accuracy)
            print '@@@@@@@@@@@@@@@@@@@@@@@@@@@', testdata_line_chunks
    # line = ''
    #         for processedwords in testdata_line_chunks:
    #             line = line + processedwords + ' '
    #         print '@@@@@@@@@@', line
    #         file.write(line+'\n')
    # file.close()
    return accuracy


def pos_features(sentence, i, history):
    features = ''
    features += sentence[i][-3:]

    if i == 0:
        features += "<START>"
    else:
        features += sentence[i - 1]
        features += history[i - 1]
    return features


def processing(train_sents):
    for tagged_sent in train_sents:
        untagged_sent = nltk.tag.untag(tagged_sent)
        for i, (word, tag) in enumerate(tagged_sent):
            featureset = pos_features(untagged_sent, i, history)
            train_set_featureset.append(featureset)
            train_set_tags.append(tag)
            # train_set.append((featureset, tag))
            history.append(tag)


training_data_size = [100, 500, 1000, 2500, 5000, 10000, 12000, 14000, 20000, 25000]

# for training_data_size_iterator in range(len(training_data_size)):
for training_data_size_iterator in range(1):
    initialize()
    Tag_list = []
    Per_tag_acc_list = []
    testdata_complete = []
    testdata_gold_complete = []

    train_set_featureset = []
    train_set_tags = []
    history = []

    train_sents = []
    with open('output.txt', 'rU') as fp:
        for line in fp:
            str = line.split(' ')
            listEachLine = []
            for i in str[:-1]:
                if i.__contains__('_'):
                    listEachLine.append((i.split('_')[0], i.split('_')[1]))
            train_sents.append(listEachLine)

    processing(train_sents)

    train_set_featureset = train_set_featureset[:training_data_size[training_data_size_iterator]]
    train_set_tags = train_set_tags[:training_data_size[training_data_size_iterator]]

    n_split = int(len(train_set_featureset) * .7)

    X_train = train_set_featureset[:n_split]
    y_train = train_set_tags[:n_split]

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    X_train_counts = count_vect.fit_transform(X_train)
    X_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # X_train_counts1 = count_vect.transform(X_test)
    # X_tfidf1 = tfidf_transformer.transform(X_train_counts1)
    y_train = label_binarize(y_train, classes=class_list)
    clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
    # clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_tfidf, y_train)

    # print(clf.score(X_tfidf1.toarray(), y_test))

    # with open('/home/devil/Thesis/testdata.txt','rU') as fp:
    with open('testdata.txt', 'rU') as fp:
        for line in fp:
            line2 = pos.parse(line[:-1])
            testdata_complete.append(line2.rstrip())

    with open('/home/devil/Thesis/map.txt', 'rU') as fp:
        for line in fp:
            map_list.append(line)

        # with open('/home/devil/Thesis/testdata_gold.txt', 'rU') as fp:
    with open('testdata_gold.txt', 'rU') as fp:
        for line in fp:
            testdata_gold_complete.append(line[:-1].rstrip())
    print testdata_complete
    print testdata_gold_complete
    print 'accuracy before correction', accuracy_pos(testdata_complete, testdata_gold_complete)
    print "..........................................................................................................."
    print 'accuracy after correction', accuracy_pos_withcorrection(testdata_complete, testdata_gold_complete, map_list)

    f1_overall_before_correction = float(overall_tpos_before_correction * 2) / \
                                   float(2 * overall_tpos_before_correction + overall_fn_before_correction
                                         + overall_fpos_before_correction)
    f1_overall_before_correction_list.append(f1_overall_before_correction)

    f1_overall_after_correction = float(overall_tpos_after_correction * 2) / float(
        2 * overall_tpos_after_correction
        + overall_fn_after_correction
        + overall_fpos_after_correction)
    f1_overall_after_correction_list.append(f1_overall_after_correction)
    print 'predicted class list features length', len(predicted_class_listfeature)
    print 'predicted class length', len(predicted_class)

    X_train_counts2 = count_vect.transform(predicted_class_listfeature)
    X_tfidf2 = tfidf_transformer.transform(X_train_counts2)

    print "printing the result", accracy_list_before_correction
    print "printing the result after correction", accuracy_list_after_correction

    print "length of accuracy list before correction", len(accracy_list_before_correction)
    print "length of accuracy list after correction", len(accuracy_list_after_correction)

    print "overall F1 score after correction", f1_overall_after_correction
    print "overall F1 score before correction", f1_overall_before_correction

    precision = dict()
    recall = dict()
    average_precision = dict()
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    lw = 2

    y_predicted_list = label_binarize(predicted_class, classes=class_list)
    y_score = clf.predict_proba(X_tfidf2)
    print 'length of y_predicted_list binarized', len(y_predicted_list)
    print 'length of y_score', len(y_score)
    print '1st y_predicted', y_predicted_list[1]
    print '1st y_score', y_score[0]


    for i in range(len(class_list)):
        precision[i], recall[i], _ = precision_recall_curve(y_predicted_list[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_predicted_list[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_predicted_list.ravel(),
                                                                    y_score.ravel())
    average_precision["micro"] = average_precision_score(y_predicted_list, y_score,
                                                         average="micro")
    plt.clf()
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    # for i, color in zip(range(len(class_list)), colors):
    #     plt.plot(recall[i], precision[i], color=color, lw=lw,
    #              label='Precision-recall curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()
# plot of the accuracy per sentances

# f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
#
# ax1.plot(accracy_list_before_correction, 'ro')
# ax1.set_title('accuracy before correction')
# ax2.plot(accuracy_list_after_correction, 'go')
# ax2.set_title('accuracy after correction')
# f.subplots_adjust(hspace=0)
# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
# plt.show()

# plot of the accuracies with increase in the training data
print 'length of training data size', len(training_data_size)
print 'length of f1_overall_after_correction', len(f1_overall_after_correction_list)
print 'printing predicted class features list', predicted_class_listfeature

# plot for task2

# x = np.array(training_data_size)
# y = np.array(f1_overall_after_correction_list)
# f = interp1d(x, y)
# f2 = interp1d(x, y, kind='cubic')
# xnew = np.linspace(training_data_size[0], training_data_size[len(training_data_size)-1], num=41, endpoint=True)
# plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
# plt.legend(['data', 'linear', 'cubic'], loc='best')
# plt.show()
