from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

import opennlp
import nltk

# pos = opennlp.OpenNLP("/home/devil/Thesis/apache-opennlp-1.8.0", "POSTagger", "en-pos-maxent.bin")
pos = opennlp.OpenNLP("/home/devil/Thesis/apache-opennlp-1.8.0", "POSTagger", "en-pos-maxent-brown.bin")


f1_overall_before_correction_list = []
f1_overall_after_correction_list = []
map_list = []
predicted_class = []
predicted_class_list = []


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
    global predicted_class_list



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
    predicted_class_list = []


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
                if testdata_line_chunks[j].split('_')[0].upper() == \
                        testdata_gold_line_chunks[j].split('/')[0].upper():
                    print testdata_line_chunks[j].split('_')[0] + '............' + \
                          testdata_gold_line_chunks[j].split('/')[0]
                    print testdata_line_chunks[j].split('_')[1]
                    print testdata_gold_line_chunks[j].split('/')[1]
                    if testdata_line_chunks[j].split('_')[1].upper() == \
                            testdata_gold_line_chunks[j].split('/')[1].upper():
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
            accuracy = float(tp*2) / float(2*tp + fn + fpos)
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
        testdata_line_chunks = testdata_completelist[i].split(' ')
        testdata_gold_line_chunks = testdata_gold_completelist[i].split(' ')
        print len(testdata_line_chunks)
        print len(testdata_gold_line_chunks)
        if len(testdata_line_chunks) != len(testdata_gold_line_chunks):
            print "line number ", i, "in the file not in sync"
        else:
            for j in range(len(testdata_line_chunks)):
                if testdata_line_chunks[j].split('_')[0].upper() == \
                        testdata_gold_line_chunks[j].split('/')[0].upper():
                    print testdata_line_chunks[j].split('_')[0] + '............' + \
                          testdata_gold_line_chunks[j].split('/')[0]
                    print testdata_line_chunks[j].split('_')[1]
                    print "gold", testdata_gold_line_chunks[j].split('/')[1]
                    if testdata_line_chunks[j].split('_')[1].upper() == \
                            testdata_gold_line_chunks[j].split('/')[1].upper():
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
                        new_tag = clf1.predict(X_tfidf1)
                        predicted_class_list.append(new_tag)
                        if not predicted_class.__contains__(new_tag):
                            predicted_class.append(new_tag)
                        new_wrd_tag_pair = testdata_line_chunks[j].split('_')[0] + '_' + new_tag[0]
                        testdata_completelist[i].replace(testdata_line_chunks[j], new_wrd_tag_pair)
                        testdata_line_chunks[j] = new_wrd_tag_pair
                        for line in map_list:
                            if testdata_line_chunks[j].split('_')[0].upper() == \
                                    line.split('_')[0].upper():
                                testdata_line_chunks[j] = line[:-1]
                        print "############", testdata_line_chunks[j]
                        if testdata_line_chunks[j].split('_')[1].upper() != \
                                testdata_gold_line_chunks[j].split('/')[1].upper():
                            fpos = fpos + 1
                            overall_fpos_after_correction = overall_fpos_after_correction + 1
                            fn = fn + 1
                            overall_fn_after_correction = overall_fn_after_correction + 1
                        if testdata_line_chunks[j].split('_')[1].upper() == \
                                testdata_gold_line_chunks[j].split('/')[1].upper():
                            tp = tp + 1
                            overall_tpos_after_correction = overall_tpos_after_correction
                            print "tp incremented"
            print "tp = ", tp
            print "fn = ", fn
            accuracy = float(tp)*2 / float(2*tp + fn + fpos)
            accuracy_list_after_correction.append(accuracy)
            print '@@@@@@@@@@@@@@@@@@@@@@@@@@@', testdata_line_chunks
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

# training_data_size = [500, 4000, 8000, 10000, 15000, 20000]
training_data_size = [90000]

# for training_data_size_iterator in range(len(training_data_size)):
for training_data_size_iterator in range(len(training_data_size)):
    initialize()
    Tag_list = []
    Per_tag_acc_list = []
    testdata_complete = []
    testdata_gold_complete = []

    train_set_featureset = []
    train_set_tags = []
    history = []

    train_sents = []

    file_list = ['stack_testdata_Brown_gold', 'output.txt']
    Genia_files = ['testdata.txt', 'testdata_gold.txt']
    Brown_files = ['testdata_brown_based_on_stack_testdata.txt',
                   'testdata_brown_based_on_stack_testdata_gold.txt']
    with open(file_list[0], 'rU') as fp:
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

    n_split = int(len(train_set_featureset) * .8)

    Classifier1X_train = train_set_featureset[:n_split]
    Classifier1Y_train = train_set_tags[:n_split]

    n_split1 = int(len(Classifier1X_train) * .2)

    Data_for_learning_from_baseX_train = Classifier1X_train[:n_split1]
    Data_for_learning_from_baseY_train = Classifier1Y_train[:n_split1]

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    Classifier1X_train_counts = count_vect.fit_transform(Classifier1X_train)
    Classifier1X_tfidf = tfidf_transformer.fit_transform(Classifier1X_train_counts)

    clf1 = RandomForestClassifier(n_estimators=10)
    clf1.fit(Classifier1X_tfidf, Classifier1Y_train)

    clf2 = RandomForestClassifier(n_estimators=10)
    clf2.fit(Classifier1X_tfidf, Classifier1Y_train)

# print(clf.score(X_tfidf1.toarray(), y_test))

# with open('/home/devil/Thesis/testdata.txt','rU') as fp:
    #with open(Genia_files[0], 'rU') as fp:
    with open(Brown_files[0], 'rU') as fp:
        for line in fp:
            line2 = pos.parse(line[:-1])
            testdata_complete.append(line2.rstrip())

    with open('/home/devil/Thesis/map.txt', 'rU') as fp:
        for line in fp:
            map_list.append(line)

# with open('/home/devil/Thesis/testdata_gold.txt', 'rU') as fp:
    #with open(Genia_files[1], 'rU') as fp:
    with open(Brown_files[1], 'rU') as fp:
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

    print "printing the result", accracy_list_before_correction
    print "printing the result after correction", accuracy_list_after_correction

    print "length of accuracy list before correction", len(accracy_list_before_correction)
    print "length of accuracy list after correction", len(accuracy_list_after_correction)

    print "overall F1 score after correction", f1_overall_after_correction
    print "overall F1 score before correction", f1_overall_before_correction


# plot of the accuracies with increase in the training data
print 'length of training data size', len(training_data_size)
print 'length of f1_overall_after_correction', len(f1_overall_after_correction_list)
print 'f1_overall_after_correction', f1_overall_after_correction_list
x = np.array(training_data_size)
y = np.array(f1_overall_after_correction_list)
f_b = interp1d(x, y)
f1_b = interp1d(x, y)
f2_b = interp1d(x, y, kind='cubic')
x1 = np.array(training_data_size)
y1 = np.array(f1_overall_before_correction_list)
f = interp1d(x1, y1)
f1 = interp1d(x1, y1)
f2 = interp1d(x1, y1, kind='cubic')
xnew = np.linspace(training_data_size[0], training_data_size[len(training_data_size)-1], num=41, endpoint=True)
plt.plot(x, y, 'o', xnew, f_b(xnew), '-', xnew, f2_b(xnew), '--')
plt.plot(x1, y1, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()