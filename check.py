from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt

import nltk
import opennlp

pos = opennlp.OpenNLP("/home/devil/Thesis/apache-opennlp-1.8.0", "POSTagger", "en-pos-maxent.bin")

accracy_list_before_correction = []
accuracy_list_after_correction = []
map_list = []

overall_tpos_before_correction = 0
overall_fpos_before_correction = 0
overall_fn_before_correction = 0

overall_tpos_after_correction = 0
overall_fpos_after_correction = 0
overall_fn_after_correction = 0

f1_overall_before_correction = 0
f1_overall_after_correction = 0

def accuracy_pos(testdata_completelist, testdata_gold_completelist):
    for i in range(len(testdata_complete)):
        tp = 0
        fpos = 0
        fn = 0
        global overall_tpos_before_correction
        global overall_fpos_before_correction
        global overall_fn_before_correction
        global f1_overall_before_correction
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
            accuracy = float(tp*2) / float(2*tp + fn + fpos)
            f1_overall_before_correction = float(overall_tpos_before_correction * 2) / \
                                           float(2 * overall_tpos_before_correction + overall_fn_before_correction
                                                 + overall_fpos_before_correction)
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
        global f1_overall_after_correction
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
                        text_for_correction = testdata_line_chunks[j].split('_')[1][-3:] + \
                                              testdata_line_chunks[j - 1].split('_')[0] + \
                                              testdata_line_chunks[j - 1].split('_')[1]
                        text_for_correction_list = [text_for_correction]
                        X_train_counts1 = count_vect.transform(text_for_correction_list)
                        X_tfidf1 = tfidf_transformer.transform(X_train_counts1)
                        new_tag = clf.predict(X_tfidf1)
                        new_wrd_tag_pair = testdata_line_chunks[j].split('_')[0] + '_' + new_tag[0]
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
            accuracy = float(tp)*2 / float(2*tp + fn + fpos)
            accuracy_list_after_correction.append(accuracy)
            f1_overall_after_correction = float(overall_tpos_after_correction * 2) / float(
                2 * overall_tpos_after_correction
                + overall_fn_after_correction
                + overall_fpos_after_correction)
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

train_set_featureset = train_set_featureset[:500]
train_set_tags = train_set_tags[:500]

n_split = int(len(train_set_featureset) * .7)

X_train = train_set_featureset[:n_split]
y_train = train_set_tags[:n_split]

print X_train[0]

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(X_train)
X_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# X_train_counts1 = count_vect.transform(X_test)
# X_tfidf1 = tfidf_transformer.transform(X_train_counts1)

clf = RandomForestClassifier(n_estimators=10)
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
print accuracy_pos(testdata_complete, testdata_gold_complete)
print "..........................................................................................................."
print accuracy_pos_withcorrection(testdata_complete, testdata_gold_complete, map_list)

print "printing the result", accracy_list_before_correction
print "printing the result after correction", accuracy_list_after_correction

print "length of accuracy list before correction", len(accracy_list_before_correction)
print "length of accuracy list after correction", len(accuracy_list_after_correction)

print "overall F1 score after correction", f1_overall_after_correction
print "overall F1 score before correction", f1_overall_before_correction

# plt.plot(accracy_list_before_correction , 'ro')
# plt.ylabel('accuracy before applying the correction model')
# plt.show()
#
#
# plt.plot(accuracy_list_after_correction , 'go')
# plt.ylabel('accuracy after applying the correction model')
# plt.show()

f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

ax1.plot(accracy_list_before_correction, 'ro')
ax1.set_title('accuracy before correction')
ax2.plot(accuracy_list_after_correction, 'go')
ax2.set_title('accuracy after correction')
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.show()