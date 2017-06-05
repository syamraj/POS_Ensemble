from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import nltk

import opennlp

pos = opennlp.OpenNLP("/home/devil/Thesis/apache-opennlp-1.8.0", "POSTagger", "en-pos-maxent.bin")


def accuracy_pos(testdata_completelist, testdata_gold_completelist):
    for i in range(len(testdata_complete)):
        tp = 0
        fpos = 0
        fn = 0
        testdata_line_chunks = testdata_completelist[i].split(' ')
        testdata_gold_line_chunks = testdata_gold_completelist[i].split(' ')
        print len(testdata_line_chunks)
        print len(testdata_gold_line_chunks)
        if len(testdata_line_chunks) != len(testdata_gold_line_chunks):
            print "line number ", i , "in the file not in sync"
        else :
            for j in range(len(testdata_line_chunks)):
                if testdata_line_chunks[j].split('_')[0] == testdata_gold_line_chunks[j].split('/')[0]:
                    print testdata_line_chunks[j].split('_')[0] +'............'+testdata_gold_line_chunks[j].split('/')[0]
                    print testdata_line_chunks[j].split('_')[1]
                    print testdata_gold_line_chunks[j].split('/')[1]
                    if testdata_line_chunks[j].split('_')[1] == testdata_gold_line_chunks[j].split('/')[1]:
                        tp = tp+1
                        print "tp incremented"
                    else :
                        fpos = fpos + 1
                        fn = fn + 1
            print "tp = ", tp
            print "fn = ", fn
            accuracy = float(tp)/float(tp + fn)
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

def correction_model():
    print "to be added"

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
clf.fit(X_tfidf.toarray(), y_train)

# print(clf.score(X_tfidf1.toarray(), y_test))

with open('/home/devil/Thesis/testdata.txt','rU') as fp:
    for line in fp:
        line2 = pos.parse(line)
        testdata_complete.append(line2)
with open('/home/devil/Thesis/testdata_gold.txt', 'rU') as fp:
    for line in fp:
        testdata_gold_complete.append(line)
print accuracy_pos(testdata_complete, testdata_gold_complete)
