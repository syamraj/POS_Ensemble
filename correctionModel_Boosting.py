import re
import nltk
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.tree import DecisionTreeClassifier

__author__ = 'devil'


# instead of of changing everything to vec at once trying to change each features and store it in the list

# trainingFeatures = ['lloSTARTloSTARTo', 'hisHelloisUHs']
# trainingFeatures1 = ['hisHelloisUHs']

def pos_features(sentence, i, history):
    features = ''
    features += sentence[i][-3:]

    if i == 0:
        features += "<START>"
    else:
        features += sentence[i - 1]
        features += history[i - 1]
    return features


train_set_featureset = []
train_set_tags = []
history = []


def processing(train_sents):
    for tagged_sent in train_sents:
        untagged_sent = nltk.tag.untag(tagged_sent)
        for i, (word, tag) in enumerate(tagged_sent):
            featureset = pos_features(untagged_sent, i, history)
            train_set_featureset.append(featureset)
            train_set_tags.append(tag)
            # train_set.append((featureset, tag))
            history.append(tag)



train_sents = []
with open('output.txt', 'rU') as fp:
    for line in fp:
        str = line.split(' ')
        listEachLine = []
        for i in str[:-1]:
            if i.__contains__('_'):
                listEachLine.append((i.split('_')[0], i.split('_')[1]))
        train_sents.append(listEachLine)

print train_sents[:2]

processing(train_sents)

# print train_set_featureset
# print train_set_tags

train_set_featureset = train_set_featureset[:500]
train_set_tags = train_set_tags[:500]

n_split = int(len(train_set_featureset) * .7)
print 'inside'
print n_split

X_train, X_test = train_set_featureset[:n_split], train_set_featureset[n_split:]
y_train, y_test = train_set_tags[:n_split], train_set_tags[n_split:]

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(X_train)
X_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_counts1 = count_vect.transform(X_test)
X_tfidf1 = tfidf_transformer.transform(X_train_counts1)

bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=400,
    learning_rate=1)

# bdt_discrete = AdaBoostClassifier(
#     DecisionTreeClassifier(max_depth=2),
#     n_estimators=600,
#     learning_rate=1.5,
#     algorithm="SAMME")

bdt_real.fit(X_tfidf.toarray(), y_train)
# bdt_discrete.fit(X_tfidf, y_train)

print bdt_real.score(X_tfidf1.toarray(), y_test)
# print bdt_discrete.predict(testToVec)
