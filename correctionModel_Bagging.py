import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

__author__ = 'devil'


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

processing(train_sents)

train_set_featureset = train_set_featureset[:10000]
train_set_tags = train_set_tags[:10000]

n_split = int(len(train_set_featureset) * .7)

X_train, X_test = train_set_featureset[:n_split], train_set_featureset[n_split:]
y_train, y_test = train_set_tags[:n_split], train_set_tags[n_split:]

print X_train[0]

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(X_train)
X_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_counts1 = count_vect.transform(X_test)
X_tfidf1 = tfidf_transformer.transform(X_train_counts1)

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_tfidf.toarray(), y_train)

print(clf.score(X_tfidf1.toarray(), y_test))
