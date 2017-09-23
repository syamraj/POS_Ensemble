import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn import svm
import matplotlib.pyplot as plt
from itertools import cycle


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

train_set_featureset = train_set_featureset[:500]
train_set_tags = train_set_tags[:500]

n_split = int(len(train_set_featureset) * .7)

X_train, X_test = train_set_featureset[:n_split], train_set_featureset[n_split:]
y_train, y_test = train_set_tags[:n_split], train_set_tags[n_split:]

# print X_train[0]

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2
sample = ['ncysimianJJ', 'rusimmunodeficiencyNN', ',virusNN', 'and,IN', 'ionandNN', 'howassaysLS', 'hisshow:', 'itethisNN', 'tositeIN', 'atetoDT', 'V-2mediateNN', 'cerHIV-2NN', 'ionenhancerNN', 'ingactivationVBZ', 'ionfollowingDT', 'ofstimulationJJ', 'ticofJJ', 'butmonocyticNN', 'notbutIN', 'ellnotNN', 'nesT-cellNN', '.linesNN', 'his<START>', 'isThisLS', 'theis:', 'rsttheNN', 'ionfirstNN', 'ofdescriptionNN', 'anofCC', 'V-2anNN', 'cerHIV-2NN', 'entenhancerNN', 'ichelementIN', 'ayswhichNN', 'uchdisplaysVBZ', 'ytesuchJJ', 'itymonocyteNN', ',specificityNN', 'and,IN', 'noandNN', 'bleno.', 'cercomparableLS', 'entenhancer:', 'haselementNN', 'eenhasIN', 'rlybeenDT', 'nedclearlyNN', 'fordefinedNN', 'V-1forNN', '.HIV-1VBZ', 'ile<START>', 'aWhileLS', 'eara:', 'tornuclearNN', '(factorNN', 's(NN', ')sCC', 'rom)NN', 'othfromNN', 'ralbothNN', 'oodperipheralIN', 'tesbloodNN', 'andmonocytesVBZ', 'TandJJ', 'llsTNN', 'ndscellsNN', 'thebindsIN', 'ppatheNN', 'Bperi-kappa.', 'iteBLS', ',site:', 'tic,NN', 'ityelectrophoreticIN', 'iftmobilityDT', 'aysshiftNN', 'estassaysNN', 'hatsuggestNN', 'herthatVBZ', 'aeitherDT', 'entaJJ', 'eindifferentJJ', 'ndsproteinNN', 'tobindsIN', 'histoNN', 'itethisNN', 'insiteNN', 'tesinVBG', 'susmonocytesIN', 'TversusVBN', 'llsTNN', 'orcellsIN', 'hatorNN', 'thethat(', 'eintheNN', 'ingprotein)', 'hisrecognizingCC', 'certhisNN', 'entenhancerNN', 'oeselement.', 'ialundergoesIN', 'iondifferentialJJ', 'inmodificationNN', 'tesinNNS', 'andmonocytesPRP', 'TandVBP', 'llsTIN', ',cellsNN', 'hus,NN', 'ingthusVBZ', 'thesupportingTO', 'iontheDT', 'atatransfectionJJ', '.dataJJ', 'her<START>', ',FurtherLS', 'ile,:', 'ficwhileNN', 'ivespecificNN', 'ingconstitutiveNN', 'tobindingCC', 'thetoNN', 'ppatheNN', 'Bperi-kappaNN', 'iteBIN', 'issiteNN', 'eenisVBZ', 'inseenJJ', 'tesinNN', ',monocytesNN', 'ion,IN', 'ithstimulationNN', 'bolwith.', 'ersphorbolLS', 'cesesters:', 'nalinducesNN', ',additionalIN', 'fic,DT', 'ingspecificNN', '.bindingNN', 'ing<START>', 'theUnderstandingLS', 'ficthe:', 'ionmonocyte-specificNN', 'offunctionNN', 'theofNN', 'ppatheCC', 'Bperi-kappaNN', 'torBNN', 'mayfactorNN']
X_train_counts = count_vect.fit_transform(X_train)
X_tfidf = tfidf_transformer.fit_transform(X_train_counts)

class_list = ['LS', ':', 'NN', 'CC', 'IN', 'VBZ', 'JJ', 'DT', 'VBG', 'VBN', '(', ')', 'NNS', 'PRP', 'VBP', 'TO', 'WDT',
              '.', 'VBD', 'VB', ',', 'PRP$', 'RB', 'MD', 'CD', 'NNP', 'FW',
              'JJS', 'RBS', 'WRB', 'RP', 'EX', 'JJR', 'RBR', 'POS', 'WP$', 'WP', 'PDT',
              'SYM', '``', "''", '-', 'NNS|FW', 'JJ|NN', 'E2A', '',
              'VBN|JJ', 'JJ|NNS', 'JJ|VBN', 'VBG|JJ', 'JJ|VBG', 'IN|PRP$', 'NN|NNS', 'N', 'CT', 'XT',
              'NNPS', 'NN|CD', 'NN|DT', 'VBD|VBN', 'JJ|RB', 'PP']
# class_list = ['JJ', 'NN']
precision = dict()
recall = dict()
average_precision = dict()

X_train_counts1 = count_vect.transform(sample)
X_tfidf1 = tfidf_transformer.transform(X_train_counts1)
y_train = label_binarize(y_train, classes=class_list)
# y_train = label_binarize(y_train, classes=['JJ', 'NN'])

clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
print X_tfidf.shape
# print X_tfidf.toarray().shape
clf.fit(X_tfidf, y_train)

# print(clf.score(X_tfidf1, y_test))
print clf.predict(X_tfidf1)

y_predicted = label_binarize(clf.predict(X_tfidf1), classes=class_list)
print 'y_predicted', y_predicted
y_score = clf.predict_proba(X_tfidf1)
print 'X_tfidf1....................', X_tfidf1
print 'type of X_tfidf1....................', type(X_tfidf1)
print 'y_score', clf.predict_proba(X_tfidf1)[0]
temp = clf.predict_proba(X_tfidf1)
print 'type of predict proba result', type(temp)
print 'length of y_score', len(y_score)
print 'length of y_predicted', len(y_predicted)

# X_train_counts2 = count_vect.transform("eneIL-2NN")
# X_tfidf2 = tfidf_transformer.transform(X_train_counts1)

# print X_tfidf2.shape
# print X_tfidf2.toarray().shape


for i in range(len(class_list)):
    precision[i], recall[i], _ = precision_recall_curve(y_predicted[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_predicted[:, i], y_score[:, i])

# Compute micro-average ROC curve and ROC area
precision["micro"], recall["micro"], _ = precision_recall_curve(y_predicted.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(y_predicted, y_score,
                                                     average="micro")
plt.clf()
plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
         label='micro-average Precision-recall curve (area = {0:0.2f})'
               ''.format(average_precision["micro"]))
for i, color in zip(range(len(class_list)), colors):
    plt.plot(recall[i], precision[i], color=color, lw=lw,
             label='Precision-recall curve of class {0} (area = {1:0.2f})'
                   ''.format(i, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(loc="lower right")
plt.show()

