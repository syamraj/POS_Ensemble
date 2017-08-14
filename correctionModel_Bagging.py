import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

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
# with open('testdata_Brown.txt', 'rU') as fp:
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

n_split = int(len(train_set_featureset) * .8)

X_train, X_test = train_set_featureset[:n_split], train_set_featureset[n_split:]
y_train, y_test = train_set_tags[:n_split], train_set_tags[n_split:]

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

# sample = ['ncysimianJJ', 'rusimmunodeficiencyNN', ',virusNN', 'and,IN', 'ionandNN', 'aystransfection.', 'howassaysLS', 'hisshow:', 'itethisNN', 'tositeIN', 'atetoDT', 'V-2mediateNN', 'cerHIV-2NN', 'ionenhancerNN', 'ingactivationVBZ', 'ionfollowingDT', 'ofstimulationJJ', 'ticofJJ', 'butmonocyticNN', 'notbutIN', 'ellnotNN', 'nesT-cellNN', '.linesNN', 'his<START>', 'isThisLS', 'theis:', 'rsttheNN', 'ionfirstNN', 'ofdescriptionNN', 'anofCC', 'V-2anNN', 'cerHIV-2NN', 'entenhancerNN', 'ichelementIN', 'ayswhichNN', 'uchdisplaysVBZ', 'ytesuchJJ', 'itymonocyteNN', ',specificityNN', 'and,IN', 'noandNN', 'bleno.', 'cercomparableLS', 'entenhancer:', 'haselementNN', 'eenhasIN', 'rlybeenDT', 'nedclearlyNN', 'fordefinedNN', 'V-1forNN', '.HIV-1VBZ', 'ile<START>', 'aWhileLS', 'eara:', 'tornuclearNN', '(factorNN', 's(NN', ')sCC', 'rom)NN', 'othfromNN', 'ralbothNN', 'oodperipheralIN', 'tesbloodNN', 'andmonocytesVBZ', 'TandJJ', 'llsTNN', 'ndscellsNN', 'thebindsIN', 'ppatheNN', 'Bperi-kappa.', 'iteBLS', ',site:', 'tic,NN', 'ityelectrophoreticIN', 'iftmobilityDT', 'aysshiftNN', 'estassaysNN', 'hatsuggestNN', 'herthatVBZ', 'aeitherDT', 'entaJJ', 'eindifferentJJ', 'ndsproteinNN', 'tobindsIN', 'histoNN', 'itethisNN', 'insiteNN', 'tesinVBG', 'susmonocytesIN', 'TversusVBN', 'llsTNN', 'orcellsIN', 'hatorNN', 'thethat(', 'eintheNN', 'ingprotein)', 'hisrecognizingCC', 'certhisNN', 'entenhancerNN', 'oeselement.', 'ialundergoesIN', 'iondifferentialJJ', 'inmodificationNN', 'tesinNNS', 'andmonocytesPRP', 'TandVBP', 'llsTIN', ',cellsNN', 'hus,NN', 'ingthusVBZ', 'thesupportingTO', 'iontheDT', 'atatransfectionJJ', '.dataJJ', 'her<START>', ',FurtherLS', 'ile,:', 'ficwhileNN', 'ivespecificNN', 'ingconstitutiveNN', 'tobindingCC', 'thetoNN', 'ppatheNN', 'Bperi-kappaNN', 'iteBIN', 'issiteNN', 'eenisVBZ', 'inseenJJ', 'tesinNN', ',monocytesNN', 'ion,IN', 'ithstimulationNN', 'bolwith.', 'ersphorbolLS', 'cesesters:', 'nalinducesNN', ',additionalIN', 'fic,DT', 'ingspecificNN', '.bindingNN', 'ing<START>', 'theUnderstandingLS', 'ficthe:', 'ionmonocyte-specificNN', 'offunctionNN', 'theofNN', 'ppatheCC', 'Bperi-kappaNN', 'torBNN', 'mayfactorNN']
X_train_counts = count_vect.fit_transform(X_train)
X_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_counts1 = count_vect.transform(X_test)
X_tfidf1 = tfidf_transformer.transform(X_train_counts1)

print "type of X_tfidf", type(X_tfidf)

clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
print X_tfidf.shape
# print X_tfidf.toarray().shape
clf.fit(X_tfidf, y_train)

print(clf.score(X_tfidf1, y_test))
print clf.predict(X_tfidf1)

# X_train_counts2 = count_vect.transform("eneIL-2NN")
# X_tfidf2 = tfidf_transformer.transform(X_train_counts1)

# print X_tfidf2.shape
# print X_tfidf2.toarray().shape
