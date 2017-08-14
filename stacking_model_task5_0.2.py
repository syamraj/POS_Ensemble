import nltk
import numpy as np
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
train_combined = np.array([])
features_combined = []
train_sents = []
test_meta_feature = []
test_meta_predicted_value = np.array([])
X_train_s = []
y_train_s = []
X_test_s = []
y_test_s = []

def initialize():
    global train_set_featureset
    global train_set_tags
    global history
    global train_sents
    train_set_featureset = []
    train_set_tags = []
    history = []
    train_sents = []


def processing(train_sents):
    for tagged_sent in train_sents:
        untagged_sent = nltk.tag.untag(tagged_sent)
        for i, (word, tag) in enumerate(tagged_sent):
            featureset = pos_features(untagged_sent, i, history)
            train_set_featureset.append(featureset)
            train_set_tags.append(tag)
            # train_set.append((featureset, tag))
            history.append(tag)


# file_list = ['output.txt', 'stack_testdata_Brown_gold']
file_list = ['stack_testdata_Brown_gold', 'output.txt']


for file_list_iter in range(len(file_list)):
    initialize()
    print "file name =", file_list[file_list_iter]
    with open(file_list[file_list_iter], 'rU') as fp:
        for line in fp:
            str = line.split(' ')
            listEachLine = []
            for i in str[:-1]:
                if i.__contains__('_'):
                    listEachLine.append((i.split('_')[0], i.split('_')[1]))
            train_sents.append(listEachLine)

    processing(train_sents)

    train_set_featureset = train_set_featureset[:50000]
    train_set_tags = train_set_tags[:50000]

    n_split = int(len(train_set_featureset) * .8)

    X_train, X_test = train_set_featureset[:n_split], train_set_featureset[n_split:]
    y_train, y_test = train_set_tags[:n_split], train_set_tags[n_split:]

    X_train1 = X_train[:int(len(X_train) * .25)]
    temp = X_train[int(len(X_train) * .25):]
    X_train2 = temp[:int(len(temp) * .25)]
    temp = temp[int(len(temp) * .25):]
    X_train3 = temp[:int(len(temp) * .25)]
    temp = temp[int(len(temp) * .25):]
    X_train4 = temp[:int(len(temp) * .25)]
    temp = temp[int(len(temp) * .25):]

    y_train1 = y_train[:int(len(y_train) * .25)]
    temp = y_train[int(len(y_train) * .25):]
    y_train2 = temp[:int(len(temp) * .25)]
    temp = temp[int(len(temp) * .25):]
    y_train3 = temp[:int(len(temp) * .25)]
    temp = temp[int(len(temp) * .25):]
    y_train4 = temp[:int(len(temp) * .25)]
    temp = temp[int(len(temp) * .25):]

    print 'X_train1_len', len(X_train1)
    print 'y_train1_len', len(y_train1)

    print 'X_train2_len', len(X_train2)
    print 'y_train2_len', len(y_train2)

    print 'X_train3_len', len(X_train3)
    print 'y_train3_len', len(y_train3)

    print 'X_train4_len', len(X_train4)
    print 'y_train4_len', len(y_train4)

    set1_train = X_train2 + X_train3 + X_train4
    set1_test = X_train1
    set1_y_train = y_train2 + y_train3 + y_train4
    set1_y_test = y_train1

    set2_train = X_train1 + X_train3 + X_train4
    set2_test = X_train2
    set2_y_train = y_train1 + y_train3 + y_train4
    set2_y_test = y_train2

    set3_train = X_train2 + X_train3 + X_train4
    set3_test = X_train3
    set3_y_train = y_train2 + y_train3 + y_train4
    set3_y_test = y_train3

    set4_train = X_train1 + X_train2 + X_train3
    set4_test = X_train4
    set4_y_train = y_train1 + y_train2 + y_train3
    set4_y_test = y_train4

    print 'set1_train_len', len(set1_train)
    print 'set1_train_len', len(set1_y_train)

    print 'set2_train_len', len(set2_train)
    print 'set2_train_len', len(set2_y_train)

    print 'set3_train_len', len(set3_train)
    print 'set3_train_len', len(set2_y_train)

    print 'set4_train_len', len(set4_train)
    print 'set4_train_len', len(set2_y_train)

    train_sets = [set1_train, set2_train, set3_train, set4_train]
    test_set = [set1_test, set2_test, set3_test, set4_test]
    train_y_set = [set1_y_train, set2_y_train, set3_y_train, set4_y_train]
    test_y_set = [set1_y_test, set2_y_test, set3_y_test, set4_y_test]

    for iteration in range(len(train_sets)):
        count_vect = CountVectorizer()
        tfidf_transformer = TfidfTransformer()
        X_train_counts = count_vect.fit_transform(train_sets[iteration])
        X_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        X_train_counts_stack_test = count_vect.transform(test_set[iteration])
        X_tfidf_stack_test = tfidf_transformer.transform(X_train_counts_stack_test)
        clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
        clf.fit(X_tfidf, train_y_set[iteration])
        print 'iteration--', clf.predict(X_tfidf_stack_test)
        print 'score------', clf.score(X_tfidf_stack_test, test_y_set[iteration])
        temp = clf.predict(X_tfidf_stack_test)
        train_combined = np.append(train_combined, temp)
        features_combined = features_combined + test_set[iteration]
        y_train_s = y_train_s + test_y_set[iteration]

    clf_test = OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_train_counts = count_vect.fit_transform(X_train)
    X_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf_test.fit(X_tfidf, y_train)
    X_train_counts_stack_test_meta = count_vect.transform(X_test)
    X_tfidf_stack_test_meta = tfidf_transformer.transform(X_train_counts_stack_test_meta)
    temp = clf_test.predict(X_tfidf_stack_test_meta)
    # test_meta_predicted_value = np.append(test_meta_predicted_value, temp)
    # test_meta_feature = test_meta_feature + X_test
    if file_list[file_list_iter] == 'stack_testdata_Brown_gold':
        y_test_s = y_test_s + y_test
        test_meta_predicted_value = np.append(test_meta_predicted_value, temp)
        test_meta_feature = test_meta_feature + X_test



# file = open('/home/devil/Thesis/train_meta.txt', 'wt')
# line = 'original_feature' + ' ' + 'predicted_value'
# file.write(line+'\n')
for filewrite_iter in range(len(features_combined)):
    line = features_combined[filewrite_iter] + train_combined[filewrite_iter]
    X_train_s.append(line)
    # file.write(line+'\n')
# file.close()

print "length of test_meta_feature", len(test_meta_feature)
print "length of test_meta_predicted_value", len(test_meta_predicted_value)
# file = open('/home/devil/Thesis/test_meta.txt', 'wt')
# line = 'original_feature' + ' ' + 'predicted_value'
# file.write(line+'\n')
for filewrite_iter1 in range(len(test_meta_feature)):
    line = test_meta_feature[filewrite_iter1] + test_meta_predicted_value[filewrite_iter1]
    X_test_s.append(line)
    # file.write(line+'\n')
# file.close()

clf_S = OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()
X_train_counts_s = count_vect.fit_transform(X_train_s)
X_tfidf_s = tfidf_transformer.fit_transform(X_train_counts_s)
clf_S.fit(X_tfidf_s, y_train_s)
X_train_counts_s = count_vect.transform(X_test_s)
X_tfidf_s = tfidf_transformer.transform(X_train_counts_s)
print "final prediction score *******", clf_S.score(X_tfidf_s, y_test_s)

# count_vect = CountVectorizer()
# tfidf_transformer = TfidfTransformer()

# sample = ['ncysimianJJ', 'rusimmunodeficiencyNN', ',virusNN', 'and,IN', 'ionandNN', 'aystransfection.', 'howassaysLS', 'hisshow:', 'itethisNN', 'tositeIN', 'atetoDT', 'V-2mediateNN', 'cerHIV-2NN', 'ionenhancerNN', 'ingactivationVBZ', 'ionfollowingDT', 'ofstimulationJJ', 'ticofJJ', 'butmonocyticNN', 'notbutIN', 'ellnotNN', 'nesT-cellNN', '.linesNN', 'his<START>', 'isThisLS', 'theis:', 'rsttheNN', 'ionfirstNN', 'ofdescriptionNN', 'anofCC', 'V-2anNN', 'cerHIV-2NN', 'entenhancerNN', 'ichelementIN', 'ayswhichNN', 'uchdisplaysVBZ', 'ytesuchJJ', 'itymonocyteNN', ',specificityNN', 'and,IN', 'noandNN', 'bleno.', 'cercomparableLS', 'entenhancer:', 'haselementNN', 'eenhasIN', 'rlybeenDT', 'nedclearlyNN', 'fordefinedNN', 'V-1forNN', '.HIV-1VBZ', 'ile<START>', 'aWhileLS', 'eara:', 'tornuclearNN', '(factorNN', 's(NN', ')sCC', 'rom)NN', 'othfromNN', 'ralbothNN', 'oodperipheralIN', 'tesbloodNN', 'andmonocytesVBZ', 'TandJJ', 'llsTNN', 'ndscellsNN', 'thebindsIN', 'ppatheNN', 'Bperi-kappa.', 'iteBLS', ',site:', 'tic,NN', 'ityelectrophoreticIN', 'iftmobilityDT', 'aysshiftNN', 'estassaysNN', 'hatsuggestNN', 'herthatVBZ', 'aeitherDT', 'entaJJ', 'eindifferentJJ', 'ndsproteinNN', 'tobindsIN', 'histoNN', 'itethisNN', 'insiteNN', 'tesinVBG', 'susmonocytesIN', 'TversusVBN', 'llsTNN', 'orcellsIN', 'hatorNN', 'thethat(', 'eintheNN', 'ingprotein)', 'hisrecognizingCC', 'certhisNN', 'entenhancerNN', 'oeselement.', 'ialundergoesIN', 'iondifferentialJJ', 'inmodificationNN', 'tesinNNS', 'andmonocytesPRP', 'TandVBP', 'llsTIN', ',cellsNN', 'hus,NN', 'ingthusVBZ', 'thesupportingTO', 'iontheDT', 'atatransfectionJJ', '.dataJJ', 'her<START>', ',FurtherLS', 'ile,:', 'ficwhileNN', 'ivespecificNN', 'ingconstitutiveNN', 'tobindingCC', 'thetoNN', 'ppatheNN', 'Bperi-kappaNN', 'iteBIN', 'issiteNN', 'eenisVBZ', 'inseenJJ', 'tesinNN', ',monocytesNN', 'ion,IN', 'ithstimulationNN', 'bolwith.', 'ersphorbolLS', 'cesesters:', 'nalinducesNN', ',additionalIN', 'fic,DT', 'ingspecificNN', '.bindingNN', 'ing<START>', 'theUnderstandingLS', 'ficthe:', 'ionmonocyte-specificNN', 'offunctionNN', 'theofNN', 'ppatheCC', 'Bperi-kappaNN', 'torBNN', 'mayfactorNN']
# X_train_counts = count_vect.fit_transform(X_train)
# X_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# print 'X_test', X_test
# X_train_counts1 = count_vect.transform(X_test)
# X_tfidf1 = tfidf_transformer.transform(X_train_counts1)

# clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
# # print X_tfidf.toarray().shape
# clf.fit(X_tfidf, y_train)

# print(clf.score(X_tfidf1, y_test))
# print clf.predict(X_tfidf1)

# X_train_counts2 = count_vect.transform("eneIL-2NN")
# X_tfidf2 = tfidf_transformer.transform(X_train_counts1)

# print X_tfidf2.shape
# print X_tfidf2.toarray().shape
