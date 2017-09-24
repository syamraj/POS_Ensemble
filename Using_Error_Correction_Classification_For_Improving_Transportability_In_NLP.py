from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

import opennlp
import nltk

pos = opennlp.OpenNLP("/home/devil/Thesis/apache-opennlp-1.8.0", "POSTagger", "en-pos-maxent.bin")
# pos = opennlp.OpenNLP("/home/devil/Thesis/apache-opennlp-1.8.0", "POSTagger", "en-pos-maxent-brown.bin")

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

def pos_features(sentence, i, history):
    features = ''
    features += sentence[i][-3:]

    if i == 0:
        features += "<START>"
    else:
        features += sentence[i - 1]
        # features += history[i - 1]
    return features


def processing(train_sents):
    for tagged_sent in train_sents:
        # print "tagged_sent", tagged_sent
        untagged_sent = nltk.tag.untag(tagged_sent)
        # print "untagged_sent", untagged_sent
        for i, (word, tag) in enumerate(tagged_sent):
            featureset = pos_features(untagged_sent, i, history)
            train_set_featureset.append(featureset)
            train_set_tags.append(tag)
            # train_set.append((featureset, tag))
            history.append(tag)

def processing_learning_data(train_sents_learning_data):
    sum1 = 0
    for tagged_sent_learning_data in train_sents_learning_data:
        untagged_sent = nltk.tag.untag(tagged_sent_learning_data)
        sum1 = sum1 + len(untagged_sent)
        # print "length of untagged sent", len(untagged_sent)
        for i, (word, tag) in enumerate(tagged_sent_learning_data):
            featureset = pos_features(untagged_sent, i, history)
            train_set_featureset_learning_data.append(featureset)
            train_set_tags_learning_data.append(tag)
            # train_set.append((featureset, tag))
            history_learning_data.append(tag)
    print "length of untagged sent", sum1

def processing_testdata(train_sents_testdata):
    sum1 = 0
    for tagged_sent_testdata in train_sents_testdata:
        untagged_sent = nltk.tag.untag(tagged_sent_testdata)
        sum1 = sum1 + len(untagged_sent)
        # print "length of untagged sent", len(untagged_sent)
        for i, (word, tag) in enumerate(tagged_sent_testdata):
            featureset = pos_features(untagged_sent, i, history)
            train_set_featureset_testdata.append(featureset)
            train_set_tags_testdata.append(tag)
            # train_set.append((featureset, tag))
            history_testdata.append(tag)

# training_data_size = [500, 4000, 8000, 10000, 15000, 20000]
training_data_size = [200]

# for training_data_size_iterator in range(len(training_data_size)):
for training_data_size_iterator in range(len(training_data_size)):
    initialize()
    Tag_list = []
    Per_tag_acc_list = []
    testdata_complete = []
    testdata_complete_test = []
    testdata_gold_complete = []

    # Lists for overall training.
    train_set_featureset = []
    train_set_tags = []
    history = []

    # Lists for learning_data.
    train_set_featureset_learning_data = []
    train_set_tags_learning_data = []
    history_learning_data = []

    # Lists for testdata.
    train_set_featureset_testdata = []
    train_set_tags_testdata = []
    history_testdata = []

    train_sents = []
    train_sents_learning_data = []
    train_sents_testdata = []

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

    file_list = ['stack_testdata_Brown_gold', 'output.txt']
    Genia_files = ['testdata.txt', 'testdata_gold.txt']
    Brown_files = ['testdata_brown_based_on_stack_testdata.txt',
                   'testdata_brown_based_on_stack_testdata_gold.txt']

    overall_text = []
    text_for_training = []
    text_to_learn_base_error = []
    text_to_learn_base_error_only_text = []

    with open(file_list[1], 'rU') as splitting:
        for line in splitting:
            overall_text.append(line[:-1].lstrip().rstrip())

    train_and_test_split = int(len(overall_text) * .2)
    text_for_training = overall_text[:train_and_test_split]

    for line in text_for_training:
        str = line.lstrip().rstrip().split(' ')
        listEachLine = []
        for i in str:
            if i.__contains__('_'):
                listEachLine.append((i.split('_')[0], i.split('_')[1]))
        train_sents.append(listEachLine)

    processing(train_sents)

    # setting up the Classifier1 (training with 80% of the data provided)

    train_set_featureset = train_set_featureset[:training_data_size[training_data_size_iterator]]
    train_set_tags = train_set_tags[:training_data_size[training_data_size_iterator]]

    Classifier1X_train = train_set_featureset
    Classifier1Y_train = train_set_tags

    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    Classifier1X_train_counts = count_vect.fit_transform(Classifier1X_train)
    Classifier1X_tfidf = tfidf_transformer.fit_transform(Classifier1X_train_counts)

    clf1 = RandomForestClassifier(n_estimators=10)
    clf1.fit(Classifier1X_tfidf, Classifier1Y_train)


    # setting up the data needed to learn the mistakes made by the base tagger.

    text_to_learn_base_error = text_for_training[:int(len(text_for_training) * .2)]

    for line_for_text_processig in text_to_learn_base_error:
        temp_line = ''
        word_list = line_for_text_processig.lstrip().rstrip().split(' ')
        for word in word_list:
            if word[0] == '_':
                word = word[1:]
            temp_line = temp_line + word.split('_')[0] + ' '
        temp_line = temp_line.replace('  ', ' ')
        text_to_learn_base_error_only_text.append(temp_line.lstrip().rstrip())

    print "Sample text processed for the use of Base POS tagger..."
    print text_to_learn_base_error_only_text[:5]

    for line1 in text_to_learn_base_error_only_text:
        line_pos = pos.parse(line1.replace('  ', ' '))
        if len(line1.split(' ')) != len(line_pos.split(' ')):
                print "this line has error-----------------------------"
                print len(line1.split(' ')), line1
                print len(line_pos.split(' ')), line_pos
        testdata_complete.append(line_pos)

    print "Printing sample sentences after processing with base POS tagger..."
    print testdata_complete[:5]

    # setting the same data as above in the format required for the Classifier1

    for line3 in text_to_learn_base_error:
        line3 = line3.replace('  ', ' ')
        line3 = line3.replace(' _', ' ')
        str3 = line3.split(' ')
        listEachLine = []
        for i3 in str3:
            if i3.__contains__('_'):
                listEachLine.append((i3.split('_')[0], '--'))
            else:
                listEachLine.append((i3, '--'))
        train_sents_learning_data.append(listEachLine)

    processing_learning_data(train_sents_learning_data)


    print "Printing sample features from the data used to learn the errors..."
    print train_set_featureset_learning_data[:100]
    print "Total length of the feature list of the data used to learn the error..."
    print len(train_set_featureset_learning_data)

    # Checking the consistency of the data processed in each step
    data_from_base_tagger = []
    for lines_in_testdata_complete in testdata_complete:
        word_list_in_lines_in_testdata_complete = lines_in_testdata_complete.split(' ')
        for token in word_list_in_lines_in_testdata_complete:
            data_from_base_tagger.append(token)

    print "Total length of the token from the data after processing from base tagger..."
    print len(data_from_base_tagger)

    token_holder5 = []
    for line5 in text_to_learn_base_error:
        line5 = line5.replace('  ', ' ')
        wordsl5 = line5.lstrip().rstrip().split(' ')
        for words5 in wordsl5:
            if words5.__contains__('_'):
                token_holder5.append(words5)
            else:
                words5 = words5[:len(words5)] + '_' + '!'
                token_holder5.append(words5)
    print "Length of text_to_learn_base_error"
    print len(token_holder5)

    token_holder6 = []
    for line6 in text_to_learn_base_error_only_text:
        wordsl6 = line6.lstrip().rstrip().split(' ')
        for words6 in wordsl6:
            token_holder6.append(words6)
    print "Length of text_to_learn_base_error_only_text"
    print len(token_holder6)

    for iter1 in range(len(text_to_learn_base_error)):
        if len(text_to_learn_base_error[iter1].split(' ')) != len(text_to_learn_base_error_only_text[iter1].split(' ')):
            print " There are miss match with text_to_learn_base_error and text_to_learn_base_error_only_text"
            print len(text_to_learn_base_error[iter1].split(' ')), text_to_learn_base_error[iter1]
            print len(text_to_learn_base_error_only_text[iter1].split()), text_to_learn_base_error_only_text[iter1]

    # learning the error and preparing new features to train the correction classifier.

    list_of_error_learned = []
    for iter2 in range(len(data_from_base_tagger)):
        temporary_string = ''
        if data_from_base_tagger[iter2].split('_')[1] != token_holder5[iter2].split('_')[1]:
            string_to_predict_list = [train_set_featureset_learning_data[iter2]]
            learning_error_train_counts = count_vect.transform(string_to_predict_list)
            learning_error_tfidf = tfidf_transformer.transform(learning_error_train_counts)
            temporary_string = temporary_string + data_from_base_tagger[iter2].split('_')[1] + \
                           clf1.predict(learning_error_tfidf)[0] + '_' + token_holder5[iter2].split('_')[1]
            list_of_error_learned.append(temporary_string)

    print "List of error learned"
    print list_of_error_learned

    # Splitting list_of_error_correction to features and labels set

    CorrectionClassifierX_featureset = []
    CorrectionClassifierY_tags = []

    for iter3 in list_of_error_learned:
        CorrectionClassifierX_featureset.append(iter3.split('_')[0])
        CorrectionClassifierY_tags.append(iter3.split('_')[1])

    # Making the correction classifier

    count_vect1 = CountVectorizer()
    tfidf_transformer1 = TfidfTransformer()

    CorrectionClassifierX_train_counts = count_vect1.fit_transform(CorrectionClassifierX_featureset)
    CorrectionClassifierX_tfidf = tfidf_transformer1.fit_transform(CorrectionClassifierX_train_counts)

    CorrectionClassifier = RandomForestClassifier(n_estimators=10)
    CorrectionClassifier.fit(CorrectionClassifierX_tfidf, CorrectionClassifierY_tags)

    # Area to give the test file

    testdata_complete_test_pure_text = []
    # with open('/home/devil/Thesis/testdata.txt','rU') as fp:
    # with open(Genia_files[0], 'rU') as fp:
    with open(Brown_files[0], 'rU') as fp:
        for line4 in fp:
            line4 = line4.replace('  ', ' ')
            line4 = line4.replace(' _', ' ')
            line2 = pos.parse(line4[:-1])
            testdata_complete_test.append(line2.lstrip().rstrip())
            testdata_complete_test_pure_text.append(line4[:-1].lstrip().rstrip())

    # Area to give the MAP file
    with open('/home/devil/Thesis/map.txt', 'rU') as fp:
        for line in fp:
            map_list.append(line)

    # Gold file for the test purpose

    # with open('/home/devil/Thesis/testdata_gold.txt', 'rU') as fp:
    # with open(Genia_files[1], 'rU') as fp:
    with open(Brown_files[1], 'rU') as fp:
        for line7 in fp:
            line7 = line7.replace('  ', ' ')
            line7 = line7.replace(' _', ' ')
            testdata_gold_complete.append(line7[:-1].lstrip().rstrip())

    # Making the test file data compatible for the prediction data.

    for iter4 in testdata_complete_test_pure_text:
        str4 = iter4.split(' ')
        testdata_list_each_line = []
        for i4 in str4:
            testdata_list_each_line.append((i4, '--'))
        train_sents_testdata.append(testdata_list_each_line)

    processing_testdata(train_sents_testdata)

    # Tokenizing the test data

    token_holder_for_testdata_complete_test = []
    for line8 in testdata_complete_test_pure_text:
        tokens8 = line8.split(' ')
        for words8 in tokens8:
            token_holder_for_testdata_complete_test.append(words8)

    token_holder_for_testdata_complete_test_gold_text = []
    token_holder_for_testdata_complete_test_gold_tag = []
    token_holder_for_testdata_complete_test_gold_all = []

    for line9 in testdata_gold_complete:
        tokens9 = line9.split(' ')
        for words9 in tokens9:
            # if words9.__contains__('_'):
            #     if words9[0] == '_':
            #         words9[0] = words9[1:]
            # else:
            #     words9 = words9 + '_' + '!'
            token_holder_for_testdata_complete_test_gold_text.append(words9.split('/')[0])
            token_holder_for_testdata_complete_test_gold_tag.append(words9.split('/')[1])
            token_holder_for_testdata_complete_test_gold_all.append(words9)

    token_holder_for_testdata_basePOS_complete = []
    for line10 in testdata_complete_test:
        words10 = line10.split(' ')
        for tokens10 in words10:
            token_holder_for_testdata_basePOS_complete.append(tokens10)

    # Area to check the F1 of correction classifier
    # Checking the size of test data.

    print "Length of the features of the testdata...and some samples"
    print len(train_set_featureset_testdata), train_set_featureset_testdata[:5]

    print "Length of the testdata gold text tokens...and some samples"
    print len(token_holder_for_testdata_complete_test_gold_text), token_holder_for_testdata_complete_test_gold_text[:5]

    print "Length of the testdata gold tag tokens...and some samples"
    print len(token_holder_for_testdata_complete_test_gold_tag), token_holder_for_testdata_complete_test_gold_tag[:5]

    print "Length of the tokens of the testdata ...and some samples"
    print len(token_holder_for_testdata_complete_test), token_holder_for_testdata_complete_test[:5]

    print "Length of the tokens of the testdata after base POS tagging...and some samples"
    print len(token_holder_for_testdata_basePOS_complete), token_holder_for_testdata_basePOS_complete[:5]

    # Checking the base classifier F1 score

    if len(token_holder_for_testdata_basePOS_complete) != len(token_holder_for_testdata_complete_test_gold_all):
        print "line number ", i, "in the file not in sync"
    else:
        for j in range(len(token_holder_for_testdata_basePOS_complete)):
            if token_holder_for_testdata_basePOS_complete[j].split('_')[0].upper() == \
                    token_holder_for_testdata_complete_test_gold_all[j].split('/')[0].upper():
                if token_holder_for_testdata_basePOS_complete[j].split('_')[1].upper() == \
                        token_holder_for_testdata_complete_test_gold_all[j].split('/')[1].upper():
                    overall_tpos_before_correction = overall_tpos_before_correction + 1
                else:
                    overall_fpos_before_correction = overall_fpos_before_correction + 1
                    overall_fn_before_correction = overall_fn_before_correction + 1

    f1_overall_before_correction = float(overall_tpos_before_correction * 2) / \
                                    float(2 * overall_tpos_before_correction + overall_fn_before_correction
                                            + overall_fpos_before_correction)

    print "F1 score before correction...", f1_overall_before_correction

    # F1 after doing the correction with CorrectionClassifier

    tp = 0
    fpos = 0
    fn = 0
    # testdata_line_chunks = testdata_completelist[i].split(' ')
    # testdata_gold_line_chunks = testdata_gold_completelist[i].split(' ')
    # print len(testdata_line_chunks)
    # print len(testdata_gold_line_chunks)
    if len(token_holder_for_testdata_basePOS_complete) != len(token_holder_for_testdata_complete_test_gold_all):
        print "line number ", i, "in the file not in sync"
    else:
        for j in range(len(token_holder_for_testdata_basePOS_complete)):
            if token_holder_for_testdata_basePOS_complete[j].split('_')[0].upper() == \
                    token_holder_for_testdata_complete_test_gold_all[j].split('/')[0].upper():
                print token_holder_for_testdata_basePOS_complete[j].split('_')[0] + '............' + \
                      token_holder_for_testdata_complete_test_gold_all[j].split('/')[0]
                print token_holder_for_testdata_basePOS_complete[j].split('_')[1]
                print "gold", token_holder_for_testdata_complete_test_gold_all[j].split('/')[1]
                if token_holder_for_testdata_basePOS_complete[j].split('_')[1].upper() == \
                        token_holder_for_testdata_complete_test_gold_all[j].split('/')[1].upper():
                    tp = tp + 1
                    overall_tpos_after_correction = overall_tpos_after_correction + 1
                    # print "tp incremented"
                else:

                    text_for_correction = token_holder_for_testdata_basePOS_complete[j].split('_')[0][-3:] + \
                                          token_holder_for_testdata_basePOS_complete[j - 1].split('_')[0]
                    text_for_correction_list = [text_for_correction]

                    X_train_counts1 = count_vect.transform(text_for_correction_list)
                    X_tfidf1 = tfidf_transformer.transform(X_train_counts1)
                    new_tag = clf1.predict(X_tfidf1)

                    text_for_correction1 = token_holder_for_testdata_basePOS_complete[j].split('_')[1] + new_tag[0]

                    print "text for correction", text_for_correction1

                    X_train_counts2 = count_vect1.transform([text_for_correction1])
                    X_tfidf2 = tfidf_transformer1.transform(X_train_counts2)
                    final_predicted_tag = CorrectionClassifier.predict(X_tfidf2)

                    predicted_class_list.append(new_tag)
                    # if not predicted_class.__contains__(new_tag):
                    #     predicted_class.append(new_tag)
                    new_wrd_tag_pair = token_holder_for_testdata_basePOS_complete[j].split('_')[0] + '_' +\
                                       final_predicted_tag[0]

                    # testdata_completelist[i].replace(token_holder_for_testdata_basePOS_complete[j], new_wrd_tag_pair)
                    token_holder_for_testdata_basePOS_complete[j] = new_wrd_tag_pair
                    for line in map_list:
                        if token_holder_for_testdata_basePOS_complete[j].split('_')[0].upper() == \
                                line.split('_')[0].upper():
                            token_holder_for_testdata_basePOS_complete[j] = line[:-1]
                    # print "############", token_holder_for_testdata_basePOS_complete[j]
                    if token_holder_for_testdata_basePOS_complete[j].split('_')[1].upper() != \
                            token_holder_for_testdata_complete_test_gold_all[j].split('/')[1].upper():
                        fpos = fpos + 1
                        overall_fpos_after_correction = overall_fpos_after_correction + 1
                        fn = fn + 1
                        overall_fn_after_correction = overall_fn_after_correction + 1
                    if token_holder_for_testdata_basePOS_complete[j].split('_')[1].upper() == \
                            token_holder_for_testdata_complete_test_gold_all[j].split('/')[1].upper():
                        tp = tp + 1
                        overall_tpos_after_correction = overall_tpos_after_correction
                        # print "tp incremented"
        # print "tp = ", tp
        # print "fn = ", fn
        accuracy = float(tp)*2 / float(2*tp + fn + fpos)
        accuracy_list_after_correction.append(accuracy)
        # print '@@@@@@@@@@@@@@@@@@@@@@@@@@@', testdata_line_chunks

    f1_overall_after_correction = float(overall_tpos_after_correction * 2) / float(2 * overall_tpos_after_correction
                                                                                   + overall_fn_after_correction
                                                                                   + overall_fpos_after_correction)

    print "F1 score after the doing the correction from the Correction Classifier...", f1_overall_after_correction


    # with open(file_list[1], 'rU') as fp:
    #     for line in fp:
    #         str = line.split(' ')
    #         listEachLine = []
    #         for i in str[:-1]:
    #             if i.__contains__('_'):
    #                 listEachLine.append((i.split('_')[0], i.split('_')[1]))
    #         train_sents.append(listEachLine)

    # processing(train_sents)
    # print train_sents

    # train_set_featureset = train_set_featureset[:training_data_size[training_data_size_iterator]]
    # train_set_tags = train_set_tags[:training_data_size[training_data_size_iterator]]
    #
    # n_split = int(len(train_set_featureset) * .8)
    #
    # Classifier1X_train = train_set_featureset[:n_split]
    # Classifier1Y_train = train_set_tags[:n_split]
    #
    # n_split1 = int(len(Classifier1X_train) * .2)
    #
    # Data_for_learning_from_baseX_train = Classifier1X_train[:n_split1]
    # Data_for_learning_from_baseY_train = Classifier1Y_train[:n_split1]
    #
    # count_vect = CountVectorizer()
    # tfidf_transformer = TfidfTransformer()
    #
    # Classifier1X_train_counts = count_vect.fit_transform(Classifier1X_train)
    # Classifier1X_tfidf = tfidf_transformer.fit_transform(Classifier1X_train_counts)
    #
    # clf1 = RandomForestClassifier(n_estimators=10)
    # clf1.fit(Classifier1X_tfidf, Classifier1Y_train)
    #
    # clf2 = RandomForestClassifier(n_estimators=10)
    # clf2.fit(Classifier1X_tfidf, Classifier1Y_train)

    # print(clf.score(X_tfidf1.toarray(), y_test))

    # Area to give the test file
    # with open('/home/devil/Thesis/testdata.txt','rU') as fp:
    # with open(Genia_files[0], 'rU') as fp:
    # with open(Brown_files[0], 'rU') as fp:
    #     for line in fp:
    #         line2 = pos.parse(line[:-1])
    #         testdata_complete.append(line2.rstrip())

    # Map file
    # with open('/home/devil/Thesis/map.txt', 'rU') as fp:
    #     for line in fp:
    #         map_list.append(line)

    # Gold file for the test purpose
    # with open('/home/devil/Thesis/testdata_gold.txt', 'rU') as fp:
    # with open(Genia_files[1], 'rU') as fp:
    # with open(Brown_files[1], 'rU') as fp:
    #     for line in fp:
    #         testdata_gold_complete.append(line[:-1].rstrip())
    # print testdata_complete
    # print testdata_gold_complete
    # print 'accuracy before correction', accuracy_pos(testdata_complete, testdata_gold_complete)
    # print "..........................................................................................................."
    # print 'accuracy after correction', accuracy_pos_withcorrection(testdata_complete, testdata_gold_complete, map_list)
    #
    # f1_overall_before_correction = float(overall_tpos_before_correction * 2) / \
    #                                float(2 * overall_tpos_before_correction + overall_fn_before_correction
    #                                      + overall_fpos_before_correction)
    # f1_overall_before_correction_list.append(f1_overall_before_correction)
    #
    # f1_overall_after_correction = float(overall_tpos_after_correction * 2) / float(
    #     2 * overall_tpos_after_correction
    #     + overall_fn_after_correction
    #     + overall_fpos_after_correction)
    # f1_overall_after_correction_list.append(f1_overall_after_correction)
    #
    # print "printing the result", accracy_list_before_correction
    # print "printing the result after correction", accuracy_list_after_correction
    #
    # print "length of accuracy list before correction", len(accracy_list_before_correction)
    # print "length of accuracy list after correction", len(accuracy_list_after_correction)
    #
    # print "overall F1 score after correction", f1_overall_after_correction
    # print "overall F1 score before correction", f1_overall_before_correction


    # plot of the accuracies with increase in the training data
    # print 'length of training data size', len(training_data_size)
    # print 'length of f1_overall_after_correction', len(f1_overall_after_correction_list)
    # print 'f1_overall_after_correction', f1_overall_after_correction_list
    # x = np.array(training_data_size)
    # y = np.array(f1_overall_after_correction_list)
    # f_b = interp1d(x, y)
    # f1_b = interp1d(x, y)
    # f2_b = interp1d(x, y, kind='cubic')
    # x1 = np.array(training_data_size)
    # y1 = np.array(f1_overall_before_correction_list)
    # f = interp1d(x1, y1)
    # f1 = interp1d(x1, y1)
    # f2 = interp1d(x1, y1, kind='cubic')
    # xnew = np.linspace(training_data_size[0], training_data_size[len(training_data_size)-1], num=41, endpoint=True)
    # plt.plot(x, y, 'o', xnew, f_b(xnew), '-', xnew, f2_b(xnew), '--')
    # plt.plot(x1, y1, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
    # plt.legend(['data', 'linear', 'cubic'], loc='best')
    # plt.show()
