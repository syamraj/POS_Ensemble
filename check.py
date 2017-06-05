import opennlp

# The model file should be inside `models` folder.
# instance = opennlp.OpenNLP("/home/devil/Thesis/apache-opennlp-1.8.0", "ChunkerME", "en-chunker.bin")
pos = opennlp.OpenNLP("/home/devil/Thesis/apache-opennlp-1.8.0", "POSTagger", "en-pos-maxent.bin")


def accuracy_pos(line2):
    tp = 0
    fpos = 0
    fn = 0
    with open('/home/devil/Thesis/testdata_gold.txt', 'rU') as fp:
        line_list = []
        for line in fp:
            line_list = line.split(' ')
        line2_list = line2.split(' ')
        if len(line_list) != len(line2_list):
            print "file not in sync"
        else:
            for i in range(len(line2_list)):
                # print line_list[i].split('_')[0] + ',,,,,,,,,,,,,,,,,,,,' + line2_list[i].split('_')[0]
                if line2_list[i].split('_')[0] == line_list[i].split('/')[0]:
                    print line_list[i].split('/')[0]+',,,,,,,,,,,,,,,,,,,,'+line2_list[i].split('_')[0]
                    if line2_list[i].split('_')[1] == line_list[i].split('/')[1]:
                        tp += 1
                    else:
                        fpos += 1
                        fn += 1
                accuracy = tp/(tp+fn)
    return accuracy


# print pos.parse('Hi this is a test')
# file = open('/home/devil/Thesis/output_testdata_python.txt','wt')
Tag_list = []
Per_tag_acc_list = []
with open('/home/devil/Thesis/testdata.txt','rU') as fp:
    for line in fp:
        line2 = pos.parse(line)
        print accuracy_pos(line2)