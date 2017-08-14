class_list = []

with open('output.txt','rU') as fp:
    for line in fp:
        line_list = line.split(' ')[:-1]
        for words in line_list:
            if len(words.split('_')) == 2:
                if not class_list.__contains__(words.split('_')[1]):
                    class_list.append(words.split('_')[1])

print class_list
