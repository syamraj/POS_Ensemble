list_brown = []

file = open('testdata_Brown_processed','wt')
with open('testdata_Brown.txt','rU') as fp:
    for line in fp:
        line1 = ''
        for words in line.split(' '):
            line1 = line1 + words.split('/')[0] + ' '
        file.write(line1+'\n')
