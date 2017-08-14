list_brown = []

file = open('stack_testdata_Brown_gold','wt')
with open('/home/devil/Thesis/ce_all','rU') as fp:
    for line in fp:
        line = line.lstrip().rstrip().rstrip('\n')
        line = line.replace('/', '_')
        if line != '':
            # print line
            file.write(line + '\n')
        # line1 = ''
        # for words in line.split(' '):
        #     line1 = line1 + words.split('/')[0] + ' '
        # file.write(line+'\n')
