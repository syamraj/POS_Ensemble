text_list = []
# with open('output.txt','rU') as fp:
with open('stack_testdata_Brown_gold','rU') as fp:
    for line in fp:
        text_list.append(line)

print len(text_list)


# n_split = int(len(text_list) * .8)
#
# print len(text_list[:n_split])
# print len(text_list[n_split:])

file = open('testdata_brown_based_on_stack_testdata_gold.txt', 'wt')

for line in text_list:
    line1 = ''
    for words in line[:-1].split(' '):
        if len(words.split('_')) == 2:
            line1 = line1 + words.replace('_', '/') + ' '
            # line1 = line1 + words.split('_')[0] + ' '
#         # else:
#         #     word
    file.write(line1+'\n')
#     # print line[:-1]