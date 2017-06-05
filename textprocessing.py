import re
import subprocess

line1 = ''
count = 1
list1 = []
# Read from a file and make it in the format for passing to the consecutive functions
file = open('output.txt','wt')
with open('Genia_text.txt','rU') as fp:
    for line in fp:
        line = line.replace('/', '_')
        if not re.match(r'=', line, re.M | re.I):
            # print line1
            line1 = line1 + line.strip('\n') + ' '
            # list1.append(line)
        #     line1 = ''
        else:
            file.write(line1+'\n')
            line1 = ''
            # line1 = line1 + ' ' + line.strip('\n')
# file.close()







