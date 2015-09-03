import random
import math
import string
import sys
import os

save_dir = '/deep/group/speech/asamar/nlp/data/numbers/'

for a in [[100000, 'enc_train.txt','dec_train.txt'],[1000, 'enc_test.txt','dec_test.txt']]:
    enc_to = open(save_dir + a[1],'w')
    dec_to = open(save_dir + a[2],'w')
    for line in range(0,a[0]):
        sentence = ''
        length = random.randint(1,9)
        for i in range(0,length):
            sentence = sentence + str(random.randint(1,10)) + ' '
        enc_to.write(sentence + '<eos>\n')
        dec_to.write(sentence + '<eos>\n')

    
