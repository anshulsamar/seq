from random import shuffle
import random
import math
import string
import sys
import os

print sys.argv

if len(sys.argv) == 1:
    print('python parseHindi.py nameOfSaveDir')
    exit()

data_dir = '/deep/group/speech/asamar/nlp/data/hindi/hindiSource/'
save_dir = '/deep/group/speech/asamar/nlp/data/' + sys.argv[1] + '/'

if not os.path.exists(save_dir):
    print('Making directory ' + save_dir)
    os.makedirs(save_dir)

print('Loading Original Hindi Dataset')
train_set_size = 273000
data = open(data_dir + 'hindencorp05.plaintext','r')
lines = data.readlines()

print('Shuffling Lines')
shuffle(lines)
count = 0

print('Creating Train and Test Source Sets')
train_f = open(save_dir .. 'ptb.train.txt')
test_f = open(save_dir .. 'ptb.test.txt')
for line in lines:
    if count < train_set_size:
        train_f.write(line)
    else:
        test_f.write(line)
data.close()
train_f.close()
test_f.close()

print('Building Vocabulary from Training Set')

punctuation = [',','.','!','?','|', ';', ':', '\'']
eng_vocab = {}
hindi_vocab = {}

for a in [['ptb.train.txt','enc_train.txt','dec_train.txt']:
    data = open(save_dir + a[0],'r')
    enc_to = open(save_dir + a[1],'w')
    dec_to = open(save_dir + a[2],'w')
    num_lines = 0
    for line in data:
        orig_line = line.lower().strip()
        s = orig_line.split('\t')
        eng_sent = string.strip(s[3])

        for i in range(0,len(eng_sent)):
            if eng_sent[i] not in punctuation:
                if eng_sent[i] in eng_vocab:
                    eng_vocab[eng_sent[i]] = eng_vocab[eng_sent[i]] + 1
                else:
                    eng_vocab[eng_sent[i]] = 1
                

        hindi_sent = string.strip(s[4])

        for i in range(0,len(hindi_sent)):
            if hindi_sent[i] not in punctuation:
                if hindi_sent[i] in hindi_vocab:
                    hindi_vocab[eng_sent[i]] = hindi_vocab[eng_sent[i]] + 1
                else:
                    hindi_vocab[eng_sent[i]] = 1

    data.close()

print('Parsing Train and Test Set')

for a in [['ptb.train.txt','enc_train.txt','dec_train.txt'],['ptb.test.txt','enc_test.txt','dec_test.txt']]:
    data = open(save_dir + a[0],'r')

    eng_vocab = sorted(eng_vocab,key = lambda x: x[1]).reverse()
    hindi_vocab = sorted(hindi_vocab,key = lambda x: x[1]).reverse()
   
    for line in data:
        orig_line = line.lower().strip()
        s = orig_line.split('\t')
        eng_sent = string.strip(s[3])
        eng_sent_n = ''
        for i in range(0,len(eng_sent)):
            if eng_sent[i] not in punctuation:
                if eng_vocab.index(eng_sent[i]) < 10000:
                    eng_sent_n = eng_sent_n + eng_sent[i] + ' ' 
                else:
                    eng_sent_n = eng_sent_n + eng_sent[i] + '<unk>'
        enc_to.write(eng_sent_n + ' <eos>\n')               

        hindi_sent = string.strip(s[4])
        hindi_sent_n = ''

        for i in range(0,len(hindi_sent)):
            if hindi_sent[i] not in punctuation:
                if hindi_vocab.index(hindi_sent[i]) < 10000:
                    hindi_sent_n = hindi_sent_n + hindi_sent[i] + ' ' 
                else:
                    hindi_sent_n = hindi_sent_n + hindi_sent[i] + '<unk>'

        dec_to.write(hindi_sent_n + ' <eos>\n')
        

    data.close()
    enc_to.close()
    dec_to.close()
