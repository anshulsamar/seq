from random import shuffle
import random
import math
import string
import sys
import os
import string
import pdb

eng_punctuation = string.punctuation + '\xe2\x80\x9c' + '\xe2\x80\x9d'
hindi_punctuation = ['\xe0\xa5\xa4']

def splitSentence(from_sent):
    split_sent = []
    start = 0
    from_sent = from_sent.split()
    for word in from_sent:
        start = 0
        if word in string.punctuation or word in hindi_punctuation:
            split_sent.append(word)
        else:
            for i in range(0,len(word)):
                if word[i] in string.punctuation or word[i] in hindi_punctuation:
                    if word[start:i] != '':
                        split_sent.append(word[start:i])
                    split_sent.append(word[i])
                    start = i + 1
            if word[start::] != '':
                split_sent.append(word[start::])
    return split_sent

def addToVocab(vocab,sent):
    for i in range(0,len(sent)):
        if sent[i] in vocab:
            vocab[sent[i]] = vocab[sent[i]] + 1
        else:
            vocab[sent[i]] = 1

def removeOOV(vocab,sent):
    new_sent = ''
    for i in range(0,len(sent)):
        if vocab.index(sent[i]) < 10000:
            new_sent = new_sent + sent[i] + ' ' 
        else:
            new_sent = new_sent + '<unk> '
    return new_sent

print sys.argv

if len(sys.argv) == 1:
    print('python parseHindi.py nameOfSaveDir')
    exit()

data_dir = '/deep/group/speech/asamar/nlp/data/hindi/hindiSource/'
save_dir = sys.argv[1] + '/'

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
train_f = open(save_dir + 'ptb.train.txt', 'w')
test_f = open(save_dir + 'ptb.test.txt', 'w')
for line in lines:
    if count < train_set_size:
        train_f.write(line)
    else:
        test_f.write(line)
data.close()
train_f.close()
test_f.close()

print('Building Vocabulary from Training Set')

punctuation = string.punctuation
eng_vocab = {}
hindi_vocab = {}

data = open(save_dir + 'ptb.train.txt','r')
num_lines = 0
for line in data:
    pdb.set_trace()
    orig_line = line.lower().strip()
    s = orig_line.split('\t')
    eng_sent = splitSentence(s[3])
    print(eng_sent)
    addToVocab(eng_vocab,eng_sent)
    hindi_sent = splitSentence(s[4])
    addToVocab(hindi_vocab,hindi_sent)
data.close()

print('Parsing Train and Test Set')
for a in [['ptb.train.txt','enc_train.txt','dec_train.txt'],['ptb.test.txt','enc_test.txt','dec_test.txt']]:
    data = open(save_dir + a[0],'r')
    enc_to = open(save_dir + a[1],'w')
    dec_to = open(save_dir + a[2],'w')
    eng_vocab = sorted(eng_vocab,key = lambda x: eng_vocab[x])
    eng_vocab.reverse()
    hindi_vocab = sorted(hindi_vocab,key = lambda x: hindi_vocab[x])
    hindi_vocab.reverse()

    for line in data:
        orig_line = line.lower().strip()
        s = orig_line.split('\t')
        eng_sent = splitSentence(s[3])
        enc_to.write(removeOOV(eng_vocab,eng_sent).strip() + '\n')
        hindi_sent = splitSentence(s[4])
        dec_to.write(removeOOV(hindi_vocab,hindi_sent).strip() + ' <eos>\n')

    data.close()
    enc_to.close()
    dec_to.close()
