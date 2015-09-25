from random import shuffle
import random
import math
import string
import sys
import os
import string
import pdb

eng_punctuation = string.punctuation + '\xe2\x80\x9c' + '\xe2\x80\x9d'
hindi_punctuation = ['\xe0\xa5\xa4'] #danda

# words like don't are split into don ' t


def splitSentence(from_sent):
    split_sent = []
    start = 0
    from_sent = from_sent.split()
    for word in from_sent:
        start = 0
        if '\xe2\x80\x9c' in word:
            word = word.replace('\xe2\x80\x9c','\"')
        if '\xe2\x80\x9d' in word:
            word = word.replace('\xe2\x80\x9d','\"')
        if word in string.punctuation or word in hindi_punctuation:
            split_sent.append(word)
        else:
            for i in range(0,len(word)):
                if word[i] in string.punctuation or word[i] in hindi_punctuation:
                    if word[start:i] != '':
                        split_sent.append(word[start:i])
                        split_sent.append(word[i])
                        start = i + 1
                    #if word[i] == '\'' and i != 0 and i != (len(word) - 1) and word[i+1] not in string.punctuation:
                    #    split_sent.append(word[i::])
                    #    start = len(word)
                    #    break
                    else:
                        split_sent.append(word[i])
                        start = i + 1
            if word[start::] != '':
                split_sent.append(word[start::])
    return split_sent

# This version doesn't do anything about punctuation inside words
# and only considers punctuation at the beginning or end of a word

def splitSentenceFast(from_sent):
    split_sent = []
    start = 0
    from_sent = from_sent.split()
    for word in from_sent:
        start = 0
        if '\xe2\x80\x9c' in word:
            word = word.replace('\xe2\x80\x9c','\"')
        if '\xe2\x80\x9d' in word:
            word = word.replace('\xe2\x80\x9d','\"')
        if word in string.punctuation or word in hindi_punctuation:
            split_sent.append(word)
        start = 0
        end = len(word)
        if word[0] in string.punctuation:
            split_sent.append(word[0])
            start = 1
        if word[-1] in string.punctuation:
            split_sent.append(word[start:-1])
            split_sent.append(word[-1])
        else:
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
        if sent[i] in vocab and vocab.index(sent[i]) < 10000:
            new_sent = new_sent + sent[i] + ' ' 
        else:
            new_sent = new_sent + '<unk> '
    return new_sent

print sys.argv

if len(sys.argv) < 3:
    print('python parseHindi.py nameOfSaveDir cutOff')
    exit()

data_dir = '/deep/group/speech/asamar/nlp/data/hindi/source/'
save_dir = sys.argv[1] + '/'
cutOff = int(sys.argv[2])
print('Only sentences less than ' + str(cutOff) + ' will be taken from English.')
print('Only sentences less than ' + str(2*cutOff) + ' will be taken from Hindi.')

if not os.path.exists(save_dir):
    print('Making directory ' + save_dir)
    os.makedirs(save_dir)

print('Loading Original Hindi Dataset')
train_set_size = 273000
data = open(data_dir + 'hindencorp05.plaintext','r')
lines = data.readlines()
count = 0

print('Creating Train and Test Source Sets')
train_f = open(save_dir + 'ptb.train.txt', 'w')
test_f = open(save_dir + 'ptb.test.txt', 'w')
for line in lines:
    if count < train_set_size:
        train_f.write(line)
    else:
        test_f.write(line)
    count = count + 1
data.close()
train_f.close()
test_f.close()

print('Building Vocabulary from Training Set')

punctuation = string.punctuation
eng_vocab = {}
hindi_vocab = {}

data = open(save_dir + 'ptb.train.txt','r')
num_lines = 0
max_eng_length = 0
max_hindi_length = 0
corresponding_line = ''
for line in data:
    orig_line = line.lower().strip()
    split_line = orig_line.split('\t')
    eng_sent = splitSentence(split_line[3])
    hindi_sent = splitSentence(split_line[4])

    if len(eng_sent) < cutOff and len(hindi_sent) < 2*cutOff:
        addToVocab(eng_vocab,eng_sent)
        addToVocab(hindi_vocab,hindi_sent)
        num_lines = num_lines + 1
        if len(eng_sent) > max_eng_length:
            max_eng_length = len(eng_sent)
        if len(hindi_sent) > max_hindi_length:
            max_hindi_length = len(hindi_sent)
            corresponding_line = orig_line

print('Lines below cutoff: ' + str(num_lines))
print('Max English Length: ' + str(max_eng_length))
print('Max Hindi Length: ' + str(max_hindi_length))
print(corresponding_line)
data.close()

print('Sorting Eng Vocab')
eng_vocab = sorted(eng_vocab,key = lambda x: eng_vocab[x])
eng_vocab.reverse()
print('Sorting Hindi Vocab')
hindi_vocab = sorted(hindi_vocab,key = lambda x: hindi_vocab[x])
hindi_vocab.reverse()

print('Parsing Train and Test Set')
for a in [['ptb.train.txt','enc_train.txt','dec_train.txt'],['ptb.test.txt','enc_test.txt','dec_test.txt']]:
    data = open(save_dir + a[0],'r')
    enc_to = open(save_dir + a[1],'w')
    dec_to = open(save_dir + a[2],'w')
    count = 0
    for line in data:
        orig_line = line.lower().strip()
        s = orig_line.split('\t')
        eng_sent = splitSentence(s[3])
        hindi_sent = splitSentence(s[4])
        if len(eng_sent) < cutOff and len(hindi_sent) < 2*cutOff:
            enc_to.write(removeOOV(eng_vocab,eng_sent).strip() + '\n')
            dec_to.write(removeOOV(hindi_vocab,hindi_sent).strip() + '\n')
            count = count + 1
            if (count % 1000 == 0):
                print('Finished parsing ' + str(count))
                sys.stdout.flush()

    data.close()
    enc_to.close()
    dec_to.close()
    print('Total parsed: ' + str(count))
