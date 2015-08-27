from random import shuffle
import math
import string

for a in [['ptb.train.txt','enc_train.txt','dec_train.txt'],['ptb.test.txt','enc_test.txt','dec_test.txt']]:
    data_dir = '/deep/group/speech/asamar/nlp/data/penn/'
    save_dir = '/deep/group/speech/asamar/nlp/data/pennShuf/'
    data = open(data_dir + a[0],'r')
    enc_to = open(save_dir + a[1],'w')
    dec_to = open(data_path_to + a[2],'w')
    
    for line in data_from:
        s = line.split()
        part = {}
        shuf = {}
        splitind = int(math.floor(len(s)/3))
        
        shuf[0] = s[0:splitind]
        shuf[1] = s[splitind:2*splitind]
        shuf[2] = s[2*splitind:len(s)]
        
        part[0] = s[0:splitind]
        part[1] = s[splitind:2*splitind]
        part[2] = s[2*splitind:len(s)]
        
        for j in range(0,3):
            for i in range(0,3):
                shuffle(shuf[j])
                enc_to.write(' '.join(shuf[j]))
                enc_to.write('<EOS>\n')
                dec_to.write(' '.join(part[j]))
                dec_to.write('<EOS>\n')


    data.close()
    enc_to.close()
    dec_to.close()
