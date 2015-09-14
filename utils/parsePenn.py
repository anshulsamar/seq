from random import shuffle
import random
import math
import string
import sys
import os

print sys.argv

if len(sys.argv) == 1:
    print('python parsePenn.py nameOfSaveDir [-shuf]')
    exit()

data_dir = '/deep/group/speech/asamar/nlp/data/penn/pennSource/'
save_dir = sys.argv[1] + '/'

if not os.path.exists(save_dir):
    print('Making directory ' + save_dir)
    os.makedirs(save_dir)

for a in [['ptb.train.txt','enc_train.txt','dec_train.txt'],['ptb.test.txt','enc_test.txt','dec_test.txt']]:
    data = open(data_dir + a[0],'r')
    enc_to = open(save_dir + a[1],'w')
    dec_to = open(save_dir + a[2],'w')
    num_lines = 0
    for line in data:
        orig_line = line.lower().strip()
        lines = []
        if '-split' in sys.argv:
            s = orig_line.split()
            while True:
                new_line = []
                i = 0
                s_len = len(s)
                while i < 5 and i < s_len:
                    new_line.append(s[0])
                    s.remove(s[0])
                    i = i + 1
                lines.append(' '.join(new_line))
                if i != 5 or len(s) == 0:
                    break
        else:
            lines[orig_line]

        for line in lines:
            if '-shufWord' in sys.argv:
                s = line.split()
                if (len(s) > 1):
                    index = random.randint(0,len(s) - 2)
                    tmp = s[index]
                    s[index] = s[index+1]
                    s[index+1] = tmp
                enc_to.write(' '.join(s))
                enc_to.write(' <eos>\n')
                dec_to.write(line + ' <eos>\n')

            elif '-shufBlocks' in sys.argv:
                
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
                        enc_to.write(' <eos>\n')
                        dec_to.write(' '.join(part[j]))
                        dec_to.write(' <eos>\n')
            else:
                enc_to.write(line + '\n')
                dec_to.write(line + ' <eos>\n')
        

    data.close()
    enc_to.close()
    dec_to.close()
