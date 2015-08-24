import string

eng_f = open('txt/enc/enc.txt','w')
hindi_f = open('txt/dec/dec.txt','w')
punctuation = [',','.','!','?','|']
with open('source/hindencorp05.plaintext') as f:
    for line in f:
        s = line.split('\t')
        eng_sent = string.strip(s[3])

        for i in range(0,len(eng_sent)):
            if eng_sent[i] in punctuation:
                sp = eng_sent.split(eng_sent[i])
                eng_sent = ' '.join(sp)

        hindi_sent = string.strip(s[4])

        for i in range(0,len(hindi_sent)):
            if hindi_sent[i] in punctuation:
                sp = hindi_sent.split(hindi_sent[i])
                hindi_sent = ' '.join(sp)

        eng_f.write(eng_sent)
        eng_f.write('\n')

        hindi_f.write(hindi_sent)
        hindi_f.write('\n')


