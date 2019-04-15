import os
import re
import copy

import numpy as np


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential, load_model


g_model = None
rootdir = '../../data'


def get_data_from_files(rootdir):
    data = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if ('.capp' in file):
                textfile = subdir+'/'+file
                with open(textfile,'r') as f :
                    lines = f.readlines()
                train = []
                test = []
                for sent in lines :
                    if '*CHI:' in sent :
                        sent = re.sub('\*[A-Z]+: ', '', sent)
                        test.append(sent)
                    else :
                        sent = re.sub('\*[A-Z]+: ', '', sent)
                        train.append(sent)
                data.append((file,train,test))

    return data


def prepare_seq(seq, maxlen):
    # Pads seq and slides windows
    x = []
    y = []
    for i, w in enumerate(seq):
        x_padded = pad_sequences([seq[:i]],
                                 maxlen=maxlen - 1,
                                 padding='pre')[0]  # Pads before each sequence
        x.append(x_padded)
        y.append(w)
    return x, y



def prepare_train_set(seqs) :
    maxlen = max([len(seq) for seq in seqs])
    x = []
    y = []

    # Slide windows over each sentence
    for seq in seqs:
        x_windows, y_windows = prepare_seq(seq, maxlen)
        x += x_windows
        y += y_windows

    x = np.array(x)
    y = np.eye(len(vocab))[(np.array(y) - 1)]  # One hot encoding

    return maxlen,x,y

def train_model(vocab_size,maxlen,x,y, output_size, hidden_size, epochs):
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size+1,  # vocabulary size. Adding an
                                                   # extra element for <PAD> word
                        output_dim = output_size,  # size of embeddings
                        input_length = maxlen-1))  # length of the padded sequences
    model.add(LSTM(hidden_size))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile('rmsprop', 'categorical_crossentropy')

    # Train network
    model.fit(x, y, epochs=epochs)
    return model

def get_seq_prob(word, context, maxlen):
    global g_model
    sub_seq = list(context)
    sub_seq.append(word)
    x, y = prepare_seq(sub_seq, maxlen)
    x = np.array(x)
    y = np.array(y) - 1  # The word <PAD> does not have a class

    p_pred = g_model.predict(x)
    log_p_seq = 0

    for i, prob in enumerate(p_pred):
        prob_word = prob[y[i]]
        log_p_seq += np.log(prob_word)

    return np.exp(log_p_seq)


def eval_production(seq, maxlen):
    result = 0
    vocab = list(seq)
    context = []

    while vocab != [] :
        (next_word, max_prob) = max([(v, get_seq_prob(v, context, maxlen)) for v in vocab], key=lambda prob:prob[1])
        context.append(next_word)
        vocab.remove(next_word)

    if context == seq :
        result = 1

    return result


def get_seq_bylength(seqs) :
    seqs_bylength = dict()
    for seq in seqs :
        seqlen = len(seq)
        if seqlen > 1:
            if seqlen in seqs_bylength:
                seqs_bylength[seqlen].append(seq)
            else :
                seqs_bylength[seqlen] = [seq]
    return seqs_bylength



def get_performance_bylength(seqs_bylength, maxlen) :
    results_bylength = dict()
    for length,seqs in seqs_bylength.items():
        results_bylength[length] = [0, len(seqs)]
        print(str(length))
        for seq in seqs:
            results_bylength[length][0] += eval_production(seq, maxlen)

    return results_bylength



data = get_data_from_files(rootdir)
for file,train,test in data:
    print(file+'\n')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train)

    vocab = tokenizer.word_index
    vocab_size = len(vocab)

    train_seqs = tokenizer.texts_to_sequences(train)
    test_seqs = tokenizer.texts_to_sequences(test)

    maxlen,x,y = prepare_train_set(train_seqs)
    print('vocab_size = '+str(vocab_size))
    print('maxlen = '+str(maxlen))
    print('TRAINING MODEL...\n')
    g_model = train_model(vocab_size,maxlen,x,y, output_size=100, hidden_size=500, epochs=10)
    g_model.save(str('../../models/'+file.split('.capp')[0]+'_model.h5'))
    #g_model = load_model('Adam_model.h5')
    seqs_bylength = get_seq_bylength(test_seqs)
    print('CALCULATING PRODUCTION PERFORMANCE...\n')
    results = get_performance_bylength(seqs_bylength, maxlen)

    with open(rootdir+'results/lstm-baseline/prod_results/'+file.split('.capp')[0]+'.prod_result.csv','w') as f :
        f.write("iter,utterance_length,nb_utterances,produced,production_score"+'\n')
        for length in results:
            f.write('1,'+str(length)+','+
                            str(results[length][1])+','+
                            str(results[length][0])+','+
                            str(results[length][0]/results[length][1])+'\n')
    del g_model
