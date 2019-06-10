'''
Main python script for training word generation model using Ebert's reviews as training data.
Steps:
1) Read txt file containing movie meta data, rating, and last X paragraphs of Ebert's review.
2) Deduce word dictionary generate bi-gram.  Word as input, next word 
(including line break) as output.
3) Split data into training, validation, and test sets
4) Create LSTM RNN model, compile
5) Run training for N epochs
6) Evaluate using test data
'''

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, Dropout, Activation
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from read_data import Review
import matplotlib.pyplot as plt
import re
import io
import os
import codecs
import pickle
import random

MIN_WORD_FREQUENCY = 5
SEQUENCE_LEN = 100
STEP = 4
DROPOUT = 0.6
BATCH_SIZE = 32
SPLIT=0.2
EPOCHS = 50
#CORPUS = "ebert_last5_2000.txt"
CORPUS = "roger_ebert_last5.txt"
RESULT = "result_5_epoch10.txt"
VOCABULARY = "vocab.txt"
MODEL = None
#MODEL = "checkpoints/LSTM-epoch010-words18707-sequence140-minfreq5-loss5.6837-acc0.0531-val_loss8.5211-val_acc0.0547"



def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=2):
    # shuffle at unison
    print('Shuffling sentences')

    tmp_sentences = []
    tmp_next_word = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)


# Data generator for fit and evaluate
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN, len(words)), dtype=np.bool)
        y = np.zeros((batch_size, len(words)), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t, word_indices[w]] = 1
            y[i, word_indices[next_word_list[index % len(sentence_list)]]] = 1
            index = index + 1
        yield x, y


def print_vocabulary(words_file_path, words_set):
    words_file = codecs.open(words_file_path, 'w', encoding='utf8')
    for w in words_set:
        if w != "\n":
            words_file.write(w+"\n")
        else:
            words_file.write(w)
    words_file.close()


def get_model(dropout=0.2):
    print('Build model...')
    model = Sequential()
    model.add(Bidirectional(LSTM(128), input_shape=(SEQUENCE_LEN, len(words))))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    return model


# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

    # Randomly pick a seed sequence (of movie metadata)
    #seed_index = np.random.randint(len(sentences+sentences_test))
    #seed = (sentences+sentences_test)[seed_index]
    review = reviews[shuffled_keys[0]]
    txt = ''
    alt = 0
    for detail in review.details:
        detail = detail.lower()
        detail = re.sub('[^a-zA-z0-9\s]', '', detail)
        txt += detail+('\n' if alt % 2 == 1 else ' ')
        alt += 1
        
    seed = txt.replace('\n', ' \n ')
    #print(seed)
    text_in_words = [w for w in seed.split(' ') if w.strip() != '' or w == '\n']
    #print(text_in_words)
    #print("seed length: ", len(text_in_words))

    #for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
    for diversity in [0.5, 1.0]:
        sentence = text_in_words
        examples_file.write('----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
        examples_file.write(' '.join(sentence))

        for i in range(SEQUENCE_LEN-len(sentence)):
            x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))
            for t, word in enumerate(sentence):
                if word in word_indices.keys():
                    x_pred[0, t, word_indices[word]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            examples_file.write(" "+next_word)
        examples_file.write('\n')
    examples_file.write('='*80 + '\n')
    examples_file.flush()


if __name__ == "__main__":

    corpus = CORPUS
    examples = RESULT
    vocabulary = VOCABULARY

    if not os.path.isdir('./checkpoints/'):
        os.makedirs('./checkpoints/')

    with io.open(corpus, encoding='utf-8') as f:
        text = f.read().lower().replace('\n', ' \n ')
    print('Corpus length in characters:', len(text))

    text_in_words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']
    print('Corpus length in words:', len(text_in_words))

    # Calculate word frequency
    word_freq = {}
    for word in text_in_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    ignored_words = set()
    for k, v in word_freq.items():
        if word_freq[k] < MIN_WORD_FREQUENCY:
            ignored_words.add(k)
            
    print("Ignored words: ", list(ignored_words)[0:100])

    words = set(text_in_words)
    print('Unique words before ignoring:', len(words))
    print('Ignoring words with frequency <', MIN_WORD_FREQUENCY)
    words = sorted(set(words) - ignored_words)
    print('Unique words after ignoring:', len(words))
    print_vocabulary(vocabulary, words)

    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))

    # cut the text in semi-redundant sequences of SEQUENCE_LEN words
    sentences = []
    next_words = []
    ignored = 0
    for i in range(0, len(text_in_words) - SEQUENCE_LEN, STEP):
        # Only add the sequences where no word is in ignored_words
        if len(set(text_in_words[i: i+SEQUENCE_LEN+1]).intersection(ignored_words)) == 0:
            sentences.append(text_in_words[i: i + SEQUENCE_LEN])
            next_words.append(text_in_words[i + SEQUENCE_LEN])
        else:
            ignored = ignored + 1
    print('Ignored sequences:', ignored)
    print('Remaining sequences:', len(sentences))

    # x, y, x_test, y_test
    (sentences, next_words), (sentences_test, next_words_test) = shuffle_and_split_training_set(
        sentences, next_words, percentage_test=SPLIT*100
    )
    
    examples_file = open(examples, "w")
    f = open('keys.pckl', 'rb')
    shuffled_keys = pickle.load(f)
    f.close()
    
    f = open('store.pckl', 'rb')
    reviews = pickle.load(f)
    f.close()
    
    file_path = "./checkpoints/LSTM-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
                "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % \
                (len(words), SEQUENCE_LEN, MIN_WORD_FREQUENCY)

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True, period=10)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    early_stopping = EarlyStopping(monitor='val_acc', patience=20)
    callbacks_list = [checkpoint, print_callback, early_stopping]

    if MODEL == None:
        model = get_model(DROPOUT)
        model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
        model.summary()
        history = model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
                                      steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
                                      epochs=EPOCHS,
                                      callbacks=callbacks_list,
                                      validation_data=generator(sentences_test, next_words_test, BATCH_SIZE),
                                      validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1)
        
    else:
        model = load_model(MODEL)
        model.summary()
        
        history = model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
                                      steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
                                      epochs=EPOCHS,
                                      callbacks=callbacks_list,
                                      validation_data=generator(sentences_test, next_words_test, BATCH_SIZE),
                                      validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1)
        
    on_epoch_end(10, None)
    
    #keys = list(reviews.keys())
    #length = 2000
    #random.Random(4).shuffle(keys)
    #f = open('keys.pckl', 'wb')
    #pickle.dump(keys, f)
    #f.close()

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy dropout:%.2f min_word:%i seq_len:%i step:%i split:%.2f' 
              % (DROPOUT,MIN_WORD_FREQUENCY,SEQUENCE_LEN,STEP,SPLIT))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('WG_acc_d%.0fm%is%ist%is%.0f' 
              % (DROPOUT*10,MIN_WORD_FREQUENCY,SEQUENCE_LEN,STEP,SPLIT*10))
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss dropout:%.2f min_word:%i seq_len:%i step:%i split:%.2f' 
              % (DROPOUT,MIN_WORD_FREQUENCY,SEQUENCE_LEN,STEP,SPLIT))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('WG_loss_d%.0fm%is%ist%is%.0f' 
              % (DROPOUT*10,MIN_WORD_FREQUENCY,SEQUENCE_LEN,STEP,SPLIT*10))
