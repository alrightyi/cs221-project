'''
Main python script for training sentiment analysis model using Ebert's reviews as training data.
Steps:
1) Read txt file containing movie metadata, rating, and last X paragraphs of Ebert's review.
2) Deduce word dictionary generate bi-gram. Using movie metadata and last X 
review paragraph as input X.  Rating as output Y.
3) Split data into training, validation, and test sets
4) Create LSTM RNN model, compile
5) Run training for N epochs
6) Evaluate using test data
'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from read_data import Review
import matplotlib.pyplot as plt
import pickle
import re
import os
import io
import pandas as pd

EMBED_DIM = 64
MAX_FEATURES = 2000
LSTM_OUT = 96
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 50
SPLIT = 0.2
CORPUS = "ebert_last5_2000.txt"
RESULT = "result.txt"
VOCABULARY = "vocab.txt"
MODEL = None
#MODEL = "checkpoints/Sentimentepoch040-loss0.4812-acc0.7683-val_loss0.7965-val_acc0.6151"
KEY = "/reviews/the-crimson-rivers-2001"



def loadReviews():
    f = open('store.pckl', 'rb')
    reviews = pickle.load(f)
    f.close()
    
    return reviews


if __name__ == "__main__":

    reviews = loadReviews()
    examples = RESULT
    vocabulary = VOCABULARY

    if not os.path.isdir('./checkpoints/'):
        os.makedirs('./checkpoints/')

    keys = list(reviews.keys())
    data = [repr(reviews[key]).lower().replace('\n', ' \n ') for key in keys]
    print(data[0])
    Y = [int(reviews[key].rr/3) for key in keys]
    count_Y = Counter(Y)
    print(count_Y)

    # Tokenizer
    max_features = MAX_FEATURES
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(data)
    X = tokenizer.texts_to_sequences(data)
    X = pad_sequences(X)
    print(X.shape)
    Y = pd.get_dummies(Y).values
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = SPLIT, random_state = 4)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)
    
    file_path = "./checkpoints/Sentimentepoch{epoch:03d}-" \
                "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True, period=10)
    #print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    early_stopping = EarlyStopping(monitor='val_acc', patience=20)
    callbacks_list = [checkpoint, early_stopping]
    
    if MODEL == None:
        # Create model
        model = Sequential()
        model.add(Embedding(max_features, EMBED_DIM,input_length = X.shape[1]))
        model.add(SpatialDropout1D(DROPOUT))
        model.add(LSTM(LSTM_OUT, dropout=DROPOUT, recurrent_dropout=DROPOUT))
        model.add(Dense(2,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        print(model.summary())

        # Train
        history = model.fit(X_train, Y_train, 
                            epochs = EPOCHS, callbacks=callbacks_list, 
                            batch_size=BATCH_SIZE, validation_split=SPLIT)
    else:
        model = load_model(MODEL)
        model.summary()
        
    if KEY != None:
        # Test example data:
        text = [repr(reviews[KEY]).lower().replace('\n', ' \n ')]
        print(KEY)
        txt = tokenizer.texts_to_sequences(text)
        txt = pad_sequences(txt, maxlen=X.shape[1], dtype='int32', value=0)
        sentiment = model.predict(txt,batch_size=1)
        print(sentiment)
        print("Predicted with Ebert:  ", "thumbs-up" if np.argmax(sentiment[0]) >= 1 else "thumbs-down")
        print("Ebert Actual:  ", "thumbs-up " if reviews[KEY].rr/3 >= 1 else "thumbs-down ", reviews[KEY].rr)

        test_example = "integrated_example.txt"
        with io.open(test_example, encoding='utf-8') as f:
            text = [f.read().lower().replace('\n', ' \n ')]
            txt = tokenizer.texts_to_sequences(text)
            txt = pad_sequences(txt, maxlen=X.shape[1], dtype='int32', value=0)
            sentiment = model.predict(txt,batch_size=1)
            print(sentiment)
            print("Predicted with WG:  ", "thumbs-up" if np.argmax(sentiment[0]) >= 1 else "thumbs-down")
            print("Ebert Actual:  ", "thumbs-up " if reviews[KEY].rr/3 >= 1 else "thumbs-down ", reviews[KEY].rr)
    
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy dropout:%.2f max_feat:%i embed_dim:%i LSTM_out:%i split:%.2f' 
              % (DROPOUT,MAX_FEATURES,EMBED_DIM,LSTM_OUT,SPLIT))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('SA_acc_d%.0fm%ie%il%is%.0f' 
              % (DROPOUT*10,MAX_FEATURES,EMBED_DIM,LSTM_OUT,SPLIT*10))
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss dropout:%.2f max_feat:%i embed_dim:%i LSTM_out:%i split:%.2f' 
              % (DROPOUT,MAX_FEATURES,EMBED_DIM,LSTM_OUT,SPLIT))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('SA_loss_d%.0fm%ie%il%is%.0f' 
              % (DROPOUT*10,MAX_FEATURES,EMBED_DIM,LSTM_OUT,SPLIT*10))

    # Evaluate
    correct = 0
    score, acc = model.evaluate(X_test, Y_test, batch_size = BATCH_SIZE)
    print("test score: %.2f acc: %.2f" % (score,acc))
    
    f = open('keys.pckl', 'rb')
    shuffled_keys = pickle.load(f)
    f.close()

    for key in shuffled_keys[-10:-1]:
        text = [repr(reviews[key]).lower().replace('\n', ' \n ')]
        print(key)
        txt = tokenizer.texts_to_sequences(text)
        txt = pad_sequences(txt, maxlen=X.shape[1], dtype='int32', value=0)
        sentiment = model.predict(txt,batch_size=1)
        print(sentiment)
        print("Predicted:  ", "thumbs-up" if np.argmax(sentiment[0]) >= 1 else "thumbs-down")
        print("Actual:  ", "thumbs-up " if reviews[key].rr/3 >= 1 else "thumbs-down ", reviews[key].rr)
