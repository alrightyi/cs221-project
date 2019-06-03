import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LambdaCallback, EarlyStopping
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from read_data import Review
import pickle
import re
import os
import io
import pandas as pd


DROPOUT = 0.6
BATCH_SIZE = 32
EPOCHS = 100
CORPUS = "ebert_last5_2000.txt"
RESULT = "result.txt"
VOCABULARY = "vocab.txt"

EMBED_DIM = 64
LSTM_OUT = 128

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
    Y = [reviews[key].rr*2 for key in keys]

    # Tokenizer
    max_features = 2000
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(data)
    X = tokenizer.texts_to_sequences(data)
    X = pad_sequences(X)
    print(X.shape)
    
    # Create model
    model = Sequential()
    model.add(Embedding(max_features, EMBED_DIM,input_length = X.shape[1]))
    model.add(SpatialDropout1D(DROPOUT))
    model.add(LSTM(LSTM_OUT, dropout=DROPOUT, recurrent_dropout=DROPOUT))
    model.add(Dense(9,activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    
    Y = pd.get_dummies(Y).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state = 42)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)
    
    file_path = "./checkpoints/Sentimentepoch{epoch:03d}-" \
                "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}"

    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=True, period=10)
    #print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    #early_stopping = EarlyStopping(monitor='val_acc', patience=10)
    callbacks_list = [checkpoint]
    
    # Train
    model.fit(X_train, Y_train, epochs = EPOCHS, callbacks=callbacks_list, batch_size=BATCH_SIZE, validation_split=0.1, verbose=2)
    
    # Test
    validation_size = 200

    # Verify
    #X_validate = X_test[-validation_size:]
    #Y_validate = Y_test[-validation_size:]
    #X_test = X_test[:-validation_size]
    #Y_test = Y_test[:-validation_size]
    #score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = BATCH_SIZE)
    #print("score: %.2f" % (score))
    #print("acc: %.2f" % (acc))

    # Evaluate
    correct = 0
    for x in range(len(X_test)):

        result = model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1)[0]

        if np.argmax(result) == np.argmax(Y_test[x]):
            correct += 1


    print("acc", correct/len(X_test)*100, "%")
    
    
    f = open('keys.pckl', 'rb')
    shuffled_keys = pickle.load(f)
    f.close()

    #test_example = "example2.txt"
    #with io.open(test_example, encoding='utf-8') as f:
    #    text = f.read().lower().replace('\n', ' \n ')
    #vectorizing the review by the pre-fitted tokenizer instance
    for key in shuffled_keys[2000:2010]:
        body = reviews[key].body
        if len(body) > 100:
            continue
        text = ''
        for b in body[-6:-1]:
            if b == '\'Advertisement\'' or 'googletag.cmd.push' in body:
                continue
            b = b.lower()
            b = re.sub('[^a-zA-z0-9\s]', '', b)
            text += b+' '
        text += '\n'
        print(text)
        txt = tokenizer.texts_to_sequences(text)
        #text = pad_sequences(text)
        txt = pad_sequences(txt, maxlen=X.shape[1], dtype='int32', value=0)
        sentiment = model.predict(txt,batch_size=1)[0]
        print("Predicted:  ", np.argmax(sentiment)/2, " stars")
        print("Actual:  ", reviews[key].rr, " stars")
    