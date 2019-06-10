# cs221-project
Roger Ebert review

This repository contains 3 key python scripts:

1) read_data.py

This file retrieves movie metadata (incl. title, year, cast & crew, genres, ratings, as well as the review text by Roger Ebert) and stores them in a pickle file.
It also processes them  distills the word corpus into a txt file.
It also calculates some basic statistics about the data.

2) word_generation.py

This file reads the stored reviews from pickle file and processes them into X & Y datasets.
It also creates an LSTM model and runs the training.

LSTM model summary:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional_1 (Bidirection (None, 256)               19288064  
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 18707)             4807699   
_________________________________________________________________
activation_1 (Activation)    (None, 18707)             0         
=================================================================
Total params: 24,095,763
Trainable params: 24,095,763
Non-trainable params: 0

Hyper parameters to tweak are:
MIN WORD FREQUENCY: min number of times a word has to appear in the corpus to be added to the training set. (default set to 5) This limits unnecessary calculations on rarely used words.
SEQUENCE LEN: number of words constructed into a sequence and fed into the model as X (default set to 140) as this is the average size of a review plus metadata.
STEP: sliding window size to the next word sequence (default set to 10)
DROPOUT: dropout rate (default set to 0.5)
BATCH SIZE: batch size (default set to 32)
SPLIT: train/val/test split (default set to 0.2, meaning 0.6/0.2/0.2 split)
EPOCHS: number of training cycles (defaults to 50)
EARLY STOPPING: number of training cycles without improvement before stopping (defaults to 10)

3) sentiment_analysis.py

This file reads the stored reviews from pickle and processes them into X & Y datasets.
It then creates an LSTM model for training, summary:
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 454, 32)           32000     
_________________________________________________________________
spatial_dropout1d_1 (Spatial (None, 454, 32)           0         
_________________________________________________________________
lstm_1 (LSTM)                (None, 48)                15552     
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 98        
=================================================================
Total params: 47,650
Trainable params: 47,650
Non-trainable params: 0

Hyper parameters to tweak are:
MAX FEATURES: max number of unique words in the vocabulary. Use to control the number of features in the model (default is 2000)
EMBED DIM: dimension of the dense embedding from the first layer of the model. (default is 64)
LSTM OUT: dimension of the LSTM layer output (default is 96)
DROPOUT: dropout rate (default set to 0.5)
BATCH SIZE: batch size (default set to 32)
SPLIT: train/val/test split (default set to 0.2, meaning 0.6/0.2/0.2 split)
EPOCHS: number of training cycles (defaults to 50)
EARLY STOPPING: number of training cycles without improvement before stopping (defaults to 10)

Sample results from the tests are here:

result_5_epoch10.txt

Word generation results from each epoch run from word_generation.

ebert_sa_results.txt

Sentiment analysis results.
