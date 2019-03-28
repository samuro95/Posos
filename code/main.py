
# coding: utf8
import numpy as np
import os
from data_helper import load_corrected_sentences, build_dictionary, load_data, clean_sequences, encode_sequences, create_train_val_sets, get_med_list, save_cleaned_data
from model_helper import multi_perceptron, get_wrong_classification_from_id, visualize_silency_map, get_wrong_classification, SVM, XGB, logistic_regression, multi_CNN, HAN, train, generate_test_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from gensim.models import KeyedVectors
from OOV import OOV
from sklearn import preprocessing
from Embedding import create_embedding, test_word, get_mean_emdedding, visualize_embedding
from data_augmentation import augment_data_random_walk, replace_embedding

data_path = '../data'
embedding_bin_file = '../data/embeddings/frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin'
#embedding_path = '../data/embeddings/polyglot-fr.pkl'

#### Load initial data
_,y_train,_ = load_data(data_path)

#### Load a correction I have done thanks to "clean_sequences" function
train_sent_corrected, test_sent = load_corrected_sentences(data_path)

#### Split training data iun training and validation sets
val_sent, y_val, train_sent, y_train, idxs_select_val = create_train_val_sets(train_sent_corrected, y_train, validation_ratio = 0.2)

print('data loaded')

####  Data augmentation

### with random walk method
#train_sent, y_train = augment_data_random_walk(train_sent, y_train)
#print('random walk augmentation done')

#### Encoding word --> integers
train_seq_encoded,test_seq_encoded,val_seq_encoded, y_train,y_val, word2index = encode_sequences(train_sent,test_sent,val_sent,y_train,y_val,maxlen = 50)

#### Load pre-trained embedding fitted to current dictionary
embeddings,word2embedding = create_embedding(bin_file, word2index)
#visualize_embedding(train_sent_corrected+test_sent_corrected,word2index,embeddings)

#### Training

# Design the model
model = multi_CNN(embeddings, train_embedding = False)
#Train the model
train(model, train_seq_encoded, y_train, val_seq_encoded, y_val, optimizer = 'sgd', batch_size = 64, nb_epochs = 300)
