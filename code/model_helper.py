# coding: utf8

import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential, Model
from keras.layers import BatchNormalization, Bidirectional, CuDNNGRU, GRU, Embedding, LSTM, Dense, Activation, Conv1D, MaxPooling1D, Dropout, GlobalMaxPooling1D, Input, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import *
from AttentionWithContext import AttentionWithContext
from xgboost import XGBClassifier
import csv
import os
from keras import backend as K
import matplotlib.pyplot as plt
from collections import Counter
import operator

def logistic_regression(x_train,y_train,x_val,y_val, C = None):
    logreg = LogisticRegression()
    clf = logreg.fit(x_train, np.argmax(y_train,axis=1))
    y_val_pred = clf.predict(x_val)
    acc = accuracy_score(np.argmax(y_val,axis=1), y_val_pred)
    return(acc)

def SVM(x_train,y_train,x_val,y_val):
    classifier = LinearSVC()
    classifier.fit(x_train, np.argmax(y_train,axis=1))
    y_val_pred = classifier.predict(x_val)
    acc = accuracy_score(np.argmax(y_val,axis=1), y_val_pred)
    return(acc)

def XGB(x_train,y_train,x_val,y_val):
    classifier = XGBClassifier()
    classifier.fit(x_train, np.argmax(y_train,axis=1))
    y_val_pred = classifier.predict(x_val)
    acc = accuracy_score(np.argmax(y_val,axis=1), y_val_pred)
    return(acc)


def multi_perceptron(x_train):
    n_classes  = 51
    model = Sequential()
    model.add(BatchNormalization(input_shape=tuple([x_train.shape[1]])))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes,activation='sigmoid'))
    return(model)


def multi_CNN(embeddings, maxlen = 50, hidden = 128, filter_sizes = [3,4,5], use_pretrained = True, train_embedding = True):
    n_classes  = 51
    ints = Input(shape=(None,))
    if use_pretrained :
        wv = Embedding(output_dim=np.shape(embeddings)[1], input_dim=np.shape(embeddings)[0], input_length = maxlen, weights=[embeddings], trainable = train_embedding)(ints)
    else :
        wv = Embedding(output_dim=np.shape(embeddings)[1], input_dim=np.shape(embeddings)[0], input_length = maxlen)(ints)
    branch_outputs = []
    for idx in range(3):
        branch_outputs.append(Dropout(0.5)(GlobalMaxPooling1D()(Conv1D(filters=hidden,
                                                           kernel_size=filter_sizes[idx],
                                                           activation='relu')(wv))))
    concat = Concatenate()(branch_outputs)
    preds = Dense(n_classes, activation='softmax')(concat)
    model = Model(ints,preds)
    return(model)

def visualize_silency_map(model,sequence,index2word) :
    saliency_input = model.layers[3].input # before convolution
    saliency_output = model.layers[10].output # class score

    gradients = model.optimizer.get_gradients(saliency_output,saliency_input)
    compute_gradients = K.function(inputs=[model.input, K.learning_phase()],outputs=gradients)

    matrix = compute_gradients([sequence,0])
    tokens = [index2word[elt] for elt in sequence[0] if elt!=0]

    matrix = matrix[0][0,:,:]
    to_plot=matrix

    fig, ax = plt.subplots()
    heatmap = ax.imshow(to_plot, cmap=plt.cm.Blues, interpolation='nearest', aspect='auto')
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.tick_params(axis='y', which='major', labelsize=32*10/len(tokens))
    fig.colorbar(heatmap)
    fig.set_size_inches(11,7)
    fig.savefig('../Visualize/saliency_map.pdf',bbox_inches='tight')
    fig.show()

def bidir_gru(my_seq,n_units,is_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    enables CUDA acceleration on GPU
    # regardless of whether training is done on GPU, model can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if is_GPU:
        return Bidirectional(CuDNNGRU(units=n_units,
                                      return_sequences=True),
                             merge_mode='concat', weights=None)(my_seq)
    else:
        return Bidirectional(GRU(units=n_units,
                                 activation='tanh',
                                 dropout=0.0,
                                 recurrent_dropout=0.0,
                                 implementation=1,
                                 return_sequences=True,
                                 reset_after=True,
                                 recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)

def bidir_lstm(my_seq,n_units):
    return Bidirectional(LSTM(units=n_units,
                                 activation='tanh',
                                 dropout=0.0,
                                 recurrent_dropout=0.0,
                                 implementation=1,
                                 return_sequences=True,
                                 reset_after=True,
                                 recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)


def HAN(x_train, embeddings, n_units = 32, use_pretrained=True, train_embedding = True, is_GPU = False):
    n_classes  = 51
    sent_ints = Input(shape=(x_train.shape[1],))
    sent_wv = Embedding(input_dim=embeddings.shape[0],
                        output_dim=embeddings.shape[1],
                        weights=[embeddings],
                        input_length=x_train.shape[1],
                        trainable = train_embedding,
                        )(sent_ints)
    sent_wv_dr = Dropout(0.5)(sent_wv)
    sent_wa = bidir_lstm(sent_wv_dr,n_units)
    sent_att_vec,word_att_coeffs = AttentionWithContext(return_coefficients=True)(sent_wa)
    sent_att_vec_dr = Dropout(0.5)(sent_att_vec)
    preds = Dense(units=n_classes,activation='sigmoid')(sent_att_vec_dr)
    model = Model(sent_ints,preds)
    print(model.summary)
    return(model)


def train(model, x_train, y_train, x_val, y_val, optimizer = 'adam', batch_size = 64, nb_epochs = 50, my_patience = 5, save_model = False, model_name = None, monitor='val_acc', early_stopping = False):
    loss_classif     =  'categorical_crossentropy'
    metrics_classif  =  ['accuracy']
    model.compile(loss=loss_classif,
                  optimizer=optimizer,
                  metrics=metrics_classif)

    if early_stopping :
        early_stopping = EarlyStopping(monitor=monitor, # go through epochs as long as accuracy on training data set increases
                                       patience=my_patience,
                                       mode='max')

        # save model corresponding to best epoch
        checkpointer = ModelCheckpoint(filepath= 'model',
                                       verbose=1,
                                       save_best_only=True)

        history = model.fit(x=x_train, y=y_train, batch_size=batch_size,
                            epochs=nb_epochs, callbacks=[early_stopping,checkpointer],
                            validation_data = (x_val, y_val))
    else :
        history = model.fit(x=x_train, y=y_train, batch_size=batch_size,
                            epochs=nb_epochs,
                            validation_data = (x_val, y_val))

    if save_model :
        model_json = model.to_json()
        if model_name is not None :
            fname = model_name
        else :
            model_name = "model"
        with open(model_name + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.h5")

def get_wrong_classification(model,x_val,y_val, word2index, train_sent, idx_val, y_train):
    index2word = {index : word for word,index in word2index.items()}
    y_prob = model.predict(y_val)
    y_classes_pred = y_prob.argmax(axis=-1)
    y_classes_val = y_val.argmax(axis=-1)
    y_train_classes = y_train.argmax(axis=-1)
    y_train_count = dict(Counter(y_train_classes))
    wrong_words = []
    wrong_class = []
    for i in range(len(idx_val)):
        if y_classes_pred[i] != y_classes_val[i] :
            seq = x_val[i]
            idx = idx_val[i]
            sent = train_sent[idx]
            sent_cleaned = [index2word[el] for el in seq if el > 0]
            wrong_words.append(sent)
            wrong_class.append(y_classes_val[i])
    print(wrong_class)
    all_tokens = [token for sublist in wrong_words for token in sublist]
    word_counts = dict(Counter(all_tokens))
    sorted_word_counts = sorted(list(word_counts.items()), key=operator.itemgetter(1), reverse=True)
    class_counts = dict(Counter(wrong_class))
    for cl in class_counts.keys():
        class_counts[cl] = class_counts[cl]/y_train_count[cl]
    sorted_class_prop = sorted(list(class_counts.items()), key=operator.itemgetter(1), reverse=True)
    for i in range(len(sorted_class_prop)) :
        sorted_class_prop[i] = (sorted_class_prop[i][0],sorted_class_prop[i][1],y_train_count[sorted_class_prop[i][0]])
    return(sorted_word_counts[:100],sorted_class_prop)

def get_wrong_classification_from_id (model, id, x_val, y_val, word2index, idx_val):
    index2word = {index : word for word,index in word2index.items()}
    y_prob = model.predict(y_val)
    y_classes_pred = y_prob.argmax(axis=-1)
    y_classes_val = y_val.argmax(axis=-1)
    wrong_sent = []
    for i in range(len(idx_val)):
        if y_classes_val[i] == id and y_classes_pred[i] != y_classes_val[i] :
            seq = x_val[i]
            idx = idx_val[i]
            sent_cleaned = [index2word[el] for el in seq if el > 0]
            wrong_sent.append(sent_cleaned)
    return(wrong_sent)

def generate_test_file(model,x_test, PATH_TO_DATA, output_file = 'output_test.csv'):
    y_prob = model.predict(x_test)
    y_classes = y_prob.argmax(axis=-1)
    y_id = []
    test_file = open(os.path.join(PATH_TO_DATA, 'input_test.csv'),'r')
    fl = test_file.readlines()
    for line in fl[1:] :
        words = text_to_word_sequence(line)
        y_id.append(int(words[0]))
    rows = zip(y_id,y_classes)
    with open(output_file, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(['ID','intention'])
        for row in rows:
            writer.writerow(row)
