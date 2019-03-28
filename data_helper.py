# coding: utf8

import numpy as np
import os
from keras.preprocessing.text import *
from keras.utils import *
from keras.preprocessing.sequence import *
import nltk
from nltk.corpus import stopwords
# requires nltk 3.2.1
from nltk import pos_tag
from nltk.stem.snowball import SnowballStemmer
import unicodedata
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import csv
from collections import defaultdict


def get_QUEARO_annorations():
    categories = defaultdict(set)
    for root, dirs, files in os.walk("../data/QUAERO_FrenchMed/corpus/"):
        for filename in files :
            if filename[-3:] == 'ann' :
                filen = os.path.join(root, filename)
                with open(filen,'r') as file :
                    for line in file.readlines() :
                        if line[0] == 'T' :
                            line = text_to_word_sequence(line, filters='\t\n')
                            category = line[1]
                            word = line[4]
                            word = unicodedata.normalize('NFD', word).encode('ascii', 'ignore').decode('UTF-8')
                            categories[category].add(word)
    to_remove_anat = ['douleurs','lait']
    for el in to_remove_anat :
        categories['anat'].remove(el)
    return(categories)


def get_med_list(med_path='../data/meds.txt'):
    med_list=[]
    med_file = open(med_path,'r')
    fl = med_file.readlines()
    for line in fl :
        med_list.append(line[:-1].lower())
    return(med_list)

def load_data(data_path, language = 'fr'):
      'return the train, test, val arrays of sentences and classes'

      train_sent = []
      y_train = []
      test_sent = []

      if language == 'en' :
          train_qu_file = open(os.path.join(data_path, 'input_train_trans.csv'),'r')
          train_sent_en = []
      else :
          train_qu_file = open(os.path.join(data_path, 'input_train.csv'),'r')

      train_cat_file = open(os.path.join(data_path, 'output_train.csv'),'r')
      test_file = open(os.path.join(data_path, 'input_test.csv'),'r')

      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n´’€' + "'" +'1234567890'

      fl = train_cat_file.readlines()
      for line in fl[1:] :
          words = text_to_word_sequence(line)
          y_train.append(int(words[1]))

      fl = train_qu_file.readlines()
      for line in fl[1:] :
          words = text_to_word_sequence(line, filters=filters)
          train_sent.append(words)

      fl = test_file.readlines()
      for line in fl[1:] :
          words = text_to_word_sequence(line, filters=filters)
          test_sent.append(words)

      return(train_sent,y_train,test_sent)

def load_corrected_sentences(data_path):
    train_sent = []
    test_sent = []
    with open(os.path.join(data_path,'train_sent_corrected.csv')) as train_file :
        reader = csv.reader(train_file)
        for row in reader :
            train_sent.append(row)
    with open(os.path.join(data_path,'test_sent_corrected.csv')) as test_file :
        reader = csv.reader(test_file)
        for row in reader :
            test_sent.append(row)
    return(train_sent,test_sent)

def clean_sequences(sequences, language = 'fr', annotate = ['anatomy'] , annotate_meds = True, lemmatizing = False, stemming = False, remove_accent = False, correcting = True, remove_stopwords = True):
    'clean all the sequences'

    med_list = get_med_list()
    annotations = get_QUEARO_annorations()

    if remove_stopwords :
        if language == 'en':
            stpwds = stopwords.words('english')
        else :
            stpwds = stopwords.words('french')
            new_stpwds = ['a','les']
            stpwds += new_stpwds
        for i in range(len(sequences)) :
            sequences[i]=[el for el in sequences[i] if el not in stpwds]

    if annotate_meds :
        for (sent_ind,sent) in enumerate(sequences) :
            for (w_ind,w) in enumerate(sent) :
                w = unicodedata.normalize('NFD', w).encode('ascii', 'ignore').decode('UTF-8')
                if w in med_list :
                    sequences[sent_ind][w_ind] = 'médicament'

    if len(annotate) > 0 :
        for category in annotate :
            ann = annotations[category[:4]]
            if len(ann)>0 :
                for (sent_ind,sent) in enumerate(sequences) :
                    for (w_ind,w) in enumerate(sent) :
                        w = unicodedata.normalize('NFD', w).encode('ascii', 'ignore').decode('UTF-8')
                        if w in ann :
                            sequences[sent_ind][w_ind] = 'anatomie'

    if stemming :
        if language == 'en':
            stemmer = SnowballStemmer("english")
        else :
            stemmer = SnowballStemmer("french")
        for i in range(len(sequences)) :
            sequences[i]=[stemmer.stem(el) for el in sequences[i]]

    if lemmatizing :
        if language == 'en':
            lemmatizer = WordNetLemmatizer()
            no_lemmatizing = []
        else :
            lemmatizer = FrenchLefffLemmatizer()
            no_lemmatizing = ['soucis', 'sous']
        for i in range(len(sequences)) :
            sequences[i]=[lemmatizer.lemmatize(el) for el in sequences[i] if el not in no_lemmatizing]

    # if correcting :
    #     if language == 'en':
    #         d = enchant.DictWithPWL("en")
    #         for i in range(len(sequences)) :
    #             sequences[i]=[d.suggest(el)[0] for el in sequences[i] if not d.check(el) and len(d.suggest(el))>0]
    #             sequences[i]=[el for el in sequences[i] if el not in stpwds]
    #     else :
    #         spell = SpellChecker(language='fr')
    #         for i in range(len(sequences)) :
    #             print(i)
    #             correct = []
    #             for w in sequences[i]:
    #                 if spell[w]:
    #                     correct.append(w)
    #                 else :
    #                     correct.append(spell.correction(w))
    #                     print(w)
    #                     print(spell.correction(w))
    #             sequences[i]=[el for el in correct if el not in stpwds]

    if remove_accent and language == 'fr':
        for i in range(len(sequences)) :
            sequences[i]=[unicodedata.normalize('NFD', el).encode('ascii', 'ignore').decode('UTF-8') for el in sequences[i]]

    print("--- seq cleaned ---")
    return(sequences)

def build_dictionary(sequences):
    dict = {}
    for seq in sequences :
        for w in seq :
            if w in dict.keys() :
                dict[w] += 1
            else :
                dict[w] = 1
    return(dict)

def encode_sequences(train_sent,test_sent,val_sent, y_train, y_val, maxlen = 50, resize = True):
    'clean a sequence of words'
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_sent+test_sent+val_sent)
    train_seq_encoded = tokenizer.texts_to_sequences(train_sent)
    test_seq_encoded = tokenizer.texts_to_sequences(test_sent)
    val_seq_encoded = tokenizer.texts_to_sequences(val_sent)
    word2index = tokenizer.word_index
    if resize :
        train_seq_encoded = np.array(pad_sequences(train_seq_encoded, maxlen=maxlen))
        test_seq_encoded =  np.array(pad_sequences(test_seq_encoded, maxlen=maxlen))
        val_seq_encoded =  np.array(pad_sequences(val_seq_encoded, maxlen=maxlen))
    y_train = to_categorical(y_train,51)
    y_val = to_categorical(y_val,51)
    return(train_seq_encoded,test_seq_encoded, val_seq_encoded, y_train, y_val, word2index)

def create_train_val_sets(x_train, y_train, validation_ratio = 0.2):
    'split train and validation sets'
    np.random.seed(12219)
    idxs_select_train = list(np.random.choice(range(len(x_train)), size=int(len(x_train)*(1-validation_ratio)),replace=False))
    idxs_select_val = list(np.setdiff1d(range(len(x_train)),idxs_select_train))
    x_val = [x_train[id] for id in idxs_select_val]
    y_val = [y_train[id] for id in idxs_select_val]
    x_train = [x_train[id] for id in idxs_select_train]
    y_train = [y_train[id] for id in idxs_select_train]
    return(x_val,y_val,x_train,y_train,idxs_select_val)

def save_cleaned_data(sequences, filename):
    with open(filename, mode='w') as f:
        writer = csv.writer(f)
        for seq in sequences:
            writer.writerow(seq)
