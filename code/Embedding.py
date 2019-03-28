
 #coding: utf8

import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras.preprocessing.text import *
import csv
import os
from collections import Counter
import matplotlib.pyplot as plt
import operator


def create_embedding(bin_file, word2index, input_embedding_dim = 200, output_embedding_dim = 128, intersect_bin_file = True):
    #wv = KeyedVectors.load_word2vec_format(bin_file, binary=True)
    word_vectors = Word2Vec(size=input_embedding_dim, min_count=1) # initialize word vectors
    word_vectors.build_vocab([[elt] for elt in list(word2index.keys())])
    if intersect_bin_file :
       word_vectors.intersect_word2vec_format(bin_file, binary=True)
    embeddings = word_vectors.wv.syn0
    #add zero vector (for padding special token)
    pad_vec = np.zeros((1, input_embedding_dim))
    embeddings = np.insert(embeddings,0,pad_vec,0)
    # # add Gaussian initialized vector (for OOV special token)
    # oov_vec = np.random.normal(size=input_embedding_dim)
    # embeddings = np.insert(embeddings,0,oov_vec,0)
    # # reduce dimension with PCA (to reduce the number of parameters of the model)
    if output_embedding_dim != input_embedding_dim :
         my_pca = PCA(n_components=output_embedding_dim)
         embeddings = my_pca.fit_transform(embeddings)
    word2embedding = {}
    for word in word2index.keys():
       word2embedding[word] = embeddings[word2index[word]]
    return(embeddings,word2embedding)


def test_word(doc,dict):
    OOV = []
    for sent in doc :
        for word in sent :
            if word not in dict and word not in OOV:
                OOV.append(word)
    return(OOV)

def get_mean_emdedding(x,word2embedding, maxlen = 100):
    d = len(list(word2embedding.values())[0])
    x_emb = np.zeros((len(x),d))
    for i,seq in enumerate(x) :
        if len(seq) == 0 :
            emb = [np.zeros(d)]
        emb = [word2embedding[word] for word in seq]
        emb = np.mean(np.array(emb))
        x_emb[i] = emb
    return(x_emb)

def visualize_embedding(sentences, word2index, embeddings, output_file = '../Visualize/word_embeddings.png') :
    d = embeddings.shape[1]
    n_mf = 100
    all_tokens = [token for sublist in sentences for token in sublist]
    t_counts = dict(Counter(all_tokens))
    sorted_t_counts = sorted(list(t_counts.items()), key=operator.itemgetter(1), reverse=True)
    mft = [elt[0] for elt in sorted_t_counts[:n_mf]]
    # store the vectors of the most frequent words in np array
    mft_vecs = np.empty(shape=(n_mf,d))
    for idx,token in enumerate(mft):
        mft_vecs[idx,:] = embeddings[word2index[token]]

    my_pca = PCA(n_components=10)
    my_tsne = TSNE(n_components=2,perplexity=10)

    mft_vecs_pca = my_pca.fit_transform(mft_vecs)
    mft_vecs_tsne = my_tsne.fit_transform(mft_vecs_pca)

    fig, ax = plt.subplots()
    ax.scatter(mft_vecs_tsne[:,0], mft_vecs_tsne[:,1],s=3)
    for x, y, token in zip(mft_vecs_tsne[:,0] , mft_vecs_tsne[:,1], mft):
        ax.annotate(token, xy=(x, y), size=8)
    fig.suptitle('t-SNE visualization of 100 most frequent word embeddings',fontsize=20)
    fig.set_size_inches(11,7)
    fig.savefig(output_file,dpi=300)
    fig.show()
