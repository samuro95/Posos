# coding: utf-8

import pickle
from operator import itemgetter
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
from time import time


def damerau_levenshtein_distance(seq1,seq2) :
    """
    Calculate the Damerau-Levenshtein distance between 2 words.
    This distance is the number of additions, deletions, substitutions,
    and transpositions needed to transform the first sequence into the
    second.
    """
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        twoago, oneago, thisrow = oneago, thisrow, [0] * len(seq2) + [x + 1]
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            # This block deals with transpositions
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                and seq1[x-1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


def formal_similarity_with_dictionary(word, dictionary, k = 2):
    """
        Return, for a word, the sorted damerau levenshtein distances with the words
        from a given dictionary, for distances smaller than the minimal.
    """
    distances = np.array([damerau_levenshtein_distance(word,w) for w in dictionary])
    sorted_distances = sorted(enumerate(distances), key=itemgetter(1))
    sorted_distances = [el for el in sorted_distances if el[1]<=k]
    if len(sorted_distances) == 0 :
        return([],[])
    else :
        indices, distances = zip(*sorted_distances)
        return(indices, distances)

class OOV :

    """
    OOV module that assigns a word to any token not included
    in the dictionary.
    """

    def __init__(self, dict_emb, dict_corpus, max_count_in_courpus = 2):
        # We consider 2 dictionaries : the one defined by the corpus and the one defined by the pretrained embedding
        self.dict_emb = list(dict_emb)
        self.dict_emb_no_accent = [unicodedata.normalize('NFD', word).encode('ascii', 'ignore').decode('UTF-8') for word in self.dict_emb]
        self.word2id_emb = {w:i for (i, w) in enumerate(self.dict_emb)}
        self.id2word_emb = dict(enumerate(self.dict_emb))
        self.dict_corpus = dict_corpus.keys()
        self.corpus_count = dict_corpus

    # def embedding_similarity(self, word):
    #     """
    #         Return, for a word in the embedding dictionary, the sorted
    #         (decreasing order) cosine similarity distances with other words.
    #     """
    #     word_index = self.word2id[word]
    #     e = self.embeddings[word_index]
    #     distances = cosine_similarity([e],self.embeddings)[0]
    #     sorted_distances = sorted(enumerate(distances), key=itemgetter(1), reverse = True)
    #     indices, distances = zip(*sorted_distances)
    #     return(indices, distances)

    def replace_OOV(self,word):
        '''
        Given a OOV word, returns the most similar word.
        '''

        word_no_accent = unicodedata.normalize('NFD', word).encode('ascii', 'ignore').decode('UTF-8')
        if word in self.dict_emb :
            # if word in voc
            return(word)
        elif word in self.dict_emb_no_accent :
            # if word without accent is in the voc without accent, return the corresponding word with accent
            return(self.dict_emb[self.dict_emb_no_accent.index(word)])
        else :
            indices, dist = formal_similarity_with_dictionary(word, self.dict_emb, k = 2)
            if len(indices) == 0 :
                # si aucun mots dans le voc à une distance 2
                return(None)
            else :
                words_in_corpus = [self.id2word_emb[ind] for ind in indices if self.id2word_emb[ind] in self.corpus_count.keys()]
                if len(words_in_corpus) == 0 :
                    indices_min = [indices[i] for i in range(len(indices)) if dist[i] == min(dist)]
                    return(self.id2word_emb[min(indices)])
                else :
                    count_in_corpus = [self.corpus_count[w] for w in words_in_corpus]
                    index_max, Max = max(enumerate(count_in_corpus), key=itemgetter(1))
                    if Max > 1 :
                        return(words_in_corpus[index_max])
                    else :
                        return(None)

    def corpus_correction(self,sequences):
        result = []
        for i,sequence in enumerate(sequences) :
            t0 = time()
            correction = [self.replace_OOV(w) for w in sequence if self.replace_OOV(w) is not None]
            result.append(correction)
            t1 = time()
            print("--- seq %i corrected in %d sec ---"%(i,t1-t0))
        return(result)
