import numpy as np
import os
import re
import itertools
import operator
import copy
import heapq
import nltk
from collections import defaultdict
import networkx as nx
import random
from gensim.models import KeyedVectors


def seq_by_class(train_sent,y_train) :
    by_class = defaultdict(list)
    for i in range(len(y_train)):
        by_class[y_train[i]].append(train_sent[i])
    return(by_class)

def terms_to_graph(lists_of_terms, window_size, overspanning):
    '''This function returns a directed, weighted igraph from lists of list of terms (the tokens from the pre-processed text)
    e.g., [['quick','brown','fox'], ['develop', 'remot', 'control'], etc]
    Edges are weighted based on term co-occurence within a sliding window of fixed size 'w' '''
    if overspanning:
        terms = [item for sublist in lists_of_terms for item in sublist]
    else:
        idx = 0
        terms = lists_of_terms[idx]
    from_to = {}
    while True:
        w = min(window_size, len(terms))
        # create initial complete graph (first w terms)
        terms_temp = terms[0:w]
        indexes = list(itertools.combinations(range(w), r=2))
        new_edges = []
        for my_tuple in indexes:
            new_edges.append(tuple([terms_temp[i] for i in my_tuple]))
        for new_edge in new_edges:
            if new_edge in from_to:
                from_to[new_edge] += 1
            else:
                from_to[new_edge] = 1
        # then iterate over the remaining terms
        for i in range(w, len(terms)):
            # term to consider
            considered_term = terms[i]
            # all terms within sliding window
            terms_temp = terms[(i - w + 1):(i + 1)]
            # edges to try
            candidate_edges = []
            for p in range(w - 1):
                candidate_edges.append((terms_temp[p], considered_term))
            for try_edge in candidate_edges:
                # if not self-edge
                if try_edge[1] != try_edge[0]:
                    # if edge has already been seen, update its weight
                    if try_edge in from_to:
                        from_to[try_edge] += 1
                    # if edge has never been seen, create it and assign it a unit weight
                    else:
                        from_to[try_edge] = 1

        if overspanning:
            break
        else:
            idx += 1
            if idx == len(lists_of_terms):
                break
            terms = lists_of_terms[idx]
    # create empty graph
    g = nx.DiGraph()
    #g = igraph.Graph(directed=True)
    # add vertices
    if overspanning:
        #g.add_vertices(sorted(set(terms)))
        g.add_nodes_from(sorted(set(terms)))
    else:
        #g.add_vertices(sorted(set([item for sublist in lists_of_terms for item in sublist])))
        g.add_nodes_from(set([item for sublist in lists_of_terms for item in sublist]))
    # add edges, direction is preserved since the graph is directed
    #g.add_edges(list(from_to.keys()))
    keys = list(from_to.keys())
    vals = list(from_to.values())
    edges = []
    for i in range(len(keys)):
        edges.append((keys[i][0],keys[i][1],vals[i]))
    g.add_weighted_edges_from(edges)
    # set edge and vertice weights
    #g.es['weight'] = list(from_to.values())  # based on co-occurence within sliding window
    #g.vs['weight'] = g.strength(weights=list(from_to.values()))  # weighted degree
    return (g)

def random_walk(graph, node, max_walk_length):
    walk = [node]
    while len(walk) < max_walk_length :
        current_node = walk[-1]
        neighbors = list(graph.neighbors(current_node))
        if len(neighbors) == 0 :
            break
        else :
            counts = np.array([graph.get_edge_data(current_node,neighbor)['weight'] for neighbor in neighbors])
            probs = counts / np.sum(counts)
            walk.append(np.random.choice(neighbors , p = probs))
    return walk

def generate_walks(graph, num_walks, max_walk_length):
    '''
    samples num_walks walks of maximum length max_walk_length+1 from a random node of graph.
    '''
    graph_nodes = graph.nodes()
    walks = []
    for i in range(num_walks):
        node = random.choice(list(graph_nodes))
        walk = random_walk(graph, node, max_walk_length)
        walks.append(walk)
    return walks

def augment_data_random_walk(train_sent, y_train, window_size = 3, proportion = 2, maxlen = 50) :
    new_train_sent = train_sent
    new_y_train = y_train
    by_class = seq_by_class(train_sent,y_train)
    for cl in by_class.keys() :
        g = terms_to_graph(by_class[cl], window_size, overspanning = False)
        n_rand_walks = proportion*len(by_class[cl])
        walks = generate_walks(g, n_rand_walks, maxlen)
        new_train_sent += walks
        new_y_train += [cl]*n_rand_walks
    return(new_train_sent,new_y_train)

def replace_embedding(train_sent, y_train, p = 0.5, bin_file = '../data/embeddings/frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin'):
    wv = KeyedVectors.load_word2vec_format(bin_file, binary=True)
    new_train_sent = []
    new_y_train = []
    by_class = seq_by_class(train_sent,y_train)
    for cl in by_class.keys() :
        for sent in by_class[cl] :
            new_sent = []
            for word in sent :
                if np.random.rand() < p :
                    new_word = wv.most_similar(word)[0][0]
                    new_sent.append(new_word)
                else :
                    new_sent.append(word)
            new_train_sent.append(new_sent)
            new_train_sent.append(sent)
            new_y_train.append(cl)
            new_y_train.append(cl)
    return(new_train_sent,new_y_train)
