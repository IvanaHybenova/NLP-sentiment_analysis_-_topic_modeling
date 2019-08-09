import pandas as pd
import numpy as np
import gensim 
from gensim.models import Word2Vec
from keras.preprocessing.text import text_to_word_sequence

class WordEmbedding:
    def __init__(self, num_features = 100, min_word_count = 5, num_workers = 4, window = 5):
        self.num_features = num_features
        self.min_word_count = min_word_count
        self.num_workers = num_workers
        self.window = window
        self.model = None
        
    def fit(self, data):
        self.model = gensim.models.Word2Vec(data, 
                                   min_count = self.min_word_count,
                                   size = self.num_features, 
                                   window = self.window, 
                                   workers = self.num_workers)
        return self.model
    
    def size(self):
        print("Total number of words in the vocabulary: ", self.model.wv.syn0.shape)
    
    def to_file(self):
        self.model.wv.save_word2vec_format('trained_embedding_word2vec.txt', binary = False)