# Adapted from https://github.com/abhishekkrthakur/is_that_a_duplicate_quora_question/blob/master/feature_engineering.py

import cPickle
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
from nltk import word_tokenize
from gensim.models import KeyedVectors

class FeaturesGenerator():
    
    def wmd(self, model, s1, s2):
        """Compute word mover distance using embedding model."""
        s1 = str(s1).split()
        s2 = str(s2).split()
        return model.wmdistance(s1, s2)
    
    def sent2vec(self, model, s):
        """Sentence to vec using pre-trained word embedding. Sentence is normalised sum of word vectors."""
        words = str(s).decode('utf-8')
        words = word_tokenize(words)
        words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                M.append(model[w])
            except:
                continue
        M = np.array(M)
        v = M.sum(axis=0)
        return v / np.sqrt((v ** 2).sum())

    def sent2vec_max(self, model, s):
        """Sentence to vec using pre-trained word embedding. Takes maximum value in each dimension of word vectors."""
        words = str(s).decode('utf-8')
        words = word_tokenize(words)
        words = [w for w in words if w.isalpha()]
        M = []
        for w in words:
            try:
                M.append(model[w])
            except:
                continue
        M = np.array(M)
        v = M.max(axis=0) if len(M) > 0 else np.array([0])
        return v / np.sqrt((v ** 2).sum())
    
    def features_set1(self, d, model):
        # some basic features
        d['len_q1'] = d.question1.apply(lambda x: len(str(x)))
        d['len_q2'] = d.question2.apply(lambda x: len(str(x)))
        d['diff_len'] = d.len_q1 - d.len_q2
        d['len_char_q1'] = d.question1.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        d['len_char_q2'] = d.question2.apply(lambda x: len(''.join(set(str(x).replace(' ', '')))))
        d['diff_chars'] = d.len_char_q1 - d.len_char_q2
        d['len_word_q1'] = d.question1.apply(lambda x: len(str(x).split()))
        d['len_word_q2'] = d.question2.apply(lambda x: len(str(x).split()))
        d['diff_words'] = d.len_word_q1 - d.len_word_q2
        d['common_words'] = d.apply(lambda x: len(set(str(x['question1']).split()).intersection(set(str(x['question2']).split()))), axis=1)
        d['fuzz_qratio'] = d.apply(lambda x: fuzz.QRatio(str(x['question1']), str(x['question2'])), axis=1)
        d['fuzz_WRatio'] = d.apply(lambda x: fuzz.WRatio(str(x['question1']), str(x['question2'])), axis=1)
        d['fuzz_partial_ratio'] = d.apply(lambda x: fuzz.partial_ratio(str(x['question1']), str(x['question2'])), axis=1)
        d['fuzz_partial_token_set_ratio'] = d.apply(lambda x: fuzz.partial_token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        d['fuzz_partial_token_sort_ratio'] = d.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        d['fuzz_token_set_ratio'] = d.apply(lambda x: fuzz.token_set_ratio(str(x['question1']), str(x['question2'])), axis=1)
        d['fuzz_token_sort_ratio'] = d.apply(lambda x: fuzz.token_sort_ratio(str(x['question1']), str(x['question2'])), axis=1)
        
        # word mover distance using sentence vector
        d['wmd'] = d.apply(lambda x: self.wmd(model, x['question1'], x['question2']), axis=1)
        return d
    
    def questions_to_vectors_sum(self, d, model):
        question1_vectors = np.zeros((d.shape[0], 300))
        error_count = 0
        for i, q in tqdm(enumerate(d.question1.values)):
            question1_vectors[i, :] = self.sent2vec(model, q)
        question1_vectors = np.nan_to_num(question1_vectors)

        question2_vectors  = np.zeros((d.shape[0], 300))
        for i, q in tqdm(enumerate(d.question2.values)):
            question2_vectors[i, :] = self.sent2vec(model, q)
        question2_vectors = np.nan_to_num(question2_vectors)
        
        return question1_vectors, question2_vectors

    def questions_to_vectors_max(self, d, model):
        question1_vectors = np.zeros((d.shape[0], 300))
        error_count = 0
        for i, q in tqdm(enumerate(d.question1.values)):
            question1_vectors[i, :] = self.sent2vec_max(model, q)
        question1_vectors = np.nan_to_num(question1_vectors)

        question2_vectors  = np.zeros((d.shape[0], 300))
        for i, q in tqdm(enumerate(d.question2.values)):
            question2_vectors[i, :] = self.sent2vec_max(model, q)
        question2_vectors = np.nan_to_num(question2_vectors)
        
        return question1_vectors, question2_vectors

    def distances(self, question1_vectors, question2_vectors):        
        d = pd.DataFrame()
        # various distances between sentence vectors
        print('calculating distances between document vectors')
        d['cosine_distance'] = [cosine(x, y) for (x, y) in zip(question1_vectors, question2_vectors)]
        d['cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(question1_vectors, question2_vectors)]
        d['jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(question1_vectors, question2_vectors)]
        d['canberra_distance'] = [canberra(x, y) for (x, y) in zip(question1_vectors, question2_vectors)]
        d['euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(question1_vectors, question2_vectors)]
        d['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(question1_vectors, question2_vectors)]

        d['skew_q1vec'] = [skew(x) for x in question1_vectors]
        d['skew_q2vec'] = [skew(x) for x in question2_vectors]
        d['ratio_skew'] = d.skew_q1vec / d.skew_q2vec
        d['kur_q1vec'] = [kurtosis(x) for x in question1_vectors]
        d['kur_q2vec'] = [kurtosis(x) for x in question2_vectors]
        d['ratio_kur'] = d.kur_q1vec / d.kur_q2vec

        return d
        