import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
from collections import Counter


class NLPAnalysis():
    
    def __init__(self, inputfile):
        
        self.script = inputfile
    
    def rep_SCRIPT_lines(self):
        pattern = r"\b[A-Z][A-Z]+\b" #All caps words
        lines = nltk.sent_tokenize(self.script)
        lines = [re.sub(pattern, '', l) for l in lines]
        lines = str(lines)
        return lines
    
    def rep_script_lines(self):
        pattern = r'[A-Za-z]+:' #Words before a colon
        lines = nltk.sent_tokenize(self.script)
        lines = [re.sub(pattern, '', l) for l in lines]
        lines = str(lines)
        return lines
    
    def bow(self,N):
        
        pattern1 = r"\b[A-Z][A-Z]+\b" #All caps words
        pattern2 = r'[A-Za-z]+:' #Words before a colon
        lines = nltk.sent_tokenize(self.script)
        lines = [re.sub(pattern1, '', l) for l in lines]
        lines = [re.sub(pattern2, '', l) for l in lines]
        lines = str(lines)
        tokens = nltk.word_tokenize(lines)
        lower_tokens = [t.lower() for t in tokens]
        alpha_only = [t for t in lower_tokens if t.isalpha()]
        no_stops = [t for t in alpha_only if t not in stopwords.words('english')]
        wrdnet_lemmas = WordNetLemmatizer()
        lemma = [wrdnet_lemmas.lemmatize(t) for t in no_stops]
        bow=Counter(lemma)
        return bow.most_common(N)

    
    def word_len(self):
    
        words = nltk.word_tokenize(self.script)
        word_lengths = [len(w) for w in words]
    
        avg_word_length = np.mean(word_lengths)
        med_word_length = np.median(word_lengths)
        return word_lengths, avg_word_length, med_word_length
    
    
    def sent_len(self):
    
        sentences = nltk.sent_tokenize(self.script)
        tkn_sentences = [nltk.regexp_tokenize(s, r'\w+') for s in sentences]
        sentence_lengths = [len(sen) for sen in tkn_sentences]
    
        avg_sentence_length = np.mean(sentence_lengths)
        med_sentence_length = np.median(sentence_lengths)
        return sentence_lengths, avg_sentence_length, med_sentence_length

