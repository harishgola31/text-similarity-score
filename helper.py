import numpy as np
import base64 
import time
import re
import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
import streamlit as st


def cleaning(text):   # defining a function stemming

    text = re.sub('[^a-zA-Z]',' ', text)
    # it removes the other characters rather than alphabetical and replace with a white space 
    text = text.lower()  
    # transforming the words into lowercase that is because stopwords are in lowercase, so that why any stopwords present
    # in the text have not left to remove because of uppercase.
    text = list(set(nltk.word_tokenize(text)))
    # Creating tokens
    lemma = WordNetLemmatizer()
    text = ' '.join([lemma.lemmatize(word) for word in text if not word in stopwords.words('english')])
    # removing the stopwords if any and then lemmatize the tokens and then join them to form a single text
    return text


def average_word_vectors(words, model, vocabulary, num_features):
    '''Calculating normalised vectort for words'''

    feature_vector = np.zeros((num_features,),dtype="float64") # created a zero vector for input feature dimensions
    nwords = 0.  # taking 0 at first as number of words
    
    for word in words:  # create a loop to calculate a vector for each word
        if word in vocabulary: # if the word is present in given vocabulary
            nwords = nwords + 1.  # adding the 1 to number of word if word exists
            feature_vector = np.add(feature_vector, model.wv[word])  # adding the current vector with the vector in model
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)  # normalising the vectors
        
    return feature_vector


def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)  # creating a vocabulary from the words present in model
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features) # returning an array of vectors for every words


def cosine_similarity(A, B):
    """
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    """
    dot = round(np.dot(A, B), 3)
    norma = round(np.sqrt(np.dot(A, A)), 3)
    normb = round(np.sqrt(np.dot(B, B)), 3)
    cos = round(dot / (norma * normb), 3)

    return cos


timestr = time.strftime("%Y%m%d-%H%M%S")
class FileDownloader(object):
	
	def __init__(self, data, filename='cosine_similarity', file_ext='txt'):
		super(FileDownloader, self).__init__()
		self.data = data
		self.filename = filename
		self.file_ext = file_ext

	def download(self):
		b64 = base64.b64encode(self.data.encode()).decode()
		new_filename = "{}_{}_.{}".format(self.filename, timestr, self.file_ext)
		st.markdown("#### Download Complete Result ###")
		href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Click Here!!</a>'
		st.markdown(href,unsafe_allow_html=True)


