import re
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import text_to_word_sequence
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def preprocessing(documents, stop_words, to_sentence = False):
    corpus = []
    for i in range(0, len(documents)):    
        document = re.sub('[^a-zA-Z]', ' ', str(documents[i]))
        document = document.lower()
        document = document.split() # tokenizing
        ps = PorterStemmer()
        document = [ps.stem(word) for word in document if not word in set(stop_words)]
        if to_sentence == True:
            document = ' '.join(document)  # in case we want to put the words back into a sentence
        corpus.append(document)
    return corpus
	
def vectorizing(data, tokenizer ,max_len):
    sequences = tokenizer.texts_to_sequences(data)
    padding = pad_sequences(sequences, maxlen = max_len)
    
    return padding


