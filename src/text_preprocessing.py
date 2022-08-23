import string
import advertools as adv

from textblob import Word


stopwords_english = list(adv.stopwords['english'])
stopwords_tagalog = list(adv.stopwords['tagalog'])
custom_stopwords = ["lang", "nga", "ha", "po", "yung", "mo"]

stopwords_all = stopwords_english + stopwords_tagalog + custom_stopwords

def remove_punctuations(s):
    return s.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))

def remove_stopwords(s):
    return ' '.join([word for word in s.split() if word not in stopwords_all])

def lemmatize(s):
    return ' '.join([Word(w).lemmatize() for w in s.split()])

def preprocess_string(s):
    s_ = remove_stopwords(remove_punctuations(str(s).lower()).strip())
    return lemmatize(s_)

def get_word_count(series):
    return series.apply(lambda x: len(x.split()))