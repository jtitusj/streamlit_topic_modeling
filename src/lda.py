from gensim import corpora, models

def train_lda(cleaned_texts, n_topics):
    # create dictionary of words from cleaned text
    dictionary = corpora.Dictionary(cleaned_texts)
    dictionary.filter_extremes(no_below=5, no_above=.95)
    
    # convert text to bag-of-words (bow)
    bow_corpus = [dictionary.doc2bow(doc) for doc in cleaned_texts]

    # apply tf-idf
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    # train gensim's lda model on tf-idf corpus
    lda = models.ldamodel.LdaModel
    lda_model = lda(corpus_tfidf, num_topics=n_topics, id2word=dictionary, passes=1, 
                    random_state=0, eval_every=None)
    
    return lda_model
    
    

