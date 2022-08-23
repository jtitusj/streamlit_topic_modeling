import streamlit as st

import plotly.express as px

from file_utils import upload_file, load_data
from text_preprocessing import preprocess_string, get_word_count
from lda import train_lda

st.set_page_config(layout="wide")

st.title("Topic Modeling")

st.sidebar.title("Options & Settings")

# sidebar
st.sidebar.header("1. Data")
with st.sidebar.expander("Prepare Data", expanded=False):
    data_file = upload_file("Upload text data")
    df = load_data(data_file)

    if data_file is not None:
        text_column = st.selectbox("Select Column Name of Text Data", df.columns)

# topic modeling sidebar
st.sidebar.header("2. Topic Modeling Settings")
with st.sidebar.expander("Select Options for Topic Modeling", expanded=False):
    model_type = st.selectbox("Select Model Type",
                             ['Latent Dirichlet Allocation (LDA)',
                              'Non-Negative Matrix Factorization (NMF)',
                               'TF-IDF + K-Means'])
    (min_topic, max_topic) = st.slider("Input Number of Topics to Explore",
                             min_value=2, max_value=30, value=[2, 15])
    min_words = st.slider("Input Minimum Number of Words for a Valid Text", 1, 20, 1)

# main app display
if data_file is not None:
    st.header("Data Overview")

    # text data frames
    with st.expander("Show Text Data", expanded=False):    
            col_raw, col_cleaned = st.columns(2)
            
            raw = df[text_column]
            col_raw.subheader('Raw Text')
            col_raw.write(raw.to_frame())

            cleaned = raw.apply(lambda x: preprocess_string(x))
            col_cleaned.subheader('Processed Text')
            col_cleaned.write(cleaned.to_frame())

    # quick stats from cleaned data
    word_count_dist = (cleaned.apply(lambda x: len(x.split())).value_counts()
                        .sort_index()
                        .rename('number of texts')
                        .reset_index()
                        .rename(columns={'index': 'word count'}))

    with st.expander("Show Word Count Distribution", expanded=False):    
        fig = px.bar(word_count_dist, x='word count', y='number of texts')
        st.plotly_chart(fig, use_container_width=True)

    st.header("Determine Topics")
    n_topics = 5
    word_count = get_word_count(cleaned)
    
    # only include texts that are long enough
    long_corpus = cleaned[word_count >= min_words]

    lda_model = train_lda(long_corpus.str.split(), n_topics)
    st.write(lda_model.print_topics())