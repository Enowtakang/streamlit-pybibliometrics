"""
cd Streamlit_Pybibliometrics
streamlit run main.py
"""
import streamlit as st
from streamlit_lottie import st_lottie
import requests
from pybliometrics.scopus import ScopusSearch
import pandas as pd
import gensim.utils
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim import corpora
import numpy as np
np.random.seed(2022)
st.set_page_config(layout='wide')


# 1. Select task page
st.sidebar.markdown("Select task page")
page = st.sidebar.radio("What do you want to do now?",
                        ("Data acquisition",
                         "Bibliometric analysis"))


# 2. Customize lottie size
st.sidebar.markdown(" ")
st.sidebar.markdown("Customize lottie dimensions")
lottie_height = st.sidebar.slider(
    "Lottie height", 100, 1000, 250)
lottie_width = st.sidebar.slider(
    "Lottie width", 100, 1000, 250)
lottie_speed = st.sidebar.slider(
    "Lottie speed", 1, 10, 2)


# 3. build lottie
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie = load_lottieurl('https://assets6.lottiefiles.com/packages/lf20_zgetuj6b.json')
st_lottie(lottie,
          height=lottie_height,
          speed=lottie_speed,
          width=lottie_width)


# 4. Title and subheading
st.title("Bibliometric Analysis App")
st.subheader("developed by Enow Takang")

# The code below just gives more space between
# the subheader and the rest of the
# stuff in the page
st.subheader(" ")


# 5. DATA ACQUISITION page
if page == "Data acquisition":
    # create a 'query' variable for text input.
    # This variable would be supplied to the
    # query argument in ScopusSearch()
    query = st.text_input(
        "Search phrase/keywords")

    # create 3 columns to enable the user
    # do the following:
    # 1. determine weather or not to download
    # the results
    # 2. Determine if they are subscribed :)
    # 3. Begin the search process

    col1, col2 = st.columns((10, 10))
    with col1:
        download = st.selectbox(
            "Do you want to download the results?",
            (True, False))

    with col2:
        subscriber = st.selectbox(
            "Are you subscribed to SCOPUS?",
            (True, False))

    try:
        s = ScopusSearch(
            query=query,
            download=download,
            subscriber=subscriber)
        # Show s in col3
        st.write(s)

        # When the user selects 'true'
        # for 'download', what happens?
        if download:
            # store results in pandas df
            df = pd.DataFrame(
                pd.DataFrame(s.results))
            # ask user for file name
            file_name = st.text_input(
                "To save results, provide file name field (with .csv extension)")
            # save data to working directory
            df.to_csv(file_name)
            st.write("Your excel file has been saved!")

    except:
        pass


# 6. BIBLIOMETRIC ANALYSIS page
if page == "Bibliometric analysis":
    # Load the downloaded data
    loaded_file = st.file_uploader(
        "Select your local file to "
        "begin bibliometric analysis")
    # stop the app if there is no selected file,
    # or load a selected file.
    if loaded_file is not None:
        data_df = pd.read_csv(loaded_file)
    else:
        st.stop()

    # show/hide dataframe
    st.markdown(" ")
    view_data_frame = st.selectbox(
        "Do you want to view the uploaded dataframe?",
        ("No", "Yes"))
    if view_data_frame == "Yes":
        st.write(data_df)
    else:
        pass

    # make some space
    st.markdown(" ")
    st.markdown(" ")

    # select specific columns
    selected_columns = st.multiselect(
        "Select specific column(s) which you "
        "would like to view",
        data_df.columns)
    if selected_columns:
        st.write(data_df[selected_columns])

    # TOPIC MODELLING
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    model_decision = st.selectbox(
        "Do you want to perform topic modelling?",
        ("No", "Yes"))
    if model_decision == "Yes":
        # collect user input on number of 'topics'
        # and number of 'passes'.
        col_p, col_q = st.columns(2)
        with col_p:
            number_of_topics = st.number_input(
                "How many topics do you want to obtain?",
                min_value=1)
        with col_q:
            number_of_passes = st.number_input(
                "Specify the number of passes (more is better)",
                min_value=1)

            # create button to be clicked
            # before results are shows
            button = st.button("View Results")

        # isolate document titles
        df = data_df["title"]

        # preprocessing
        stemmer = SnowballStemmer("english")


        def lemmatize_stemming(text):
            return stemmer.stem(
                WordNetLemmatizer().lemmatize(text, pos='v'))


        def preprocess(text):
            result = []
            for token in gensim.utils.simple_preprocess(text):
                if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                    result.append(lemmatize_stemming(token))
            return result

        # Map all documents through the
        # 'preprocess' function
        processed_docs = df.map(preprocess)

        # Transform data into a bag_of_words model
        dictionary = gensim.corpora.Dictionary(processed_docs)
        count = 0

        for k, v in dictionary.iteritems():
            count += 1
            if count > 10:
                break

        # Create a bag_of_words (bow) corpus
        bow_corpus = [
            dictionary.doc2bow(doc) for doc in processed_docs]

        # note this area
        n = len(df) - 1
        bow_doc_n = bow_corpus[n]

        # Create Latent Dirichlet Association model
        lda_model = gensim.models.LdaModel(
            bow_corpus,
            # insert user input collected above
            num_topics=number_of_topics,
            id2word=dictionary,
            # insert user input collected above
            passes=int(number_of_passes))

        for idx, topic in lda_model.print_topics(-1):
            if button:
                st.write("Topic {} \nWords: {}".format(idx, topic))

    else:
        pass
