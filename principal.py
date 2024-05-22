#%pip install nltk -U
#%pip install spacy -U
%pip install gensim
#%pip install pyldavis
#%pip install gutenbergpy

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import nltk
import re
import string
import gensim
import numpy as np

# for cleaning prefatory matter from Project Gutenberg texts
from gutenbergpy import textget

# for tokenization
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download('wordnet')

# for stopword removal
from nltk.corpus import stopwords
nltk.download('stopwords')

# for lemmatization and POS tagging
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')

# for LDA
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import datapath

# for LDA evaluation
import pyLDAvis
import pyLDAvis.gensim_models as gensimvisualize

#temp_file = datapath("/content/drive/MyDrive/ColabNotebooks/LDA_SesionesCongreso/modelo")
temp_file = datapath("modelo")
lda_model = LdaModel.load(temp_file)

# print LDA topics
for topic in lda_model.print_topics(num_topics=10, num_words=10):
    print(topic)

def main():
    st.title("Probando Streamlit Cloud")
    st.write("Esta web debe contener un modelo de LDA")
    # Create some example data
    data = pd.DataFrame({
        'Category': ['A', 'B', 'C', 'D'],
        'Values': [10, 20, 15, 25]
    })

    # Display the data as a table
    st.write("Example Data:")
    st.write(data)

    # Create a bar chart
    st.write("Bar Chart:")
    fig, ax = plt.subplots()
    ax.bar(data['Category'], data['Values'])
    st.pyplot(fig)

#if __name__ == "__main__":
#    main()
