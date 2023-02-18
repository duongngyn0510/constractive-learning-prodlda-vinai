from sklearn.datasets import fetch_20newsgroups
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
import re
import pandas as pd
import pickle

categories = ['talk.politics.guns', 'sci.crypt', 'alt.atheism', 'comp.graphics', 'rec.autos']
data = fetch_20newsgroups(subset='all', 
                          categories=categories,
                          shuffle=False, 
                          remove=('headers', 'footers', 'quotes'))

text, label = data.data, data.target
df = pd.DataFrame({'text': text, 'label': label})

# cleaning
stemmer = SnowballStemmer("english")
def stem(text):
    word_list = word_tokenize(text)
    lemmatized_doc = ""
    for word in word_list:
        lemmatized_doc = lemmatized_doc + " " + stemmer.stem(word)
    return lemmatized_doc.strip()

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def remove_special_characters(text):
    return re.sub(' +', ' ', text)

def remove_stopwords(text):
    stop_word = set(stopwords.words('english'))
    text = [word for word in text.split() if word not in stop_word]
    text = ' '.join(x.lower() for x in text)
    return text

df['text'] = df['text'].apply(stem).apply(remove_special_characters).apply(remove_special_characters).apply(remove_stopwords)

count_vectorizer = CountVectorizer()
bow = count_vectorizer.fit_transform(df['text'])
with open ('bow.pkl', 'wb') as f:
    pickle.dump(bow, f)

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(df['text'])
with open ('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open ('label.pkl', 'wb') as f:
    pickle.dump(label, f)