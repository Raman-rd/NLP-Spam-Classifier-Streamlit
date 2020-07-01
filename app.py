import streamlit as st

import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import pickle

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename,'rb'))
cv = pickle.load(open('tranform.pkl','rb'))
st.title(" Email Spam Classifier")

st.write("""
*** This application uses Naive Bayes Classifier and try to
predict if an Email is Spam or Ham *** """)

data = st.text_input("Enter message")
data = [data]
vect = cv.transform(data).toarray()
my_prediction = clf.predict(vect)
if st.button('predict'):
    if my_prediction == 0:
        st.write("Not a Spam")
    elif my_prediction == 1:
        st.write("Spam")

st.write("# Made by Raman")
