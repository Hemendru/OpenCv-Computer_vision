import nltk

nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
ps = PorterStemmer()


# data preprocessing
# lower case 
# tokenization
#removing special characters and removing characters
#stemming


def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)



tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
st.title('Email Spam Classifier')
input_email = st.text_area('Enter the email content here:')



if st.button('Predict'):
    # transform the input email
    input_email = transform_text(input_email)
     # vectorize the input email
    vector_email = tfidf.transform([input_email])
# predict the email
    prediction = model.predict(vector_email)   
    # Display the result
    if prediction[0] == 1:
        st.write('The email is classified as: **SPAM**')
    else:
        st.write('The email is classified as: **NOT SPAM**')