import pandas as pd
import pickle

data = pd.read_csv(r'C:\Users\Neha\Downloads\AI\10th- Ensemble Learning\Machine Learning CAPSTONE PROJECT\MOVIE GENERE\movie_genre_prediction\movie_train.csv')

genre_mapper = {'other': 0, 'action': 1, 'adventure': 2, 'comedy':3, 'drama':4, 'horror':5, 'romance':6, 'sci-fi':7, 'thriller': 8}
data['genre'] = data['genre'].map(genre_mapper)

data.drop('id', axis=1, inplace=True)

import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=  []
ps = PorterStemmer() 

for i in range(0, data.shape[0]):

  # Cleaning special character from the dialog/script
  dialog = re.sub(pattern='[a-zA-Z]', repl=' ', string=data[''][i])

  # Converting the entire dialog/script into lower case
  dialog = dialog.lower()

  # Tokenizing the dialog/script by words
  words = dialog.split()

  # Removing the stop words
  dialog_words = [word for word in words if word not in set(stopwords.words('english'))]

  # Stemming the words
  words = [ps.stem(word) for word in dialog_words]

  # Joining the stemmed words
  dialog = ' '.join(words)

  # Creating a corpus
  corpus.append(dialog)