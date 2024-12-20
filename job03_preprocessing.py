import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle
from konlpy.tag import Okt  # Kkma #MeCab
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('C:/work/news_category_classfication/crawling_data_2/naver_news_titles_20241220_0925.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
df.info()
print(df.category.value_counts())

X = df['title']  # titles -> title로 수정
Y = df['category']

print(X[0])
okt = Okt()
okt_x = okt.morphs(X[0], stem=True)
print(okt_x)
# kkma = Kkma()
# kkma_x = kkma.morphs(X[0])
# print(kkma_x)
# exit()

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
print(labeled_y[:3])

label = encoder.classes_
print(label)

with open('./models/encoder.pickle', 'wb') as f:
   pickle.dump(encoder, f)

onehot_Y = to_categorical(labeled_y)
print(onehot_Y)

for i in range(len(X)):
   X[i] = okt.morphs(X[i], stem=True)
print(X)

stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col=0)
print(stopwords)

for sentence in range(len(X)):
   words = []
   for word in range(len(X[sentence])):
       if len(X[sentence][word]) > 1:
           if X[sentence][word] not in list(stopwords['stopword']):
               words.append(X[sentence][word])
   X[sentence] = ' '.join(words)
print(X[:5])

token = Tokenizer()
token.fit_on_texts(X)
tokened_X = token.texts_to_sequences(X)
wordsize = len(token.word_index) + 1
print(wordsize)

print(tokened_X[:5])

