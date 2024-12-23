import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pickle
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# okt 객체 초기화 부분 추가
okt = Okt()

df = pd.read_csv('C:/work/news_category_classfication/crawling_data/naver_headline_news_20241223.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
df.info()
print(df.category.value_counts())

X = df['titles']
Y = df['category']

# pickle load/dump 부분 수정
with open('./models/encoder.pickle', 'rb') as f:  # wb를 rb로 수정
    encoder = pickle.load(f)

label = encoder.classes_
print(label)

labeled_y = encoder.transform(Y)
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

# 중복된 코드 제거
tokened_X = token.texts_to_sequences(X)

# max변수를 max_len으로 변경
max_len = 14
for i in range(len(tokened_X)):
    if len(tokened_X[i]) > max_len:
        tokened_X[i] = tokened_X[i][:max_len]
X_pad = pad_sequences(tokened_X, 16)

print(tokened_X[:5])

# 경로 수정
model = load_model('./models/news_category_classification_model_0.699999988079071.h5')
preds = model.predict(X_pad)

predicts = []
for pred in preds:
   most = label[np.argmax(pred)]
   pred[np.argmax(pred)] = 0
   second = label[np.argmax(pred)]
   predicts.append([most, second])
df['predict'] = predicts
print(df.head(30))

score = model.evaluate(X_pad, onehot_Y)
print(score[1])

df['OX'] = 0
for i in range(len(df)):
   if df.loc[i, 'category'] in df.loc[i, 'predict']:
       df.loc[i, 'OX'] = 1
print(df['OX'].mean())
