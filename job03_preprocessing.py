# 뉴스 카테고리 분류를 위한 데이터 전처리 작업
# 뉴스 제목을 분석하여 카테고리를 예측하는 모델을 만들기 위한 데이터 준비 과정입니다

# 필요한 라이브러리들을 가져옵니다
import pandas as pd  # 데이터 처리를 위한 판다스 - 엑셀처럼 데이터를 다룰 수 있게 해줍니다
import numpy as np   # 숫자 계산을 위한 넘파이 - 복잡한 수학 계산을 쉽게 할 수 있게 해줍니다
from sklearn.model_selection import train_test_split  # 데이터 분리 - 학습용과 테스트용 데이터를 나눕니다
from sklearn.preprocessing import LabelEncoder  # 카테고리 데이터를 숫자로 변환 - 문자로 된 카테고리를 컴퓨터가 이해할 수 있는 숫자로 바꿔줍니다
from keras.utils import to_categorical  # 원-핫 인코딩 변환 - 카테고리를 좀 더 효과적인 형태로 변환합니다
import pickle  # 파일 저장 - 전처리한 데이터를 파일로 저장할 때 사용합니다
from konlpy.tag import Okt  # 한글 형태소 분석기 - 한글 문장을 단어 단위로 쪼개줍니다
from tensorflow.keras.preprocessing.text import Tokenizer  # 텍스트를 숫자로 변환 - 단어를 컴퓨터가 이해할 수 있는 숫자로 바꿔줍니다
from tensorflow.keras.preprocessing.sequence import pad_sequences  # 문장 길이 맞추기 - 서로 다른 길이의 문장을 동일한 길이로 맞춰줍니다

# 1. 데이터 불러오기 및 기본 전처리
# 크롤링한 뉴스 데이터를 읽어옵니다
df = pd.read_csv('C:/work/news_category_classfication/crawling_data_2/naver_news_titles_20241220_0925.csv')
df.drop_duplicates(inplace=True)  # 같은 내용의 뉴스는 하나만 남기고 제거합니다
df.reset_index(drop=True, inplace=True)  # 데이터 순서를 처음부터 다시 매깁니다
print(df.head())  # 데이터가 잘 불러와졌는지 앞부분 5개를 확인합니다
df.info()  # 전체적인 데이터 정보를 확인합니다
print(df.category.value_counts())  # 각 카테고리별로 뉴스가 몇 개씩 있는지 확인합니다

# 2. 데이터 분리
# 분석에 사용할 데이터(X:뉴스제목)와 예측하고자 하는 값(Y:카테고리)을 나눕니다
X = df['title']
Y = df['category']

# 3. 형태소 분석 테스트
# 첫 번째 뉴스 제목으로 형태소 분석이 잘 되는지 테스트합니다
print(X[0])  # 첫 번째 뉴스 제목을 출력해봅니다
okt = Okt()  # 한글 형태소 분석기를 준비합니다
okt_x = okt.morphs(X[0], stem=True)  # 첫 번째 제목을 단어 단위로 쪼개고 기본형으로 변환합니다
print(okt_x)  # 분석 결과를 확인합니다

# 4. 카테고리 데이터 전처리
# 문자로 된 카테고리를 숫자로 변환하는 작업을 합니다
encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)  # 카테고리를 0, 1, 2... 같은 숫자로 변환합니다
print(labeled_y[:3])  # 변환된 결과 확인

label = encoder.classes_  # 어떤 카테고리가 어떤 숫자가 되었는지 저장합니다
print(label)

# 나중에 또 사용할 수 있도록 변환기를 파일로 저장합니다
with open('./models/encoder.pickle', 'wb') as f:
   pickle.dump(encoder, f)

# 카테고리를 원-핫 인코딩으로 변환합니다 (예: [1,0,0,0] 같은 형태로)
onehot_Y = to_categorical(labeled_y)
print(onehot_Y)

# 5. 뉴스 제목 형태소 분석
# 모든 뉴스 제목을 단어 단위로 쪼갭니다
for i in range(len(X)):
   X[i] = okt.morphs(X[i], stem=True)
print(X)

# 6. 불용어 처리
# 분석에 도움이 되지 않는 단어들을 제거합니다
stopwords = pd.read_csv('./crawling_data/stopwords.csv', index_col=0)
print(stopwords)

# 불용어를 제거하고 의미있는 단어만 남깁니다
for sentence in range(len(X)):
   words = []
   for word in range(len(X[sentence])):
       if len(X[sentence][word]) > 1:  # 너무 짧은 단어는 제외
           if X[sentence][word] not in list(stopwords['stopword']):  # 불용어 제외
               words.append(X[sentence][word])
   X[sentence] = ' '.join(words)  # 단어들을 다시 하나의 문장으로 합칩니다
print(X[:5])

# 7. 텍스트를 숫자로 변환
# 컴퓨터가 이해할 수 있도록 텍스트를 숫자로 바꿉니다
token = Tokenizer()
token.fit_on_texts(X)  # 단어 사전을 만듭니다
tokened_X = token.texts_to_sequences(X)  # 단어를 사전에 따라 숫자로 변환합니다
wordsize = len(token.word_index) + 1  # 단어 사전의 크기를 저장합니다
print(wordsize)

print(tokened_X[:5])  # 변환된 결과를 확인합니다

# 8. 문장 길이 맞추기
# 모든 문장의 길이를 동일하게 맞춥니다
max = 0
for i in range(len(tokened_X)):  # 가장 긴 문장의 길이를 찾습니다
   if max < len(tokened_X[i]):
       max = len(tokened_X[i])
print(max)

# 모든 문장을 가장 긴 문장의 길이에 맞춰 늘립니다 (짧은 문장은 0으로 채움)
X_pad = pad_sequences(tokened_X, max)
print(X_pad)
print(len(X_pad[0]))

# 9. 학습용/테스트용 데이터 분리
# 모델 학습용 데이터와 성능 평가용 데이터를 나눕니다
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# 10. 전처리 완료된 데이터 저장
# 처리된 데이터를 파일로 저장하여 나중에 사용할 수 있게 합니다
np.save('./crawling_data/news_data_X_train_max_{}_wordsize_{}'.format(max, wordsize), X_train)
np.save('./crawling_data/news_data_Y_train_max_{}_wordsize_{}'.format(max, wordsize), Y_train)
np.save('./crawling_data/news_data_X_test_max_{}_wordsize_{}'.format(max, wordsize), X_test)
np.save('./crawling_data/news_data_Y_test_max_{}_wordsize_{}'.format(max, wordsize), Y_test)


