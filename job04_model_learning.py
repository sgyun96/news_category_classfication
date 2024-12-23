# import numpy as np
# import matplotlib.pyplot as plt
# from keras.src.layers import Embedding
# from tensorflow.keras.models import *
# from keras.layers import *
#
# X_train = np.load( './crawling_data/news_data_X_train_max_14_wordsize_5536.npy', allow_pickle=True)
# X_test = np.load( './crawling_data/news_data_X_test_max_14_wordsize_5536.npy', allow_pickle=True)
# Y_train = np. load( './crawling_data/news_data_Y_train_max_14_wordsize_5536.npy', allow_pickle=True)
# Y_test = np.load( './crawling_data/news_data_Y_test_max_14_wordsize_5536.npy', allow_pickle=True)
# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)
#
# model = Sequential()
# model.add(Embedding(5536, 300, input_length=14))
# model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
# model.add(MaxPool1D(pool_size=1))
# model.add(GRU(128, activation='tanh', return_sequences=True))
# model.add(Dropout(0.3))
# model.add(GRU(64, activation='tanh', return_sequences=True))
# model.add(Dropout(0.3))
# model.add(GRU(64, activation='tanh'))
# model.add(Dropout(0.3))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(6, activation='softmax'))
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # 모델 컴파일
# model.summary()

"""여기부터 새 코드"""
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GRU, Dense, Dropout, Flatten
# from tensorflow.keras.models import Sequential
#
# X_train = np.load('./crawling_data/news_data_X_train_max_14_wordsize_5536.npy', allow_pickle=True)
# X_test = np.load('./crawling_data/news_data_X_test_max_14_wordsize_5536.npy', allow_pickle=True)
# Y_train = np.load('./crawling_data/news_data_Y_train_max_14_wordsize_5536.npy', allow_pickle=True)
# Y_test = np.load('./crawling_data/news_data_Y_test_max_14_wordsize_5536.npy', allow_pickle=True)
# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)
#
# model = Sequential()
# model.add(Embedding(5536, 300, input_length=14))
# model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=1))  # MaxPool1D를 MaxPooling1D로 수정
# model.add(GRU(128, activation='tanh', return_sequences=True))
# model.add(Dropout(0.3))
# model.add(GRU(64, activation='tanh', return_sequences=True))
# model.add(Dropout(0.3))
# model.add(GRU(64, activation='tanh'))
# model.add(Dropout(0.3))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(6, activation='softmax'))
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.summary()

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GRU, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

X_train = np.load('./crawling_data/news_data_X_train_14_wordsize_5536.npy', allow_pickle=True)
X_test = np.load('./crawling_data/news_data_X_test_14_wordsize_5536.npy', allow_pickle=True)
Y_train = np.load('./crawling_data/news_data_Y_train_14_wordsize_5536.npy', allow_pickle=True)
Y_test = np.load('./crawling_data/news_data_Y_test_14_wordsize_5536.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# 입력 형태를 명시적으로 지정
input_shape = (14,)  # 입력 길이가 14인 시퀀스

model = Sequential()
model.add(Input(shape=input_shape))
model.add(Embedding(5536, 300))
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(GRU(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128,
                   epochs=10, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test,verbose=0)
print('Final test set accuracy', score[1])
model.save('./models/news_category_classification_model_{}.h5'.format(
   fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()