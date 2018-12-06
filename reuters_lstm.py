from keras.datasets import reuters
from time import time
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, LSTM, Conv1D, MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import TensorBoard
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz", num_words=10000)

x_train = pad_sequences(x_train, 600, padding='pre', truncating='pre')
x_test = pad_sequences(x_test, 600, padding='pre', truncating='pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


tb = TensorBoard(log_dir="logs/lstm/{}".format(time()))
lstm_model = Sequential()
lstm_model.add(Embedding(10000, 8, input_length=600))
lstm_model.add(Dropout(.25))
lstm_model.add(Conv1D(64, 5, padding='valid', activation='relu'))
lstm_model.add(MaxPooling1D(pool_size=4))
lstm_model.add(LSTM(70))
lstm_model.add(Dense(46))
lstm_model.add(Activation('sigmoid'))

lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(x_train, y_train, epochs = 15, batch_size = 25, callbacks=[tb])

score, accuracy = lstm_model.evaluate(x_test, y_test, verbose = 0)