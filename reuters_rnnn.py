from keras.datasets import reuters
from time import time
from keras.models import Sequential
from keras.layers import Activation, Dense, SimpleRNN, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import TensorBoard
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz", num_words=10000)

x_train = pad_sequences(x_train, 600, padding='pre', truncating='pre')
x_test = pad_sequences(x_test, 600, padding='pre', truncating='pre')

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


tb = TensorBoard(log_dir="logs/{}".format(time()))
rnn = Sequential()
rnn.add(Embedding(10000, 8, input_length=600))
rnn.add(SimpleRNN(32))
rnn.add(Dense(1))
rnn.add(Activation('sigmoid'))

rnn.summary()
rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

rnn.fit(x_train, y_train, epochs=15, batch_size=50, callbacks=[tb])
score, accuracy = rnn.evaluate(x_test, y_test, verbose=0)
print('Test score', score)
print('Test accuracy', accuracy)
