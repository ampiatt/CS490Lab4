from time import time
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense, SimpleRNN, Embedding
from keras.callbacks import TensorBoard
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import TensorBoard
import numpy

#read in data
trainingdata = pd.read_csv('trainingdata.csv', encoding='latin1')
testingdata = pd.read_csv('testdata.csv', encoding='latin1')

# reorder the data so that it mixes up the rows (keeps columns together
trainingdata = trainingdata.sample(frac=1).reset_index(drop=True)
testingdata = testingdata.sample(frac=1).reset_index(drop=True)

# tokenize data
train_data = numpy.array(trainingdata.iloc[:, 5].values)
test_data = numpy.array(testingdata.iloc[:, 5].values)
tokenize = Tokenizer(10000)
tokenize.fit_on_texts(train_data)
tokenize.fit_on_texts(test_data)
seq_train = tokenize.texts_to_sequences(train_data)
seq_test = tokenize.texts_to_sequences(test_data)

# pad or truncate the data to standardize tweet length
x_train = pad_sequences(seq_train, 50, padding='pre', truncating='pre')
x_test = pad_sequences(seq_test, 50, padding='pre', truncating='pre')

# store classification results in array
y_train = numpy.array(trainingdata.iloc[:,0].values)
y_test = numpy.array(testingdata.iloc[:,0].values)

shape = x_train.shape

tb = TensorBoard(log_dir="logs/{}".format(time()))
rnn = Sequential()
rnn.add(Embedding(10000, 8, input_length=50))
rnn.add(SimpleRNN(32))
rnn.add(Dense(1))
rnn.add(Activation('sigmoid'))

rnn.summary()
rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

rnn.fit(x_train, y_train, epochs = 15, batch_size = 500, callbacks=[tb])
# score, accuracy = rnn.evaluate(x_test, y_test, verbose = 0)
# print('Test score', score)
# print('Test accuracy', accuracy)