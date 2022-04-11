from dataPrep import prepData, unique_words
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv

from tensorflow.keras import layers
from tensorflow.keras import losses

labels, data = prepData('subtask1-heterographic-test.txt', 'subtask1-heterographic-test.xml')
labels2, data2 = prepData('subtask1-homographic-test.txt', 'subtask1-homographic-test.xml')

labels = labels + labels2
data = data + data2

questions = []
new_labels = []
with open('JEOPARDY_CSV.csv', 'r', encoding="utf8") as csvfile:
    csvreader = csv.reader(csvfile)
    i = 0
    for row in csvreader:
        if i >= 5000:
            break
        i += 1
        questions.append(row[5])
        new_labels.append(1)

train_labels = labels[:3224] + new_labels[:4000]
test_labels = labels[3224:] + new_labels[4000:]

train_data = data[:3224] + questions[:4000]
test_data = data[3224:] + questions[4000:]

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

train_data = np.array(train_data)
test_data = np.array(test_data)

num_unique = unique_words(train_data)

#tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=num_unique)
tokenizer.fit_on_texts(train_data)

word_index = tokenizer.word_index

#convert text to tokenized sequence
train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(test_data)

#padding
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_length = 20

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding="post", truncating="post")

reverse_word_index = dict([(idx, word) for (word, idx) in word_index.items()])

def decode(sequence):
    return " ".join([reverse_word_index.get(idx, "?") for idx in sequence])

#Building model
model = keras.models.Sequential()
model.add(layers.Embedding(num_unique, 32, input_length=max_length))
model.add(layers.LSTM(64, dropout=.01))
model.add(layers.Dense(1, activation="sigmoid"))

loss = keras.losses.BinaryCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(learning_rate=.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

model.fit(train_padded, train_labels, epochs=20, validation_data=(test_padded, test_labels), verbose=2)

model.save("C:/Users/Epicr/OneDrive/Desktop/coding/punProject/punModel/")

