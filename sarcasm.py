#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:51:29 2020

@author: heisenbug
"""


import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 10000
embedding_dim = 16
max_length = 32
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
trainig_size = 20000

with open('/home/heisenbug/Workspace/TensorFlow/Lesson5/Sarcasm_Headlines_Dataset.json', 'r') as f:
    datastore = json.load(f)
    
sentences = []
labels = []
urls = []

for items in datastore:
    sentences.append(items['headline'])  
    labels.append(items['is_sarcastic'])
    urls.append(items['article_link'])
    
training_sentences = sentences[0 : trainig_size]
testing_sentences = sentences[trainig_size :]

training_labels = labels[0 : trainig_size]
testing_labels = labels[trainig_size :]

import numpy as np

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, padding = padding_type,
                                maxlen = max_length, truncating = trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, padding = padding_type,
                                maxlen = max_length, truncating = trunc_type)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# model.summary()
num_epochs = 30
model.fit(training_padded,
          training_labels,
          epochs = num_epochs,
          validation_data = (testing_padded, testing_labels),
          verbose = 2
         )

# import matplotlib.pyplot as plt

# def plot_graphs(history, string):
#     plt.plot(history.history[string])
#     plt.plot(history.history['val_' + string])
#     plt.xlabel('Epochs')
#     plt.ylabel(string)
#     plt.legend([string, 'val_' + string])
#     plt.show()
    
# plt_graph(history, 'acc')
# plt_graph(history, 'loss')