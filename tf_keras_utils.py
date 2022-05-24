#!/usr/bin/env python

"""
Keras has a lovely API for sequential vectorization of text_strings.
This module is designed to utilise those possibilitites. And save vectorized
train and test_data.
"""
import load_data
import tensorflow as tf
import numpy as np
import load_data
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing import text
from sklearn.feature_extraction.text import TfidfVectorizer


def sequence_vectorize(train_texts, val_texts):
    """Vectorizes texts as sequence vectors.

    1 text = 1 sequence vector with fixed length.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index

import sys

n_arg = len(sys.argv) - 1
print("Total arguments passed:", n_arg)

if n_arg > 0:
    INPUT_DATA_DIR = sys.argv[1]
else:
    INPUT_DATA_DIR = "./data/cases_small_preprocessedLEMM"

    
OUTPUT_DATA_DIR = f"{INPUT_DATA_DIR}_vectorized"
load_data.mkdir(OUTPUT_DATA_DIR)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'


# Limit on the length of text sequences. Sequences longer than this
# will be truncated.
MAX_SEQUENCE_LENGTH = 1024


X = []
Y = []

MIN_TOKENS = 200
for i, (text_, y) in enumerate(load_data.yield_preprocessed_json_from_disk(INPUT_DATA_DIR)):
    if len(text_.split()) > MIN_TOKENS:
        Y.append(y)
        X.append(text_)


# shuffling data to break  storage order patterns
np.random.seed(0)
idx = np.arange(len(X))
np.random.shuffle(idx)
X = [X[j] for j in idx]
Y = [Y[j] for j in idx]

cases_info = pd.read_csv('data/cases_info.csv')  # mapping case_numbers to category_ids
cases_d = dict(zip(cases_info['Number'], cases_info['CategoryID']))
del cases_info

Y_id = np.array(list(map(lambda case: cases_d[case], Y)))


skip_indices = set(np.where(Y_id == -1)[0])  # -1 was used for non labeled casenumbers
keep_mask = (Y_id != -1)

id2i = dict(zip(set(Y_id[keep_mask]), range(len(set(Y_id[keep_mask])))))
i2id = {v:k for k, v in id2i.items()}

X = [x for i, x in enumerate(X) if i not in skip_indices]
Y = [y for i, y in enumerate(Y) if i not in skip_indices]
Y_id = [y_id for i, y_id in enumerate(Y_id) if i not in skip_indices]

Y_idx = np.array(list(map(lambda y_id: id2i[y_id], Y_id)))


np.random.seed(0)
DS_SIZE = len(Y_idx)
TEST_SIZE = .2
indices = np.arange(DS_SIZE)
train_indices, test_indices = train_test_split(indices, test_size=TEST_SIZE, stratify=Y_idx)


X_train_list = []
X_test_list = []
for train_idx in train_indices:
    X_train_list.append(X[train_idx])
for test_idx in test_indices:
    X_test_list.append(X[test_idx])


print("FEATING TEXT DATA TO VECTORIZE")


x_train, x_val, word_index = sequence_vectorize(X_train_list, X_test_list)

print("SAVING VECTORIZED DATA")
np.save(os.path.join(OUTPUT_DATA_DIR, "indices_train.npy"), train_indices)
np.save(os.path.join(OUTPUT_DATA_DIR, "indices_test.npy"), test_indices)

np.save(os.path.join(OUTPUT_DATA_DIR, "x_train.npy"), x_train)
np.save(os.path.join(OUTPUT_DATA_DIR, "x_test.npy"), x_val)

np.save(os.path.join(OUTPUT_DATA_DIR, "y_train.npy"), Y_idx[train_indices])
np.save(os.path.join(OUTPUT_DATA_DIR, "y_test.npy"), Y_idx[test_indices])

with open(os.path.join(OUTPUT_DATA_DIR, "word_index.json"), 'w') as f:
    json.dump(word_index, f)
