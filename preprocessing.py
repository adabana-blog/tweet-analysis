import re
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
from janome.tokenizer import Tokenizer
from tensorflow.keras.preprocessing.sequence import skipgrams, make_sampling_table
t = Tokenizer(wakati=True)


def tokenize(text):
    return t.tokenize(text)


def tokenize_base_form(text):
    tokens = [token.base_form for token in t.tokenize(text)]
    return tokens


def normalize_number(text, reduce=False):
    if reduce:
        normalize_text = re.sub(r"\d+", "0", text)
    else:
        normalize_text = re.sub(r"\d", "0", text)
    return normalize_text


def build_vocabulary(texts, num_words=None):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=num_words, oov_token="<UNK>"
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def create_dataset(text, vocab, num_words, window_size, negative_samples):
    data = vocab.texts_to_sequences([text]).pop()
    sampling_table = make_sampling_table(num_words)
    couples, labels = skipgrams(data, num_words,
                                window_size=window_size,
                                negative_samples=negative_samples,
                                sampling_table=sampling_table)
    word_target, word_context = zip(*couples)
    word_target = np.reshape(word_target, (-1, 1))
    word_context = np.reshape(word_context, (-1, 1))
    labels = np.asarray(labels)
    return [word_target, word_context], labels


def preprocess_dataset(texts):
    texts = [" ".join(tokenize(str(text))) for text in texts]
    return texts
