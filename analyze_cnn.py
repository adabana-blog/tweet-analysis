from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from utils import filter_embeddings
from preprocessing import preprocess_dataset, build_vocabulary
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from model import CNNModel
import pickle
from janome.tokenizer import Tokenizer
import sys

batch_size = 128
epochs = 50
maxlen = 300
model_path = "model/cnn_model.h5"
num_words = 40000
# num_label = 5
num_label = 4


def predict(text):
    wv = KeyedVectors.load("model/word2vec.model", mmap='r')
    model = load_model("model/cnn_model.h5")
    with open('tokenizer.pickle', 'rb') as handle:
        vocab = pickle.load(handle)

    # t = Tokenizer(wakati=True)
    # tokenized_text = " ".join(t.tokenize(str(text)))
    text = [text]
    tokenized_text = preprocess_dataset(text)
    text_sequences = vocab.texts_to_sequences(tokenized_text)
    text_sequences = pad_sequences(text_sequences, maxlen=maxlen, truncating="post")
    print("--------------------------------------------")
    print(text_sequences.shape)
    print("--------------------------------------------")
    print(tokenized_text)
    print("--------------------------------------------")
    return model.predict(text_sequences)


def train():
    df_tweets = pd.read_csv("data/df_tweets", index_col=0)
    df_tweets["text"] = preprocess_dataset(df_tweets["text"])
    df_tweets = df_tweets.dropna(how='any')
    df_tweets = df_tweets.drop(df_tweets.index[df_tweets["Irrelevant"] == 1])

    x = df_tweets["text"]
    # y = df_tweets[["posi_and_nega", "posi", "nega", "neutral", "Irrelevant"]]
    y = df_tweets[["posi_and_nega", "posi", "nega", "neutral"]]
    y = np.asarray(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    vocab = build_vocabulary(x_train, num_words)
    with open('model/tokenizer.pickle', 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)
    x_train = vocab.texts_to_sequences(x_train)
    x_test = vocab.texts_to_sequences(x_test)
    x_train = pad_sequences(x_train, maxlen=maxlen, truncating="post")
    x_test = pad_sequences(x_test, maxlen=maxlen, truncating="post")

    wv = KeyedVectors.load("model/word2vec.model", mmap='r')
    wv = filter_embeddings(wv, vocab.word_index, num_words)

    model = CNNModel(num_words, num_label, embeddings=wv).build()
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["acc"])

    callbakcs = [
        EarlyStopping(patience=3),
        ModelCheckpoint(model_path, save_best_only=True)
    ]

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.5,
        callbacks=callbakcs,
        shuffle=True
    )

    # test_texts = vocab.sequences_to_texts(x_test)
    # pred_proba = model.predict(x_test_multi)


def main(mode):
    print(sys.argv[1])
    if mode == "train":
        train()
    elif mode == "pred":
        proba = predict("ライブ配信ありがとうございました目が笑っている笑顔")
        print(proba)


if __name__ == "__main__":
    main(sys.argv[1])
