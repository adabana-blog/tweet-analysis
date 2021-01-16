from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Embedding, Conv1D, GlobalMaxPooling1D


class CNNModel:

    def __init__(self, input_dim, output_dim,
                 filters=250, kernel_size=3,
                 emb_dim=300,  embeddings=None, trainable=True):
        self.input = Input(shape=(None,), name="Input")
        if embeddings is None:
            self.embedding = Embedding(input_dim=input_dim,
                                       output_dim=emb_dim,
                                       trainable=trainable,
                                       name="embedding")
        else:
            self.embedding = Embedding(input_dim=embeddings.shape[0],
                                       output_dim=embeddings.shape[1],
                                       trainable=trainable,
                                       weights=[embeddings],
                                       name="embedding")
        self.conv = Conv1D(filters,
                           kernel_size,
                           padding="valid",
                           activation="relu",
                           strides=1)
        self.pool = GlobalMaxPooling1D()
        self.fc = Dense(output_dim, activation="sigmoid")

    def build(self):
        x = self.input
        embedding = self.embedding(x)
        conv = self.conv(embedding)
        pool = self.pool(conv)
        y = self.fc(pool)
        return Model(inputs=x, outputs=y)
