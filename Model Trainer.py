# conda activate tritter

# pandas
import pandas as pd

# nltk
import nltk
from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Word2vec
import gensim

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dropout, LSTM, Dense
from keras.models import Sequential, load_model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Utility
import re
import numpy as np
import time
import pickle


nltk.download('stopwords')


trumpTweets = 'output.csv'
otherTweets = 'training.1600000.processed.noemoticon.csv'

DATASET_ENCODING = "ISO-8859-1"
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
TRAIN_SIZE = 0.8

# TEXT CLEANING
TWITTERPIC_RE = 'pic.twitter.com.*'
HASHTAG_RE = '# \w+'
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"


def preprocess(text):
    # Remove link, user, and special characters
    text = re.sub(TWITTERPIC_RE, ' ', str(text).lower()).strip()
    text = re.sub(HASHTAG_RE, ' ', str(text).lower()).strip()
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = [token for token in text.split() if token not in stop_words]
    
    return " ".join(tokens)



def main()
    with open(trumpTweets, encoding="utf-8") as f:
        linelist = f.readlines()

    fileLineList = [i for i in linelist if i.count(';') == 9]
    fileLineList = [i.split(';') for i in fileLineList]

    # Read in and set headers
    dfTr = pd.DataFrame(fileLineList)
    new_header = dfTr.iloc[0]
    dfTr = dfTr[1:]
    dfTr.columns = new_header

    # clean data
    dfTr.text = dfTr.text.str.strip('"')
    dfTr = dfTr.drop(['retweets', 'favorites', 'geo', 'mentions', 'hashtags', 'id', 'permalink\n'], axis=1)
    dfTr = dfTr.assign(username='Trump')
    dfTr = dfTr[~dfTr.text.str.contains('@ realDonaldTrump')]


    # dfOth.columns
    dfOth = pd.read_csv(otherTweets, encoding=DATASET_ENCODING , names=DATASET_COLUMNS)
    dfOth = dfOth.drop(['target', 'ids', 'flag'], axis=1)
    dfOth = dfOth.rename(columns={'user': 'username'})
    dfOth = dfOth.assign(username='Other')
    dfOth = dfOth[['username', 'date', 'text']]


    dfAll = pd.concat([dfTr, dfOth])

    stop_words = stopwords.words('english')
    dfAll.text = dfAll.text.apply(lambda x: preprocess(x))


    df_train, df_test = train_test_split(dfAll, test_size=1-TRAIN_SIZE, random_state=42)
    print("TRAIN size:", len(df_train))
    print("TEST size:", len(df_test))


    documents = [_text.split() for _text in df_train.text]

    # https://stackoverflow.com/questions/53343027/gensim-on-windows-c-extension-not-loaded-training-will-be-slow
    w2v_model = gensim.models.word2vec.Word2Vec(
        size=W2V_SIZE, 
        window=W2V_WINDOW, 
        min_count=W2V_MIN_COUNT, 
        workers=8)

    w2v_model.build_vocab(documents)

    words = w2v_model.wv.vocab.keys()
    vocab_size = len(words)
    print("Vocab size", vocab_size)


    w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)


    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(df_train.text)

    vocab_size = len(tokenizer.word_index) + 1
    print("Total words", vocab_size)

    x_train = pad_sequences(tokenizer.texts_to_sequences(df_train.text), maxlen=SEQUENCE_LENGTH)
    x_test = pad_sequences(tokenizer.texts_to_sequences(df_test.text), maxlen=SEQUENCE_LENGTH)


    encoder = LabelEncoder()
    encoder.fit(df_train.username.tolist())

    y_train = encoder.transform(df_train.username.tolist())
    y_test = encoder.transform(df_test.username.tolist())

    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    print("y_train",y_train.shape)
    print("y_test",y_test.shape)


    embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
            
    print(embedding_matrix.shape)


    embedding_layer = Embedding(
        vocab_size, 
        W2V_SIZE, 
        weights=[embedding_matrix], 
        input_length=SEQUENCE_LENGTH,
        trainable=False)



    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()



    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])


    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
        EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

    history = model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        verbose=1,callbacks=callbacks)

    score = model.evaluate(
        x_test, y_test, 
        batch_size=BATCH_SIZE)
    print("ACCURACY:",score[1])
    print("LOSS:",score[0])


    model.save(KERAS_MODEL)
    w2v_model.save(WORD2VEC_MODEL)
    pickle.dump(tokenizer, open(TOKENIZER_MODEL, "wb"), protocol=0)
    pickle.dump(encoder, open(ENCODER_MODEL, "wb"), protocol=0)



if __name__ == "__main__":
    main()