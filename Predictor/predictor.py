# Keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import input

# MODEL FILENAMES
KERAS_MODEL = "model2.h5"
TOKENIZER_MODEL = "tokenizer2.pkl"

# KERAS
SEQUENCE_LENGTH = 300

model = load_model(KERAS_MODEL)
with open(TOKENIZER_MODEL, "rb") as f:
    tokenizer = pickle.load(f)

def predict(text):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    return float(model.predict([x_test])[0])

def main():
    tweet = input("enter sentence")
    score = predict(tweet)
    score_percent = score * 100
    print("this sentence is {score}% trump".format({'score': score_percent}))

if __name__ == "__main__":
    main()