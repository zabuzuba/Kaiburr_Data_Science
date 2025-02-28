import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
strategy = tf.distribute.MirroredStrategy()  # Uses GPU if available, otherwise CPU

# Load dataset
df = pd.read_csv("/kaggle/input/jodiac/complaints.csv")

# Select relevant columns
df = df[["Consumer complaint narrative", "Product"]].dropna()

# Map categories
df["Product"] = df["Product"].map({
    "Credit reporting, repair, or other": 0,
    "Debt collection": 1,
    "Consumer Loan": 2,
    "Mortgage": 3
}).dropna()

# Text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text

df["Clean_Text"] = df["Consumer complaint narrative"].apply(clean_text)

# Tokenization & Padding
MAX_WORDS = 10000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df["Clean_Text"])

sequences = tokenizer.texts_to_sequences(df["Clean_Text"])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")

# Label Encoding
label_encoder = LabelEncoder()
df["Category"] = label_encoder.fit_transform(df["Product"])
num_classes = len(df["Category"].unique())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df["Category"], test_size=0.2, random_state=42)

# Build BiLSTM model
EMBEDDING_DIM = 100
LSTM_UNITS = 128
DROPOUT_RATE = 0.3
BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 1e-3

model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
    Dropout(DROPOUT_RATE),
    Bidirectional(LSTM(LSTM_UNITS)),
    Dropout(DROPOUT_RATE),
    BatchNormalization(),
    Dense(64, activation="relu"),
    Dropout(DROPOUT_RATE),
    Dense(num_classes, activation="softmax")
])

# Compile model
model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=["accuracy"])

# Train model
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# Evaluate model
y_pred = np.argmax(model.predict(X_test), axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Prediction function
def predict_category(text):
    cleaned_text = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    pred = np.argmax(model.predict(padded))
    return label_encoder.inverse_transform([pred])[0]

# Example prediction
example_text = "I have a complaint about my mortgage loan charges."
print("Predicted Category:", predict_category(example_text))
