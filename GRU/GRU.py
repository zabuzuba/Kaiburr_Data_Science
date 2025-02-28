import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
strategy = tf.distribute.MirroredStrategy()  # Uses GPU if available, otherwise CPU

# Load dataset
df = pd.read_csv("/kaggle/input/jodiac/complaints.csv")

# Keep only necessary columns
df = df[['Consumer complaint narrative', 'Product']].dropna()

# Mapping product categories
df['Category'] = df['Product'].map({
    "Credit reporting, repair, or other": 0,
    "Debt collection": 1,
    "Consumer Loan": 2,
    "Mortgage": 3
})
df = df.dropna(subset=['Category'])  # Remove unmapped categories

# Text preprocessing
MAX_WORDS = 10000
MAX_LEN = 200
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Consumer complaint narrative'])
sequences = tokenizer.texts_to_sequences(df['Consumer complaint narrative'])
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
vocab_size = len(tokenizer.word_index) + 1

# Encode labels
label_encoder = LabelEncoder()
df['Category'] = label_encoder.fit_transform(df['Category'])
num_classes = len(df['Category'].unique())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['Category'], test_size=0.2, random_state=42)

# Define GRU model
EMBEDDING_DIM = 100
GRU_UNITS = 128
DROPOUT_RATE = 0.3
BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 1e-3

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    GRU(GRU_UNITS, return_sequences=True, recurrent_activation="sigmoid"),
    Dropout(DROPOUT_RATE),
    GRU(GRU_UNITS, recurrent_activation="sigmoid"),
    Dropout(DROPOUT_RATE),
    BatchNormalization(),
    Dense(64, activation="relu"),
    Dropout(DROPOUT_RATE),
    Dense(num_classes, activation="softmax")
])

optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

# Save model
model.save("consumer_complaint_classifier.h5")

# Test on sample statements
test_statements = [
    "My credit score dropped significantly due to incorrect information on my credit report.",
    "I am receiving constant harassment calls from a debt collector regarding a loan I never took.",
    "The interest rate on my personal loan was changed without prior notice.",
    "My mortgage application was denied even though I met all the eligibility criteria."
]

sequences = tokenizer.texts_to_sequences(test_statements)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding="post", truncating="post")
predictions = model.predict(padded_sequences)
predicted_classes = predictions.argmax(axis=1)

for i, statement in enumerate(test_statements):
    print(f"Statement: {statement}")
    print(f"Predicted Category: {predicted_classes[i]}\n")