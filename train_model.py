import tensorflow as tf
import numpy as np
import pickle

# Training data
sentences = [
    "hello", "hi", "good morning", "good evening",
    "I have a headache", "my head hurts", "fever and chills",
    "I am coughing a lot", "bad cough", "dry cough"
]

labels = [
    "greeting", "greeting", "greeting", "greeting",
    "headache", "headache", "fever",
    "cough", "cough", "cough"
]

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=20)

# Encode labels
label_set = sorted(list(set(labels)))
label_index = {label: idx for idx, label in enumerate(label_set)}
indexed_labels = np.array([label_index[label] for label in labels])

# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=16, input_length=20),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(label_set), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(padded_sequences, indexed_labels, epochs=50)

# Save model and tokenizer
model.save('chat_model.h5')

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('labels.pkl', 'wb') as f:
    pickle.dump(label_set, f)

print("✅ Model, tokenizer, and labels saved!")
