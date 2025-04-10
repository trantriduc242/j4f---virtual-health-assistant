from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import pickle
import numpy as np

# Load the model and supporting files
model = tf.keras.models.load_model('chat_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'response': 'No message received.'})

    # Preprocess user input
    sequences = tokenizer.texts_to_sequences([user_message])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=20)

    # Predict intent
    prediction = model.predict(padded)
    predicted_label = labels[np.argmax(prediction)]

    return jsonify({'response': f"I detected your intent as: {predicted_label}."})

if __name__ == '__main__':
    app.run(debug=True)