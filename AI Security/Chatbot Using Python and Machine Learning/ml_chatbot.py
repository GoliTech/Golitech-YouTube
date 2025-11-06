# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import json

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize text processing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Model parameters
max_words = 1000
max_len = 20
embedding_dim = 128
dense_units = 64
# Define training data
training_data = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": [
                "Hi", "Hello", "Hey", "Good morning", "Good afternoon",
                "Hi there", "Greetings", "How are you", "What's up", 
                "Hello there", "Hey there", "Nice to meet you", "Howdy",
                "Good day", "Hi friend", "Hello friend", "Hey buddy",
                "Morning", "Evening", "Afternoon", "Hi bot", "Hello bot"
            ],
            "responses": [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Hey! How may I assist you?"
            ]
        },
        {
            "tag": "goodbye",
            "patterns": [
                "Bye", "See you later", "Goodbye", "Take care",
                "I'm leaving", "See ya", "Later", "I have to go",
                "Talk to you later", "I'm off", "Have a good day",
                "Catch you later", "I'm out", "Till next time",
                "Farewell", "Ciao", "Adios", "Bye bye", "Got to go",
                "I need to leave", "Time to go"
            ],
            "responses": [
                "Goodbye! Have a great day!",
                "Bye! Come back soon.",
                "Take care! Hope to see you again."
            ]
        },
        {
            "tag": "thanks",
            "patterns": [
                "Thank you", "Thanks", "Thanks a lot", "Thanks for your help",
                "I appreciate it", "That's helpful", "Great, thanks",
                "Thank you so much", "Thanks for that", "Much appreciated",
                "That was helpful", "You're the best", "That helps a lot",
                "Perfect, thanks", "Thanks buddy", "Awesome, thanks",
                "You helped me", "Good job", "Well done", "I'm grateful",
                "Excellent help", "Thanks for the assistance"
            ],
            "responses": [
                "You're welcome!",
                "Happy to help!",
                "Any time!",
                "No problem at all."
            ]
        },
        {
            "tag": "help",
            "patterns": [
                "Help", "I need help", "Can you help me", "What can you do",
                "How does this work", "What are your features", "Assist me",
                "Show me what you can do", "Instructions please", "Guide me",
                "I'm confused", "I'm stuck", "What should I do", "How to use this",
                "Give me some help", "Help me out", "I need assistance",
                "How to", "Tell me how", "Explain how", "Show me", "Explain",
                "What is this", "How do I", "What does this do"
            ],
            "responses": [
                "I can help with answering questions, providing information, or just chatting.",
                "I'm a chatbot trained to respond to various types of queries.",
                "Ask me anything, and I'll try my best to help!"
            ]
        },
        {
            "tag": "capabilities",
            "patterns": [
                "What can you do", "What are your features", "What are you capable of",
                "Tell me about your abilities", "How can you help me", 
                "What tasks can you perform", "What are your functions",
                "What services do you offer", "How might you assist me",
                "Tell me about yourself", "What are you", "Who are you"
            ],
            "responses": [
                "I'm a simple chatbot that can recognize greetings, goodbyes, thanks, and help requests.",
                "I can understand basic intents and respond to them appropriately.",
                "I'm here to demonstrate basic natural language processing capabilities."
            ]
        },
        {
            "tag": "fallback",
            "patterns": [
                "1+1", "Calculate", "Math", "Science", "Politics", 
                "Tell me a joke", "Weather", "News", "Sports",
                "Random", "Whatever", "Anything", "IDK", "Test"
            ],
            "responses": [
                "I'm not trained to handle that type of request yet.",
                "That's beyond my current capabilities.",
                "I can only respond to basic conversation patterns right now."
            ]
        }
    ]
}

# Prepare text and labels from training data
texts = []
labels = []
responses = {}

# Add data augmentation by creating slight variations of patterns
def augment_pattern(pattern):
    augmented = []
    # Original pattern
    augmented.append(pattern)

    # Lowercase variant
    augmented.append(pattern.lower())

    # Add a common typo (if long enough)
    if len(pattern) > 5:
        # Swap two adjacent characters in the middle
        mid = len(pattern) // 2
        typo = pattern[:mid-1] + pattern[mid] + pattern[mid-1] + pattern[mid+1:]
        augmented.append(typo)

    return augmented

# Process and augment training data
for intent in training_data['intents']:
    responses[intent['tag']] = intent['responses']
    for pattern in intent['patterns']:
        augmented_patterns = augment_pattern(pattern)
        for aug_pattern in augmented_patterns:
            texts.append(aug_pattern)
            labels.append(intent['tag'])

# Text preprocessing function
def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())

    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]

    return ' '.join(tokens)

# Preprocess all texts
processed_texts = [preprocess_text(text) for text in texts]

# Create tokenizer
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(processed_texts)

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(processed_texts)

# Pad sequences
X = pad_sequences(sequences, maxlen=max_len)

# Convert labels to numerical format
unique_labels = list(set(labels))
label_dict = {label: i for i, label in enumerate(unique_labels)}
reverse_label_dict = {i: label for i, label in enumerate(unique_labels)}
y = np.array([label_dict[label] for label in labels])

# Build the neural network model
model = Sequential([
    Embedding(max_words, embedding_dim, input_length=max_len),
    Dropout(0.2),
    Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
    GlobalMaxPooling1D(),
    Dense(dense_units, activation='relu'),
    Dropout(0.3),
    Dense(len(unique_labels), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with early stopping
print("Training the model...")
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[callback]
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {accuracy:.4f}")

# Function to predict intent from text
def predict_intent(text):
    # Preprocess input
    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=max_len)

    # Get prediction
    prediction = model.predict(padded)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    # Get intent tag
    intent = reverse_label_dict[predicted_class]

    return intent, confidence

# Function to generate response based on predicted intent
def get_response(intent, confidence):
    if confidence < 0.4:  # Threshold based on observed performance
        return "I'm not sure I understand. Could you please rephrase that?"

    return random.choice(responses[intent])

# Main chat loop
print("Bot: Hello! How can I help you today? (type 'quit' to exit)")

while True:
    # Get user input
    user_input = input("You: ")

    # Exit condition
    if user_input.lower() == 'quit':
        print("Bot: Goodbye!")
        break

    # Get intent and confidence
    intent, confidence = predict_intent(user_input)

    # Get and display response
    response = get_response(intent, confidence)
    print(f"Bot: {response}")
