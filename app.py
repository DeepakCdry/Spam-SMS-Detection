from flask import Flask, request, jsonify, render_template
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os

app = Flask(__name__)

# --- NLTK Downloads ---
# Ensure 'punkt' tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' not found, downloading...")
    nltk.download('punkt')

# Ensure 'stopwords' corpus is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK 'stopwords' not found, downloading...")
    nltk.download('stopwords')

# --- Load Model & Vectorizer ---
# Initialize tfidf and model as None, to be loaded later
tfidf = None
model = None

try:
    # Attempt to load the TF-IDF vectorizer from 'vectorizer.pkl'
    # 'rb' mode is for reading in binary
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    # Attempt to load the trained machine learning model from 'model.pkl'
    model = pickle.load(open('model.pkl', 'rb'))
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    # If the files are not found, print an error message
    print(f"Error loading model/vectorizer: {e}. Make sure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
except Exception as e:
    # Catch any other potential errors during loading
    print(f"An unexpected error occurred during model/vectorizer loading: {e}")

# --- Text Preprocessing Function ---
# Initialize Porter Stemmer, used for reducing words to their root form
ps = PorterStemmer()

def transform_text(text):
    """
    Transforms raw text by performing the following steps:
    1. Lowercasing
    2. Tokenization (splitting text into words)
    3. Removing non-alphanumeric characters
    4. Removing English stopwords
    5. Applying stemming (reducing words to their root form)

    Args:
        text (str): The input SMS message.

    Returns:
        str: The transformed text, ready for vectorization.
    """
    text = text.lower()  # Convert text to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text into a list of words

    y = []
    for i in text:
        # Keep only alphanumeric characters and remove stopwords
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))  # Apply stemming and add to list

    return " ".join(y)  # Join the processed words back into a single string

# --- Routes ---
@app.route('/')
def home():
    """
    Renders the home page of the application.
    This is the entry point for the user interface.
    """
    # Render 'index.html' template. 'message' is passed as an empty string
    # initially so the textarea doesn't show 'None' if accessed before first submission.
    return render_template('index.html', message="")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the prediction request when a user submits an SMS message.
    It preprocesses the message, vectorizes it, and uses the loaded model
    to predict if it's spam or not.
    """
    # Check if the model and vectorizer were loaded successfully
    if tfidf is None or model is None:
        return render_template('index.html', prediction_text="Error: Model not loaded properly. Please check server logs.", message=request.form.get('message', ''))

    # Get the message from the HTML form submission
    message = request.form.get('message')

    # If no message was provided, prompt the user
    if not message:
        return render_template('index.html', prediction_text="Please enter a message to predict.", message="")

    try:
        # Preprocess the input message using the defined function
        transformed_msg = transform_text(message)
        # Transform the preprocessed message into a TF-IDF vector
        # tfidf.transform expects an iterable (like a list) of strings
        vector_input = tfidf.transform([transformed_msg]).toarray()

        # Make a prediction using the loaded machine learning model
        # [0] is used because model.predict returns an array
        result = model.predict(vector_input)[0]

        # Determine the output string based on the prediction result (0 for ham, 1 for spam)
        output = "Spam" if result == 1 else "Not Spam"

        # Render the 'index.html' template again, passing the prediction result
        # and crucially, the original message back to keep it in the textarea.
        return render_template('index.html', prediction_text=f'Prediction: {output}', message=message)

    except Exception as e:
        # Catch any errors during prediction (e.g., issues with text transformation)
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text=f"An error occurred during prediction: {e}", message=message)

# --- Run App ---
if __name__ == '__main__':
    # Get the port from environment variable (for deployment) or default to 10000
    port = int(os.environ.get('PORT', 10000))
    # Run the Flask application
    # host='0.0.0.0' makes the app accessible externally (e.g., in a Docker container)
    app.run(host='0.0.0.0', port=port, debug=True) # debug=True for development, set to False in production
