from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model('sentiment_model.h5')

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
        .container { max-width: 600px; margin: auto; }
        h1 { color: #333; }
        textarea { width: 100%; height: 100px; margin-bottom: 10px; }
        button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        #result { margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sentiment Analysis</h1>
        <form method="post">
            <textarea name="text" placeholder="Enter your review here..."></textarea><br>
            <button type="submit">Analyze Sentiment</button>
        </form>
        {% if result %}
        <div id="result">
            Sentiment: {{ result }}
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        # Preprocess and vectorize the input
        vectorized_text = vectorizer.transform([text]).toarray()
        # Make prediction
        prediction = model.predict(vectorized_text)[0][0]
        result = 'Positive review' if prediction > 0.5 else 'Negative review'
    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == '__main__':
    app.run(debug=True)