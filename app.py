from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the vectorizer and model
with open('vectorizer.plk', 'rb') as f:
    vectorizer = pickle.load(f)

with open('Naive_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=["GET", "POST"])
def main_function():
    if request.method == "POST":
        text = request.form.get('email', '')
        
        # Transform the input using the vectorizer
        vectorized_text = vectorizer.transform([text])
        
        # Make a prediction
        prediction = model.predict(vectorized_text)[0]
        
        # Convert prediction to a readable format
        prediction_text = "Spam" if prediction == 1 else "Not Spam"
        
        return render_template("show.html", prediction=prediction_text)
    
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
