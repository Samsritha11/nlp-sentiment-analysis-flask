from flask import Flask, render_template, request
import pickle

# Load saved model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    data = [review]
    transformed = vectorizer.transform(data)
    prediction = model.predict(transformed)[0]
    return render_template('index.html', review=review, sentiment=prediction)

if __name__ == '__main__':
    app.run(debug=True)
