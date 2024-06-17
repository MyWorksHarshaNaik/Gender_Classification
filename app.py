from flask_bootstrap import Bootstrap  # type: ignore
from flask import Flask, request, render_template  # noqa
import joblib

gender_app = Flask(__name__)
Bootstrap(gender_app)

# Load the CountVectorizer and model once during app initialization
cv = joblib.load("gender_vectorizer.pkl")
clf_1 = joblib.load("naivebayes.pkl")


@gender_app.route('/')
def index():
    return render_template('index.html')


@gender_app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        name_query = request.form['name_query']
        data = [name_query]
        vct = cv.transform(data).toarray()
        my_prediction = clf_1.predict(vct)
        return render_template('results.html', prediction=my_prediction[0], name=name_query.upper())  # noqa
    else:
        return render_template('index.html')


if __name__ == '__main__':
    gender_app.run(debug=True)
