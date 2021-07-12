from flask import Flask, request, render_template 
from flask import redirect, url_for
from flask_sqlalchemy import SQLAlchemy

import sys
sys.path.insert(0, '/home/bettyld/PJ/Documents/NLP_PJ/philo')

from model_deployment.tfidf import TfidModel

LAST_N_SEARCH = 20

# start flask
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
db = SQLAlchemy(app)
#  db.create_all()

class InputAnswer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(20))
    input = db.Column(db.String(80), nullable=False)
    answer = db.Column(db.String(120))
    author = db.Column(db.String(20))

# @app.route('/')
# def get_history():
#     searchs = InputAnswer.query.all()
#     history = [{'input': s.input, 'answer':s.answer} for s in searchs]
#     return render_template('index.html', history=history)


# @app.route('/', methods=['POST'])
# def receive_data():
#     user = request.form['fname']
#     text = request.form['thoughts']
#     db[user] = text
#     # do something with your text
    def __repr__(self):
        return f'{self.user}: {self.input} - {self.answer}'

model = TfidModel()

def update_history():
    searchs = InputAnswer.query.all()
    history = [{'author':s.author,
                'user':s.user,
                'input': s.input,
                'answer':s.answer} for s in searchs[-LAST_N_SEARCH:]]
    return history

@app.route('/')
def home():
    history = update_history()
    return render_template('index.html', history=history)

# @app.route('/')
# def get_history():
#     searchs = InputAnswer.query.all()
#     history = [{'input': s.input, 'answer':s.answer} for s in searchs]
#     return render_template('index.html', history=history)


# @app.route('/', methods=['POST'])
# def receive_data():
#     user = request.form['fname']
#     text = request.form['thoughts']
#     db[user] = text
#     # do something with your text
#     # return redirect(url_for('success', name=user))
#     return render_template('index.html', waiting='Looking for a match...')
 

# @app.route('/success/<name>')
# def success(name):
#     text = db[name]
#     best_res = test_sentence(text)[0]
#     print(best_res)
#     quote, author = best_res[0], best_res[1]
#     return "<xmp>" + str(quote) + str(author)+ " </xmp> "


@app.route('/', methods=['POST'])
def predict():
    """
    To make a prediction on one sample of the text
    satire or fake news
    :return: a result of prediction in HTML page
    """
    # Receive
    user = request.form['fname']
    text = request.form['thoughts']
    user= str(user).capitalize()
    # Predict
    best_res = model.test_sentence(text)[0]
    quote, author = best_res[0], best_res[1]

    # Add to database
    to_store = InputAnswer(user=user, author=author, input=text, answer=quote)
    db.session.add(to_store)
    db.session.commit()
    # load database
    history = update_history()

    return render_template('index.html', input=text, author=author, answer=quote, history=history)
