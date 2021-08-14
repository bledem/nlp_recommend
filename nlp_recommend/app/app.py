from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
import dill
import os

CUR_DIR = os.path.dirname(__file__) # app folder
PARENT_DIR = os.path.dirname(os.path.dirname(CUR_DIR)) # nlp_recommend
MODEL_DIR = os.path.dirname(PARENT_DIR)

import sys
sys.path.insert(0, PARENT_DIR)

from nlp_recommend.models.container import Container


# start flask
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'

print('loading', os.path.join(PARENT_DIR, 'models/psychology_container.pkl'), CUR_DIR)
model_psycho  = dill.load(open(os.path.join(PARENT_DIR, 'models/psychology_container.pkl'), 'rb'))
model_philo = dill.load(open(os.path.join(PARENT_DIR, 'models/philosophy_container_light.pkl'), 'rb'))

print('loaded philo!', model_philo.warper.predict(model_philo.model, 'test'))
print('loaded psycho!', model_psycho.warper.predict(model_psycho.model, 'test'))
print('---> Go into your browser at http://0.0.0.0:5000 <---')
# db = SQLAlchemy(app)
#  db.create_all()
# models = {
#         'philosophy': {'model': TfIdfModel(dataset='philosophy'),   
#                        'cls':SentimentCls(dataset='philosophy'), 'warper': Warper(dataset='philosophy')},
#         'psychology': {'model': CombinedModel(dataset='psychology'),   
#                        'cls':SentimentCls(dataset='psychology'), 'warper': Warper(dataset='psychology')}
#                        }
# class InputAnswer(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user = db.Column(db.String(20))
#     input = db.Column(db.String(80), nullable=False)
#     answer = db.Column(db.String(120))
#     author = db.Column(db.String(20))

# @app.route('/')
# def get_history():
#     searchs = InputAnswer.query.all()
#     history = [{'input': s.input, 'answer':s.answer} for s in searchs]
#     return render_template('index.html', history=history)

# def update_history():
#     searchs = InputAnswer.query.all()
#     history = [{'author': s.author,
#                 'user': s.user,
#                 'input': s.input,
#                 'answer': s.answer} for s in searchs[-LAST_N_SEARCH:]]
#     return history


@app.route('/')
def home():
    # history = update_history()
    # return render_template('index.html', history=history)
    print('RENDERING')
    default_preds = {'title':None, 'quote':None, 'author':None}

    return render_template('index.html', philo_preds=default_preds, 
                            psycho_preds=default_preds,)

# @app.route('/')
# def get_history():
#     searchs = InputAnswer.query.all()
#     history = [{'input': s.input, 'answer':s.answer} for s in searchs]
#     return render_template('index.html', history=history)


def parse_preds(result, warped):
    pred = {'title':None, 'quote':None, 'author':None}
    if len(result)>0:
            pred['title'] = result.title.values[0]
            pred['quote'] = result.sentence.values[0]
            pred['author']= result.author.values[0]
            print('warped',warped)
            if warped['before'] is not None or warped['after'] is not None:
                print('DEBUG', warped['before'][0:5])
                pred['before'] = ' .'.join(warped['before'])
                pred['after'] = ' .'.join(warped['after'])
    return pred

def get_predictions(text, container):
    best_index, warped_dict = container.warper.predict(
        container.model, text, return_index=True)
    best_index, sentiment = container.cls.match_filter(text, best_index)
    result = container.warper.corpus[['sentence', 'author', 'title']].iloc[best_index]
    preds = parse_preds(result, warped_dict)
    return preds

@app.route('/', methods=['POST'])
def predict():
    """
    To make a prediction on one sample of the text
    satire or fake news
    :return: a result of prediction in HTML page
    """
    # Receive
    # dataset = request.form['dataset']
    # user = request.form['fname']
    text = request.form['thoughts']
    # user = str(user).capitalize()
    # Predict
    
    #Philo
    philo_preds = get_predictions(text, model_philo)
    print('philo_pred', philo_preds)
    philo_preds = {key: elt.capitalize() for key, elt in philo_preds.items() if elt}
   
    #Psycho
    psycho_preds = get_predictions(text, model_psycho)
    psycho_preds = {key: elt.capitalize() for key, elt in psycho_preds.items() if elt}
    # Add to database
    # to_store = InputAnswer(user=user, author=author, input=text, answer=quote)
    # db.session.add(to_store)
    # db.session.commit()
    # # load database
    # history = update_history()

    return render_template('index.html', input=text, 
                            philo_preds=philo_preds, 
                            psycho_preds=psycho_preds,
                            )#, history=history)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)