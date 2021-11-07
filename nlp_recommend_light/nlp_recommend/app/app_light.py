from flask import Flask, request, render_template, jsonify
# from flask_sqlalchemy import SQLAlchemy
import dill
import os

import sys

CUR_DIR = os.path.dirname(__file__) # app folder
PARENT_DIR = os.path.dirname(os.path.dirname(CUR_DIR)) # nlp_recommend
MODEL_DIR = os.path.join(os.path.dirname(PARENT_DIR), 'training')

sys.path.insert(0, PARENT_DIR)

# from nlp_recommend.utils.utils import rerank

# start flask
app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'

# print('loading', os.path.join(MODEL_DIR, 'models/psychology_container.pkl'), CUR_DIR)
model_psycho = dill.load(open(os.path.join(MODEL_DIR, 'models/adventure_container_verylight.pkl'), 'rb'))
model_philo = model_adv = model_psycho

# conversational_pipeline = pipeline("conversational")
# conv = Conversation("Hey what's up?")
# conversational_pipeline([conv])

# print('loaded philo!', model_philo.warper.predict(model_philo.model, 'test'))
# print('loaded psycho!', model_psycho.warper.predict(model_psycho.model, 'test'))
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
    default_preds = {'title':None, 'quote':None, 'author':None}
    default_ans = 'What\'s up today?'
    return render_template('index.html', philo_preds=default_preds, 
                            psycho_preds=default_preds, adv_preds=default_preds, ans=default_ans)

# @app.route('/')
# def get_history():
#     searchs = InputAnswer.query.all()
#     history = [{'input': s.input, 'answer':s.answer} for s in searchs]
#     return render_template('index.html', history=history)

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
    # Predict books
    preds = {}
    for name, model in models.items():
        preds_raw, title_preds = get_predictions(text, model)
        preds[name] = {key: elt.capitalize() for key, elt in preds_raw.items() if elt}
        print('title_preds', title_preds)
        preds[name]['recommended_title'] = title_preds

    # Answer the user
    # conv.add_user_input(text)
    # ans = conversational_pipeline([conv]).generated_responses[-1]
    ans = 'I see, but what do you mean by that?'

    # Add to database
    # to_store = InputAnswer(user=user, author=author, input=text, answer=quote)
    # db.session.add(to_store)
    # db.session.commit()
    # # load database
    # history = update_history()
    return jsonify(input=text, 
                            philo_preds=preds['philo'], 
                            psycho_preds=preds['psycho'],
                            adv_preds=preds['adv'])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)