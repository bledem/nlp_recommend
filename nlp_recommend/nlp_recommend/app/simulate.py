
import os
from transformers import pipeline

import sys

CUR_DIR = os.path.dirname(__file__) # app folder
PARENT_DIR = os.path.dirname(os.path.dirname(CUR_DIR)) # nlp_recommend
MODEL_DIR = os.path.join(os.path.dirname(PARENT_DIR), 'training')

sys.path.insert(0, PARENT_DIR)

from nlp_recommend.utils.utils import rerank
from nlp_recommend.models import ContainerSpacy

class Models:
    # sent_model = pipeline('sentiment-analysis', model='distilbert-base-uncased') #loading hugging face pipeline
    model_philo = ContainerSpacy('philosophy')
    model_psycho = ContainerSpacy('psychology')
    model_adv = ContainerSpacy('adventure')
    
def filter_preds(idx, warped) -> list[dict]:
    res = []
    for i in idx:
        pred = {'title':None, 'quote':None, 'author':None}
        pred['title'] = warped['sentence'][i]['title']
        pred['before'] = ' .'.join(warped['before'][i]['sentence'])
        pred['after'] = ' .'.join(warped['after'][i]['sentence'])
        pred['quote'] = warped['sentence'][i]['sentence']
        pred['author'] = warped['sentence'][i]['author']
        res.append(pred)
    return res

def get_predictions(text, container, topk=5):
    # quote pred
    best_valid_index, warped_dict = container.warper.predict(
        container.model, text, return_index=True, topk=topk)
    best_index = container.cls.match_filter(text, best_valid_index)
    print('best_index', best_valid_index, 'to', best_index)
    result = container.warper.corpus.loc[best_index.org_idx, ['sentence', 'author', 'title']]
    print('result', result)
    # 
    filtered_idx = [i for i, elt in enumerate(best_valid_index) if elt in best_index.index]
    preds = filter_preds(filtered_idx, warped_dict)
    # title pred
    title_preds = {}
    if len(result)>0:
        title_preds['title'] = rerank(result['title'].values)[0]
        print('debug title preds', title_preds)
        title_preds['author'] = result.loc[result.title == title_preds['title'],
                                         'author'].values[0]
    return preds, title_preds

def predict(text, models):
    """
    To make a prediction on one sample of the text
    satire or fake news
    :return: a result of prediction in HTML page
    """
    # Receive
    # Predict books
    models = {'philo': models.model_philo, 'psycho': models.model_psycho, 'adv': models.model_adv}
    preds = {}
    for name, model in models.items():
        preds_raw, title_preds = get_predictions(text, model)
        preds[name] = {key: elt.capitalize() for key, elt in preds_raw[0].items() if elt}
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
    


if __name__ == '__main__':
    models = Models()
    text = 'I am at home'
    predict(text, models)