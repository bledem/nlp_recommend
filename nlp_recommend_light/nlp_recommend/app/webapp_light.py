from flask import Flask, request, render_template
import argparse
import sys
import os
import re
import logging

CUR_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(os.path.dirname(CUR_DIR))
sys.path.insert(0, PARENT_DIR)

from nlp_recommend.utils.utils import rerank
from nlp_recommend.models import ContainerSpacy
from nlp_recommend.const import WEIGHT_DIR
logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=WEIGHT_DIR)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default='0.0.0.0')
    args = parser.parse_args()
    return args

def create_app(config=None):  
    app = Flask(__name__)
    @app.route('/')
    def home():
        default_preds = {'title':None, 'quote':None, 'author':None}
        default_ans = 'Something like'
        return render_template('index.html', philo_preds=default_preds, 
                                psycho_preds=default_preds, adv_preds=default_preds, ans=default_ans)


    def filter_preds(idx, warped):
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
        """_summary_

        Args:
            text (_type_): _description_
            container (_type_): _description_
            topk (int, optional): _description_. Defaults to 5.

        Returns:
            preds: List[dict['title':str, 'before':str, 'after':str, 'quote':str, 'author':str]]
            title_preds: List[dict['title':str, 'author':str]]
        """
        # quote pred
        best_valid_index, warped_dict = container.warper.predict(
            container.model, text, return_index=True, topk=topk)
        best_index = container.cls.match_filter(text, best_valid_index)
        print('best_index', best_valid_index, 'to', best_index)
        result = container.warper.corpus.loc[best_index.org_idx, ['sentence', 'author', 'title']]
        print('result', result)
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

    def clean_sentence(text):
        # capitalize
        match = re.search("[a-zA-Z]", text)
        capitalized_list = list(text)
        capitalized_list[match.start()] = capitalized_list[match.start()].upper() # capitalize
        capitalized_text = ''.join(capitalized_list)
       
        # remove spurious spaces
        regex_pattern = r"([!.?])"
        split_text = re.split(regex_pattern, capitalized_text)
        split_list = [e.rstrip() for e in split_text]
        split_list = [' '+e if e[0] != ' ' and len(e)>2 else e for e in split_list]
        capitalized_text = ''.join(split_list)

        # add punctuation
        if capitalized_text[-1] != '.':
            capitalized_text += '.'
        print(capitalized_text)
        return capitalized_text

    @app.route('/', methods=['POST'])
    def predict():
        """
        To make a prediction on one sample of the text
        satire or fake news
        :return: a result of prediction in HTML page
        """
        # Receive
        text = request.form['thoughts']
        # Predict books
        models = {'philo': model_philo, 'psycho': model_psycho, 'adv': model_adv}
        preds = {}
        for name, model in models.items():
            preds_raw, title_preds = get_predictions(text, model)
            preds[name] = {key: clean_sentence(elt) for key, elt in preds_raw[0].items() if elt}
            logger.info('title_preds', title_preds)
            preds[name]['recommended_title'] = title_preds

            
        ans = 'Something Like'

        return render_template('index.html', ans=ans, input=text, 
                                philo_preds=preds['philo'], 
                                psycho_preds=preds['psycho'],
                                adv_preds=preds['adv'])

    return app

if __name__ == '__main__':
    args = arg_parser()
    global weight_dir
    global model_philo
    global model_psycho
    global model_adv
    weight_dir = args.data_path

    model_philo = ContainerSpacy('philosophy', weight_dir=weight_dir)
    model_psycho = ContainerSpacy('psychology', weight_dir=weight_dir)
    model_adv = ContainerSpacy('adventure', weight_dir=weight_dir)
    host = args.host
    port = args.port
    print(f'---> Go into your browser at http://{host}:{port} <---')
    app = create_app()
    app.run(host=host, port=port, debug=True)