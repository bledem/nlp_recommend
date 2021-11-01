"""
Requires models or weights folder.
"""

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline, BertModel as BertM, BertTokenizer
import torch
import pickle
import os
import numpy as np
from tqdm import tqdm
import hydra

from nlp_recommend.models.base import BaseModel
from nlp_recommend.const import PARENT_DIR, WEIGHT_DIR
from nlp_recommend.settings import BERT_BATCH_SIZE, TOPK


class BertModel(BaseModel):
    def __init__(self,
         topk=3,
         bert_model='sentence-transformers/paraphrase-mpnet-base-v2',
         device='cpu',
         small_memory=True,
         batch_size=BERT_BATCH_SIZE,
         dataset='philosophy',
         training= 'pretrained',
         version='v1.0',
         weight_dir=WEIGHT_DIR):
        super().__init__(name='Transformers')
        if training == 'pretrained':
            self.model_name = bert_model.split('/')[1] if '/' in bert_model else bert_model
        else:
            self.model_name = bert_model.split('/')[-1] if '/' in bert_model else bert_model
        self.bert_model = bert_model
        self._set_device(device)
        self.small_device = 'cpu' if small_memory else self.device
        self.dataset = dataset
        self.batch_size = batch_size
        self.topk = topk
        self.training = training
        self.mat_path = os.path.join(
            weight_dir, 'weights',
            dataset, f'{self.model_name}_{dataset}_{training}_{version}.pkl')
        self.load_model()
        self.load()
        if not hasattr(self, 'embed_mat'):
            print(f'Check weight path at', self.mat_path)
            print('No cache data found, add data argument')
            print('you can run fit_transform on your data')
            # self.fit_transform(data)
        assert (self.model)

    def _set_device(self, device):
        if device == -1 or device == 'cpu':
            self.device = 'cpu'
        elif device == 'cuda' or device == 'gpu':
            self.device = 'cuda'
        elif isinstance(device, int) or isinstance(device, float):
            self.device = 'cuda'
        else:  # default
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        if 'finetune' in self.training:
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
            self.model = BertM.from_pretrained(self.bert_model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
            self.model = AutoModel.from_pretrained(self.bert_model)
        device = -1 if self.device == 'cpu' else 0
        self.pipeline = pipeline('feature-extraction',
                                 model=self.model, tokenizer=self.tokenizer, device=device)

    def load(self):
        if os.path.exists(self.mat_path):
            self.embed_mat = pickle.load(open(self.mat_path, "rb"))

    def fit_transform(self, data):
        nb_batchs = 1 if (len(data) < self.batch_size) else len(
            data) // self.batch_size
        batchs = np.array_split(data, nb_batchs)
        mean_pooled = []
        for batch in tqdm(batchs, total=len(batchs), desc='Training...'):
            mean_pooled.append(self.transform(batch))
        mean_pooled_tensor = torch.tensor(
            len(data), dtype=float).to(self.small_device)
        mean_pooled = torch.cat(mean_pooled, out=mean_pooled_tensor)
        self.embed_mat = mean_pooled

    @staticmethod
    def mean_pooling(token_embeddings, attention_mask=None):
        if attention_mask is not None:
            input_mask_expanded = attention_mask.unsqueeze(
                -1).expand(token_embeddings.size()).float()
        else:
            input_mask_expanded = torch.ones(token_embeddings.shape)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def transform(self, data):
        """1.69 s ± 95 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) for 50 sentences
        """
        if 'str' in data.__class__.__name__:
            data = [data]
        data = list(data)
        token_dict = self.tokenizer(
            data,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt")
        # each of the 512 (max lenght) tokens has a 768 or 384-d vector depends on model)
        # if this a CLS task, only the first token is required
        # shape is len(data), len(sentence), 768
        token_embed = torch.tensor(self.pipeline(data)).to(self.device)
        # if len(token_embed) == 1:
        #     token_embed = token_embed[0][0] # taking only the first token CLS 
        # else:
        #     raise 'give one sentence at a time'
        # attention mask (simply differentiates padding from non-padding).
        attention_mask = token_dict['attention_mask'].to(self.device)
        # average pooling of masked embeddings
        mean_pooled = self.mean_pooling(
            token_embed, attention_mask)
        # mean_pooled = self.mean_pooling(token_embed)
        # mean_pooled = token_embed.unsqueeze(0)
        # print(mean_pooled.shape) #TODO take only CLS token
        mean_pooled = mean_pooled.to(self.small_device)
        return mean_pooled

    def save_embeddings(self, out_path=None):
        """ """
        if not out_path:
            out_path = self.mat_path
        with open(self.mat_path, 'wb') as fw:
            pickle.dump(self.embed_mat, fw)

    def predict(self, in_sentence, topk=TOPK):
        input_vec = self.transform(in_sentence)
        mat = cosine_similarity(input_vec, self.embed_mat)
        # best cos sim for each token independantly
        # TODO swap the lines and update the Warper
        # best_index = self.extract_best_indices(mat, topk=self.topk)
        best_index = self.extract_best_indices(mat, topk=topk)
        return best_index

    def predict_title(self, margin):
        """ """


@hydra.main(config_path='../conf/model', config_name='bert.yaml')
def test_model(cfg):
    test_sentence = 'My life'
    bert_model = hydra.utils.instantiate(cfg.bert) 
    prediction = bert_model.predict(test_sentence)
    return prediction

if __name__ == '__main__':
    # prediction = test_model()
    # print(prediction)
    # test_sentence = 'My life'
    # bert_model = BertModel()
    # bert_model.predict(test_sentence)
    bert_model = BertModel(dataset='philosophy', topk=3,
                       bert_model='sentence-transformers/paraphrase-mpnet-base-v2',
                       small_memory=True,
                       device='cpu', batch_size=8)