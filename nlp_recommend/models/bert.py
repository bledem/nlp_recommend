from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import pickle
import os
import numpy as np
from tqdm import tqdm

from nlp_recommend.models.base import BaseModel
from nlp_recommend.settings import TOPK, BERT_MODEL, BERT_BATCH_SIZE
from nlp_recommend.const import PARENT_DIR

MODEL_NAME = BERT_MODEL.split('/')[1] if '/' in BERT_MODEL else BERT_MODEL


class BertModel(BaseModel):
    def __init__(self, data=None, dataset='philosophy', model_name=BERT_MODEL, device=-1, small_memory=True, batch_size=BERT_BATCH_SIZE):
        super().__init__(name='Transformers')
        self.model_name = model_name
        self._set_device(device)
        self.small_device = 'cpu' if small_memory else self.device
        self.dataset = dataset
        self.batch_size = batch_size
        self.mat_path = os.path.join(
            PARENT_DIR, 'weights', dataset, f'{MODEL_NAME}_{dataset}.pkl')
        self.load_model()
        self.load()
        if not hasattr(self, 'embed_mat'):
            assert data is not None, 'No cache data found, add data argument'
            self.fit_transform(data)
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
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
    def mean_pooling(token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(
            -1).expand(token_embeddings.size()).float()
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
        token_embed = torch.tensor(self.pipeline(data)).to(self.device)
        # each of the 512 token has a 768 or 384-d vector depends on model)
        attention_mask = token_dict['attention_mask'].to(self.device)
        # average pooling of masked embeddings
        mean_pooled = self.mean_pooling(
            token_embed, attention_mask)
        mean_pooled = mean_pooled.to(self.small_device)
        return mean_pooled

    def save_embeddings(self):
        """ """
        with open(self.mat_path, 'wb') as fw:
            pickle.dump(self.embed_mat, fw)

    def predict(self, in_sentence, topk=TOPK):
        input_vec = self.transform(in_sentence)
        mat = cosine_similarity(input_vec, self.embed_mat)
        # best cos sim for each token independantly
        best_index = self.extract_best_indices(mat, topk=topk)
        return best_index


if __name__ == '__main__':
    test_sentence = 'My life'
    tfidf_model = BertModel()
    tfidf_model.predict(test_sentence)
