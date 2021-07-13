from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import pickle
import os
import numpy as np
from tqdm import tqdm

from nlp_recommend.models.base import BaseModel
from nlp_recommend.settings import TOPK, BERT_MODEL
from nlp_recommend.const import PARENT_DIR

MAT_PATH = os.path.join(PARENT_DIR, 'weights/bert_model.pkl')


class BertModel(BaseModel):
    def __init__(self, data=None, model_name=BERT_MODEL):
        super().__init__(name='Transformers')
        self.model_name = model_name
        self.load_model()
        self.load()
        if not hasattr(self, 'embed_mat'):
            assert data is not None, 'No cache data found, add data argument'
            self.fit_transform(data)
        assert (self.model)

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.pipeline = pipeline('feature-extraction',
                                 model=self.model, tokenizer=self.tokenizer)

    def load(self, mat_path=MAT_PATH):
        if os.path.exists(mat_path):
            self.embed_mat = pickle.load(open(mat_path, "rb"))

    def fit_transform(self, data):
        batchs = np.array_split(data, len(data)//1000)
        mat_vec = []
        for batch in tqdm(batchs, total=len(batchs)):
            batch = batch.tolist()
            mat_vec.extend(self.transform(batch))
        self.embed_mat = mat_vec

    def transform(self, data):
        """1.69 s ± 95 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) for 50 sentences
        """
        token_dict = self.tokenizer(
            data,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt")
        # each of the 512 token has a 768-d vector
        embeddings = np.array(self.pipeline(data))  # shape 1, 5, 768
        # retrieve attention mask
        # print('embeddings', embeddings.shape)
        mask = token_dict['attention_mask'].unsqueeze(
            -1).expand(embeddings.shape).float()
        mask = mask.detach().numpy()
        # apply mask
        mask_embeddings = embeddings * mask
        # sum all the tokens on the axis 1 to keep one 768-d "sentence" vector
        summed = np.sum(mask_embeddings, axis=1)
        # Then sum the number of values that must be given attention in each position of the tensor:
        summed_mask = np.clip(mask.sum(1), a_min=1e-9, a_max=None)
        mean_pooled = summed / summed_mask
        return mean_pooled

    def save_embeddings(self, mat_path=MAT_PATH):
        """ """
        with open(mat_path, 'wb') as fw:
            pickle.dump(self.embed_mat, fw)

    def predict(self, in_sentence):
        if 'str' in in_sentence.__class__.__name__:
            sentence = [in_sentence]
        input_vec = self.transform(sentence)
        mat = cosine_similarity(input_vec, self.embed_mat)
        # best cos sim for each token independantly
        best_index = self.extract_best_indices(mat, topk=3)
        return best_index


if __name__ == '__main__':
    test_sentence = 'My life'
    tfidf_model = BertModel()
    tfidf_model.predict(test_sentence)
