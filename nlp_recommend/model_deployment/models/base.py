import numpy as np

TOPK = 3


class BaseModel():
    def __init__(self, name):
        self.name = name

    def fit_transform(self):
        pass

    def save_embeddings(self):
        pass

    def predict(self, topk):
        pass

    @staticmethod
    def extract_best_indices(score_mat, topk=TOPK, mask=None):
        """
        Use sum over all tokens
        score_mat (array): cos matrix of shape (nb_in_tokens, nb_dict_tokens)
        topk (int): number of indices to return (from high to lowest in order)
        """
        # return the sum on all tokens of cosinus for each sentence
        if len(score_mat.shape) > 1:
            cos_sim = np.mean(score_mat, axis=0)
        else:
            cos_sim = score_mat
        index = np.argsort(cos_sim)[::-1]  # from highest idx to smallest score
        if mask is not None:
            assert mask.shape == score_mat.shape
            mask = mask[index]
        else:
            mask = np.ones(len(cos_sim))
        # eliminate 0 cosine distance
        mask = np.logical_or(cos_sim[index] != 0, mask)
        best_index = index[mask][:topk]
        return best_index
