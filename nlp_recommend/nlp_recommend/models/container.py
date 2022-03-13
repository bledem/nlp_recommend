
from nlp_recommend.models import BertModel, SpacyModel, CombinedModel, SentimentCls, Warper, TfIdfModel
from nlp_recommend.const import WEIGHT_DIR

class ContainerBase():
    def __init__(self, dataset, weight_dir=WEIGHT_DIR, sent_model= None):
        """ContainerBase
        Args:
            dataset (str): name of the dataset genre (psychology, adventure, etc)
            weight_dir (_type_, optional): Path to directory containing ``dataset`` and ``weights`` folders. Defaults to WEIGHT_DIR.
            sent_model (pipeline hf, optional): sentiment analysis model. Defaults to None.
        """
        self.weight_dir = weight_dir
        self.cls = SentimentCls(dataset=dataset, weight_dir=weight_dir, model=sent_model)
        self.warper = Warper(dataset=dataset, dataset_path=weight_dir)
        
class Container(ContainerBase):
    def __init__(self, dataset, weight_dir, sent_model):
        super().__init__(dataset, weight_dir, sent_model)
        self.model = CombinedModel(dataset=dataset, weight_dir=weight_dir)
        
class ContainerLight(ContainerBase):
    def __init__(self, dataset, weight_dir=WEIGHT_DIR, sent_model=None):
        super().__init__(dataset, weight_dir, sent_model)
        self.model = BertModel(dataset=dataset, weight_dir=weight_dir)

class ContainerVeryLight(ContainerBase):
    def __init__(self, dataset, weight_dir=WEIGHT_DIR, sent_model=None):
        super().__init__(dataset, weight_dir, sent_model)
        self.model = TfIdfModel(dataset=dataset, weight_dir=weight_dir)

class ContainerSpacy(ContainerBase):
    def __init__(self, dataset, weight_dir=WEIGHT_DIR, sent_model=None):
        super().__init__(dataset, weight_dir, sent_model)
        self.model = SpacyModel(dataset=dataset, weight_dir=weight_dir)
       