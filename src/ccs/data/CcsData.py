from datasets import Dataset


class CcsData:
    def __init__(self,dataset:Dataset,ds_name:str,config:str):
        self.dataset = dataset
        self.ds_name = ds_name
        self.config = config
        

    def build_cola(dataset:Dataset):
        pass