from datasets import Dataset

GLUE = "nyu-mll/glue"

class CcsData:
    def __init__(self,dataset:Dataset,ds_name:str,config:str):
        self.dataset = dataset
        self.ds_name = ds_name
        self.config = config

    def get_batch_data(self,data_size:int)->dict|None:
        if self.ds_name == GLUE:
            return self.get_batch_glue(data_size)
        else:
            return None

    def get_batch_glue(self,data_size:int)->dict:
        batch_data = {True:[],False:[]}
        for i in range(data_size):
            prompt = f"Sentence:{self.dataset["sentence"][i]} "
            claim = "Claim: The sentence is a grammatical English sentence. "
            t_prompt = "I think the claim is True"
            f_prompt = "I think the claim is False"
            true_prompt = (f"{prompt}"
                           f"{claim}"
                           f"{t_prompt}")
            false_prompt = (f"{prompt}"
                            f"{claim}"
                            f"{f_prompt}")
            batch_data[True].append(true_prompt)
            batch_data[False].append(false_prompt)

        return batch_data