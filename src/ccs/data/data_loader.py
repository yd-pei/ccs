from datasets import load_dataset
from src.ccs.data import CcsData


SUPPORTED_DS = {
    "nyu-mll/glue":"cola" 
}

def get_ds_config(dataset:str):
    if dataset in SUPPORTED_DS.keys():
        return SUPPORTED_DS[dataset]
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    
def load_cola()->CcsData:
    cola = load_dataset(
        "nyu-mll/glue",
        get_ds_config("nyu-mll/glue"),
        split="train"
    )
    
    ccs_cola = CcsData(cola,
        "nyu-mll/glue",
        get_ds_config("nyu-mll/glue")    
    )
    return ccs_cola