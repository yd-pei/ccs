from datasets import load_dataset
from src.ccs.data import CcsData


SUPPORTED_DS = {
    "nyu-mll/glue":"cola" 
}

GLUE = "nyu-mll/glue"

def get_ds_config(dataset:str):
    if dataset in SUPPORTED_DS:
        return SUPPORTED_DS[dataset]
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    
def load_cola(split)->CcsData:
    cola = load_dataset(
        GLUE,
        get_ds_config(GLUE),
        split
    )
    
    ccs_cola = CcsData(cola,
        GLUE,
        get_ds_config(GLUE)
    )
    return ccs_cola