from datasets import load_dataset

SUPPORTED_DS = {
    "nyu-mll/glue":"cola" 
}

def get_ds_config(dataset:str):
    if dataset in SUPPORTED_DS.keys():
        return SUPPORTED_DS[dataset]
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    
