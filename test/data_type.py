from datasets import load_dataset

glue = load_dataset(
    "nyu-mll/glue",
    "cola",
    split="train"
)

print(type(glue))
