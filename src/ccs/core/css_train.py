import torch

def css_train(hidden_layer:dict):
    t_normalized_tensor = normalize(hidden_layer[True])
    f_normalized_tensor = normalize(hidden_layer[False])

    return

def normalize(hidden_tensor:torch.Tensor):
    # for each x_i in hidden_tensor, x_i = x_i - avg(x)
    # hidden_tensor: (1000, 768)
    avg_tensor = torch.mean(hidden_tensor, dim=0)
    normalized_tensor = hidden_tensor - avg_tensor
    return normalized_tensor

def loss_function():
    pass

def logistic_regression():
    pass