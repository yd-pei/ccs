import transformers
from src.ccs.data import data_loader
import torch


def ccs_train(parsed_args):
    """
    :param parsed_args:
    :return: the logistic regression model
    """
    cola = data_loader.load_cola(parsed_args.split)

    model = transformers.AutoModel.from_pretrained(parsed_args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        parsed_args.tokenizer
    )

    train_dataset = cola.get_batch_data(parsed_args.iteration)

    hidden_layer = {True: [], False: []}

    for i in range(0, parsed_args.iteration, parsed_args.batch_size):
        """
        Produce the hidden layer collected from LMs.
        """
        end = (
            parsed_args.iteration
            if i + parsed_args.batch_size > parsed_args.iteration
            else i + parsed_args.batch_size
        )

        t_prompt = train_dataset[True][i:end]
        f_prompt = train_dataset[False][i:end]

        hidden_layer[True].append(
            get_batch_hidden_layer(model, tokenizer, t_prompt)
        )
        hidden_layer[False].append(
            get_batch_hidden_layer(model, tokenizer, f_prompt)
        )


def get_batch_hidden_layer(model, tokenizer, prompt):
    inputs = tokenizer(prompt,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")

    outputs = model(input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_hiddenstates=True)

    all_layers_states = outputs.encoder_hidden_states

    last_layer_states = all_layers_states[-1]
    last_token_hidden_state = last_layer_states[:, -1, :]

    phi = last_token_hidden_state
