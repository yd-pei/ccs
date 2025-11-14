import argparse
from src.ccs.core.ccs_core import ccs_train
# import huggingface_hub as hf
# import os

# TOKEN_VAR_NAME = 'HF_TOKEN'

def get_parser():
    parser = argparse.ArgumentParser(
        description="This is the replication of ccs algorithm.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-m','--model',
        type=str,
        default='Qwen/Qwen3-4B',
        help='Model repo name on hf.'
    )

    parser.add_argument(
        '-d','--dataset',
        type=str,
        default='nyu-mll/glue',
        choices=['nyu-mll/glue'],
        help='Dataset repo name on hf.'
    )

    return parser

def main(args=None):
    parser = get_parser()
    parsed_args = parser.parse_args(args)

    print(f"Model: {parsed_args.model} being used.")
    print(f"Dataset: {parsed_args.dataset} being used.")

    ccs_train(parsed_args)

    """ try:
        hf_login_token = os.environ.get(TOKEN_VAR_NAME)
        hf.login(token=hf_login_token)
    except Exception as e:
        print() """

if __name__ == "__main__":
    main()