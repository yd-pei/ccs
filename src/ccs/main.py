import argparse
from src.ccs.core.ccs_core import ccs_main
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

    parser.add_argument(
        '-s','--split',
        type=str,
        default='train',
        help='Split repo name on hf.'
    )

    parser.add_argument(
        '-t','--task',
        type=str,
        default='train',
        choices=['train', 'evaluate'],
        help='What task to run.'
    )

    parser.add_argument(
        '-i','--iteration',
        type=int,
        default=1000,
        help='How many data samples to be used.'
    )

    parser.add_argument(
        '-b','--batch',
        type=int,
        default=32,
        help='How many batches to use for GPU training.'
    )

    return parser

def main(args=None):
    parser = get_parser()
    parsed_args = parser.parse_args(args)

    print(f"Model: {parsed_args.model} being used.")
    print(f"Dataset: {parsed_args.dataset} being used.")

    ccs_main(parsed_args)

    """ try:
        hf_login_token = os.environ.get(TOKEN_VAR_NAME)
        hf.login(token=hf_login_token)
    except Exception as e:
        print() """

if __name__ == "__main__":
    main()