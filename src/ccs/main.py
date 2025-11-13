import argparse
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

    return parser

def main(args=None):
    parser = get_parser()
    parsed_args = parser.parse_args(args)

    hf_model = parsed_args.model
    print(f"Model: {hf_model} being used.")

    """ try:
        hf_login_token = os.environ.get(TOKEN_VAR_NAME)
        hf.login(token=hf_login_token)
    except Exception as e:
        print() """

if __name__ == "__main__":
    main()