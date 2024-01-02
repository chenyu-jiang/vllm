import argparse
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--input_file", type=str, default="./test_dump_token_ids.tsv")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    with open(args.input_file, "r") as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                continue
            line = line.strip()
            if line == "":
                continue
            token_id, input_tokens, output_token = line.split("\t")
            token_id = int(token_id)
            input_tokens = [int(x) for x in input_tokens.split(",")]
            output_token = int(output_token)
            input_tokens_str = tokenizer.decode(input_tokens)
            output_token_str = tokenizer.decode(output_token)
            print("=" * 80)
            print(f"token_id: {token_id}")
            print(f"input tokens: {input_tokens_str}")
            print(f"output token: {output_token_str}")
            input()

if __name__ == "__main__":
    main()
