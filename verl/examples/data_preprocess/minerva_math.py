"""Preprocess the Minerva Math evaluation dataset to parquet format"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/minvera")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--instruction_following_template", default=None)
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    data_source = "math-ai/minervamath"

    dataset = datasets.load_dataset(
        data_source,
    )["test"]
    if args.instruction_following_template is not None:
        print("Using custom instruction following template:")
        print(args.instruction_following_template)
        instruction_following = args.instruction_following_template
    else:
        instruction_following = "Please think step-by-step and put your final answer within \\boxed{}."
    if args.prefix != "":
        print("Using custom prefix:")
        print(args.prefix)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = args.prefix + question_raw + " " + instruction_following

            solution = str(example.pop("answer"))
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    test_dataset = dataset.map(function=make_map_fn("test"), with_indices=True)
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    print(test_dataset[0])
    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)