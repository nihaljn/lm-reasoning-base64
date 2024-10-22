import argparse
import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from arithmetic_evaluator import ArithmeticEvaluator

SYS_PROMPT_STORE = {
    "assistant": "You are a helpful assistant."
}

EVALUATOR_STORE = {
    "english_evaluator": ArithmeticEvaluator,
    "english_cot_evaluator": None,
    "base64_evaluator": None,
    "base64_cot_evaluator": None,
}


def runner(args: tuple) -> dict:
    global evaluators
    evaluator_index, sample, few_shot_examples = args
    prompt, target = sample["prompt"], sample["target"]
    metadata = {"index": sample["index"]}
    return evaluators[evaluator_index].evaluate(
        prompt, target, metadata, few_shot_examples, 
        preprocess_few_shot_examples=True
    )


def main():
    global evaluators
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_root_dir", type=str, required=True)
    parser.add_argument("--few_shot_data_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--rpm_limit", type=float, default=20)
    parser.add_argument("--num_examples", type=int, default=None)
    parser.add_argument("--system_prompt", type=str, default="assistant",
                        choices=["assistant"])
    parser.add_argument("--evaluator", type=str, default="english_evaluator",)
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        choices=["gpt-4o"])
    args = parser.parse_args()

    # read the data
    assert os.path.exists(args.data_path), f"{args.data_path} does not exist"
    data = pd.read_csv(args.data_path).astype(str)[:args.num_examples]
    if args.few_shot_data_path:
        assert os.path.exists(args.few_shot_data_path), (
            f"{args.few_shot_data_path} does not exist"
        )
        few_shot_data = pd.read_csv(args.few_shot_data_path).to_dict("records")
    else:
        few_shot_data = None

    if args.num_threads <= 0:
        raise ValueError("num_threads should be a positive integer")
    
    # set up evaluators
    rpm_limit = args.rpm_limit / args.num_threads
    EvaluatorClass = EVALUATOR_STORE[args.evaluator]
    filename = f"{args.model_name}_{args.num_examples}_{args.temperature}.jsonl"
    output_path = os.path.join(args.out_root_dir, filename)
    if not os.path.exists(args.out_root_dir):
        print(f"Creating directory: {args.out_root_dir}")
        os.makedirs(args.out_root_dir)
    evaluators = [
        EvaluatorClass(
            model_name=args.model_name,
            system_message=SYS_PROMPT_STORE[args.system_prompt],
            rpm_limit=rpm_limit,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            out_path=output_path + f"_{i}"
        ) for i in range(args.num_threads)
    ]
    
    # distribute tasks
    tasks = []
    for i in range(len(data)):
        # evaluator = evaluators[i % args.num_threads]
        task = (i % args.num_threads, data.iloc[i], few_shot_data)
        tasks.append(task)
    # run the tasks
    with Pool(args.num_threads) as p:
        list(tqdm(p.imap(runner, tasks), total=len(tasks)))
    
    # cleanup
    with open(output_path, "w") as f:
        for i in range(args.num_threads):
            with open(output_path + f"_{i}", "r") as g:
                for line in g:
                    f.write(line)
            os.remove(output_path + f"_{i}")


if __name__ == "__main__":
    main()
