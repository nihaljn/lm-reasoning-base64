import argparse
import os
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from arithmetic_evaluator import (ArithmeticEvaluator,
                                  ArithmeticEvaluator_Base64,
                                  ArithmeticEvaluator_Base64_CoT_Base64,
                                  ArithmeticEvaluator_Base64_CoT_English)
from translation_evaluator import TranlsationEvaluator_Base64ToEnglish

SYS_PROMPT_STORE = {
    "assistant": "You are a helpful assistant.",
    "base64_assistant": "You are a helpful assistant who can understand and respond only in base64.",
    "base64_cot_english_assistant": "You are a helpful assistant who can understand and respond only in base64. "
                                    "You think step-by-step in English within <thoughts> tags before following instructions.",
    "base64_cot_base64_assistant": "You are a helpful assistant who can understand and respond only in base64. "
                                   "You think step-by-step in base64 within <thoughts> tags before following instructions.",
}

EVALUATOR_STORE = {
    "english_evaluator": ArithmeticEvaluator,
    "base64_evaluator": ArithmeticEvaluator_Base64,
    "base64_cot_english_evaluator": ArithmeticEvaluator_Base64_CoT_English,
    "base64_cot_base64_evaluator": ArithmeticEvaluator_Base64_CoT_Base64,
    "translation_evaluator": TranlsationEvaluator_Base64ToEnglish,
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
                        choices=["assistant", "base64_assistant",
                                 "base64_cot_english_assistant",
                                 "base64_cot_base64_assistant"])
    parser.add_argument("--evaluator", type=str, default="english_evaluator",
                        choices=["english_evaluator", "base64_evaluator",
                                 "base64_cot_english_evaluator",
                                 "base64_cot_base64_evaluator",
                                 "translation_evaluator"])
    parser.add_argument("--model_name", type=str, default="gpt-4o",
                        choices=["gpt-4o"])
    parser.add_argument("--few_shot_k", type=int, default=0)
    args = parser.parse_args()

    # read the data
    assert os.path.exists(args.data_path), f"{args.data_path} does not exist"
    data = pd.read_csv(args.data_path).astype(str)
    if not args.num_examples or args.num_examples > len(data):
        args.num_examples = len(data)
    data = data[:args.num_examples]
    if args.few_shot_data_path and args.few_shot_k > 0:
        assert os.path.exists(args.few_shot_data_path), (
            f"{args.few_shot_data_path} does not exist"
        )
        few_shot_data = pd.read_csv(args.few_shot_data_path).to_dict("records")
        args.few_shot_k = min(args.few_shot_k, len(few_shot_data))
        few_shot_data = few_shot_data[:args.few_shot_k]
    else:
        few_shot_data = None

    if args.evaluator == "translation_evaluator":
        # target should be the prompt itself
        print("Setting target to prompt for translation evaluator")
        data["target"] = data["prompt"]
        if few_shot_data is not None:
            few_shot_data["target"] = few_shot_data["prompt"]

    if args.num_threads <= 0:
        raise ValueError("num_threads should be a positive integer")
    
    # set up evaluators
    args.num_threads = min(args.num_threads, len(data))
    rpm_limit = args.rpm_limit / args.num_threads
    EvaluatorClass = EVALUATOR_STORE[args.evaluator]
    filename = f"{args.model_name}_n{args.num_examples}_t{args.temperature}_k{args.few_shot_k}"
    filename += f"_{args.system_prompt}_{args.evaluator}.jsonl"
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
        task = (i % args.num_threads, data.iloc[i], few_shot_data)
        tasks.append(task)
    # run the tasks
    with Pool(args.num_threads) as p:
        list(tqdm(p.imap_unordered(runner, tasks), total=len(tasks), ncols=80))
    
    # cleanup
    with open(output_path, "w") as f:
        for i in range(args.num_threads):
            with open(output_path + f"_{i}", "r") as g:
                for line in g:
                    f.write(line)
            os.remove(output_path + f"_{i}")


if __name__ == "__main__":
    main()
