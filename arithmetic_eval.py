import json
import os
import random
import re
import time

import pandas

import common
from common import ANSWER_PATTERN
from custom_types import Eval, SamplerBase
from translate import english_to_base64, base64_to_english

SAMPLER = None
ARITHMETIC_TEMPLATE = """Solve the following arithmetic problem. The only line of your response should be of the following format: 'Answer: $NUMBER' (without quotes) where NUMBER is the answer to the problem.

{prompt}"""


def format_arithmetic(row: dict) -> str:
    return ARITHMETIC_TEMPLATE.format(prompt=row["prompt"])


def format_arithmetic_response(row: dict) -> str:
    return f"Answer: {row['target']}"


def item_handler(row: dict) -> dict:
    global SAMPLER
    # create the prompt
    prompt = format_arithmetic(row)
    prompt_messages = [
        SAMPLER._pack_message(content=prompt, role="user")
    ]
    # get the response
    ct = time.time()
    response_text = SAMPLER(prompt_messages)
    elapsed = time.time() - ct
    # look for the answer
    match = re.search(ANSWER_PATTERN, response_text)
    extracted_answer = match.group(1) if match else None
    # score
    score = 1.0 if extracted_answer == str(row["target"]) else 0.0
    # return the session info
    ret_dict = {
        "index": row["index"],
        "prompt": prompt,
        "correct_answer": row["target"],
        "extracted_answer": extracted_answer,
        "score": score,
        "_messages": prompt_messages,
        "_response_text": response_text,
        "_elapsed_time": elapsed,
    }
    with open(row["output_file"], "a") as f:
        f.write(json.dumps(ret_dict)+"\n")
    return ret_dict


def item_handler_base64(row: dict) -> dict:
    global SAMPLER

    base64_log_dict = {}

    # create the prompt
    prompt = format_arithmetic(row)
    base64_log_dict["_original_prompt"] = prompt
    
    # translate to base64
    prompt = english_to_base64(prompt)
    
    # get the response
    prompt_messages = [
        SAMPLER._pack_message(content=prompt, role="user")
    ]
    ct = time.time()
    response_text = SAMPLER(prompt_messages)
    elapsed = time.time() - ct

    # translate back to English
    try:
        base64_log_dict["_base64_response"] = response_text
        response_text = base64_to_english(response_text)
    except Exception as e:
        base64_log_dict["_base64_error"] = str(e)
        response_text = ""
    
    # look for the answer
    match = re.search(ANSWER_PATTERN, response_text)
    extracted_answer = match.group(1) if match else None
    # score
    score = 1.0 if extracted_answer == str(row["target"]) else 0.0
    # return the session info
    ret_dict = {
        "index": row["index"],
        "prompt": prompt,
        "correct_answer": row["target"],
        "extracted_answer": extracted_answer,
        "score": score,
        "_messages": prompt_messages,
        "_response_text": response_text,
        "_elapsed_time": elapsed,
    }
    ret_dict.update(base64_log_dict)
    with open(row["output_file"], "a") as f:
        f.write(json.dumps(ret_dict)+"\n")
    return ret_dict


def item_handler_few_shot(row: dict) -> dict:
    global SAMPLER, FEW_SHOT_EXAMPLES
    # create the prompt
    prompt = format_arithmetic(row)
    prompt_messages = []
    for example in FEW_SHOT_EXAMPLES:
        prompt_messages += [
            SAMPLER._pack_message(content=format_arithmetic(example), role="user"),
            SAMPLER._pack_message(content=format_arithmetic_response(example), role="assistant")
        ]
    prompt_messages += [
        SAMPLER._pack_message(content=prompt, role="user")
    ]
    # get the response
    ct = time.time()
    response_text = SAMPLER(prompt_messages)
    elapsed = time.time() - ct
    # look for the answer
    match = re.search(ANSWER_PATTERN, response_text)
    extracted_answer = match.group(1) if match else None
    # score
    score = 1.0 if extracted_answer == str(row["target"]) else 0.0
    # return the session info
    ret_dict = {
        "index": row["index"],
        "prompt": prompt,
        "correct_answer": row["target"],
        "extracted_answer": extracted_answer,
        "score": score,
        "_messages": prompt_messages,
        "_response_text": response_text,
        "_elapsed_time": elapsed,
    }
    with open(row["output_file"], "a") as f:
        f.write(json.dumps(ret_dict)+"\n")
    return ret_dict


def item_handler_few_shot_base64(row: dict) -> dict:
    global SAMPLER, FEW_SHOT_EXAMPLES

    base64_log_dict = {}

    # create the prompt
    prompt = format_arithmetic(row)
    base64_log_dict["_original_prompt"] = prompt
    
    # translate to base64
    prompt = english_to_base64(prompt)
    
    # get the response
    prompt_messages = []
    for example in FEW_SHOT_EXAMPLES:
        p, t = format_arithmetic(example), format_arithmetic_response(example)
        p, t = english_to_base64(p), english_to_base64(t)
        prompt_messages += [
            SAMPLER._pack_message(content=p, role="user"),
            SAMPLER._pack_message(content=t, role="assistant")
        ]
    prompt_messages += [
        SAMPLER._pack_message(content=prompt, role="user")
    ]
    ct = time.time()
    response_text = SAMPLER(prompt_messages)
    elapsed = time.time() - ct

    # translate back to English
    try:
        base64_log_dict["_base64_response"] = response_text
        response_text = base64_to_english(response_text)
    except Exception as e:
        base64_log_dict["_base64_error"] = str(e)
        response_text = ""
    
    # look for the answer
    match = re.search(ANSWER_PATTERN, response_text)
    extracted_answer = match.group(1) if match else None
    # score
    score = 1.0 if extracted_answer == str(row["target"]) else 0.0
    # return the session info
    ret_dict = {
        "index": row["index"],
        "prompt": prompt,
        "correct_answer": row["target"],
        "extracted_answer": extracted_answer,
        "score": score,
        "_messages": prompt_messages,
        "_response_text": response_text,
        "_elapsed_time": elapsed,
    }
    ret_dict.update(base64_log_dict)
    with open(row["output_file"], "a") as f:
        f.write(json.dumps(ret_dict)+"\n")
    return ret_dict


class ArithmeticEval(Eval):
    def __init__(
            self, 
            num_examples: int | None = None, 
            n_digits: int | None = None,
            num_threads: int = 1,
            op: str = "addition"
    ):
        df = pandas.read_csv(f"data/arithmetic/{op}_{n_digits}digits.csv")
        few_shot_df = pandas.read_csv(f"data/arithmetic/{op}_{n_digits}digits_few_shot.csv")
        examples = [row.to_dict() for _, row in df.iterrows()]
        few_shot_examples = [row.to_dict() for _, row in few_shot_df.iterrows()]
        # filter examples if needed
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.few_shot_examples = few_shot_examples
        self.num_threads = num_threads
        self.fn = item_handler

    def __call__(self, sampler: SamplerBase, output_file: str) -> list[dict]:
        global SAMPLER
        SAMPLER = sampler

        for i, example in enumerate(self.examples):
            # add a target file path to each example depending on num_threads
            # if num_threads is 1 simply use output file
            if self.num_threads == 1:
                example["output_file"] = output_file
            else:
                example["output_file"] = output_file + f"_{i % self.num_threads}"
                if not os.path.exists(example["output_file"]):
                    with open(example["output_file"], "w") as f:
                        f.write("")

        if not os.path.exists(output_file):
            with open(output_file, "w") as f:
                f.write("")
        else:
            print(f"Continuing from {output_file}")
        try:
            # filter the examples already in the output file
            with open(output_file, "r") as f:
                done_indices = set()
                for line in f:
                    done_indices.add(json.loads(line)["index"])
            self.examples = [e for e in self.examples if e["index"] not in done_indices]
            # run the examples
            results = common.map_with_progress(
                self.fn, self.examples, num_threads=self.num_threads
            )
        except Exception as e:
            results = []
            print(f"Error: {e}")
        except KeyboardInterrupt:
            results = []
            print("Interrupted")

        # merge all the output files
        if self.num_threads > 1:
            with open(output_file, "a") as f:
                for i in range(self.num_threads):
                    with open(output_file + f"_{i}", "r") as f2:
                        f.write(f2.read())
                    os.remove(output_file + f"_{i}")
        
        print(f"Results written to {output_file}")

        return results
    

class ArithmeticEval_Base64(ArithmeticEval):
    """Arithmetic Eval but in Base64"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fn = item_handler_base64
    
    def __call__(self, sampler: SamplerBase, output_file: str) -> list[dict]:
        """Override the call method to handle base64 translation"""
        global SAMPLER 
        SAMPLER = sampler
        _sys_message = SAMPLER.system_message
        SAMPLER.update_parameters(
            system_message="You are a helpful assistant who can understand and respond only in base64."
        )
        ret = super().__call__(SAMPLER, output_file)
        SAMPLER.update_parameters(system_message=_sys_message)
        return ret


class ArithmeticEval_FewShot(ArithmeticEval):
    def __init__(
            self, 
            num_examples: int | None = None, 
            n_digits: int | None = None,
            num_threads: int = 1,
            op: str = "addition",
            k: int = 1
    ):
        super().__init__(num_examples, n_digits, num_threads, op)
        global FEW_SHOT_EXAMPLES
        FEW_SHOT_EXAMPLES = self.few_shot_examples[:k]
        self.fn = item_handler_few_shot


class ArithmeticEval_FewShot_Base64(ArithmeticEval_FewShot):
    """Arithmetic Eval Few Shot but in Base64"""
    def __init__(
            self, 
            num_examples: int | None = None, 
            n_digits: int | None = None,
            num_threads: int = 1,
            op: str = "addition",
            k: int = 1
    ):
        super().__init__(num_examples, n_digits, num_threads, op)
        global FEW_SHOT_EXAMPLES
        FEW_SHOT_EXAMPLES = self.few_shot_examples[:k]
        self.fn = item_handler_few_shot_base64
    
    def __call__(self, sampler: SamplerBase, output_file: str) -> list[dict]:
        """Override the call method to handle base64 translation"""
        global SAMPLER 
        SAMPLER = sampler
        _sys_message = SAMPLER.system_message
        SAMPLER.update_parameters(
            system_message="You are a helpful assistant who can understand and respond only in base64."
        )
        ret = super().__call__(SAMPLER, output_file)
        SAMPLER.update_parameters(system_message=_sys_message)
        return ret
