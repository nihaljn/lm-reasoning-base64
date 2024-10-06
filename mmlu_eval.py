"""
Measuring Massive Multitask Language Understanding
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2009.03300
"""
import json
import os
import random
import re
import time

import pandas

import common
from common import (ANSWER_PATTERN_MULTICHOICE, format_multichoice_question,
                    format_multichoice_question_no_cot,
                    format_multichoice_answer_no_cot)
from custom_types import Eval, SamplerBase
from translate import base64_to_english, english_to_base64

subject2category = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}
SAMPLER = None


def item_handler(row: dict) -> dict:
    global SAMPLER
    # create the prompt
    # prompt = format_multichoice_question(row)
    prompt = format_multichoice_question_no_cot(row)
    prompt_messages = [
        SAMPLER._pack_message(content=prompt, role="user")
    ]
    # get the response
    ct = time.time()
    response_text = SAMPLER(prompt_messages)
    elapsed = time.time() - ct
    # look for the answer
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    extracted_answer = match.group(1) if match else None
    # score
    score = 1.0 if extracted_answer == row["Answer"] else 0.0
    # return the session info
    ret_dict = {
        "index": row["UniqueIndex"],
        "prompt": prompt,
        "correct_answer": row["Answer"],
        "extracted_answer": extracted_answer,
        "score": score,
        "category": subject2category.get(row["Subject"], "other"),
        "_subject": row["Subject"],
        "_response_text": response_text,
        "_elapsed_time": elapsed,
    }
    with open(row["output_file"], "a") as f:
        f.write(json.dumps(ret_dict)+"\n")
    return ret_dict


def item_handler_few_shot(row: dict) -> dict:
    global SAMPLER
    # create the prompt
    # prompt = format_multichoice_question(row)
    prompt = format_multichoice_question_no_cot(row)
    prompt_messages = []
    for example in FEW_SHOT_EXAMPLES:
        prompt_messages += [
            SAMPLER._pack_message(content=format_multichoice_question_no_cot(example), role="user"),
            SAMPLER._pack_message(content=format_multichoice_answer_no_cot(example), role="assistant")
        ]
    prompt_messages.append(
        SAMPLER._pack_message(content=prompt, role="user")
    )
    # get the response
    ct = time.time()
    response_text = SAMPLER(prompt_messages)
    elapsed = time.time() - ct
    # look for the answer
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    extracted_answer = match.group(1) if match else None
    # score
    score = 1.0 if extracted_answer == row["Answer"] else 0.0
    # return the session info
    ret_dict = {
        "index": row["UniqueIndex"],
        "prompt": prompt,
        "correct_answer": row["Answer"],
        "extracted_answer": extracted_answer,
        "score": score,
        "category": subject2category.get(row["Subject"], "other"),
        "_messages": prompt_messages,
        "_subject": row["Subject"],
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
    # prompt = format_multichoice_question(row)
    prompt = format_multichoice_question_no_cot(row)
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
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    extracted_answer = match.group(1) if match else None
    # score
    score = 1.0 if extracted_answer == row["Answer"] else 0.0
    # return the session info
    ret_dict = {
        "index": row["UniqueIndex"],
        "prompt": prompt,
        "correct_answer": row["Answer"],
        "extracted_answer": extracted_answer,
        "score": score,
        "category": subject2category.get(row["Subject"], "other"),
        "_subject": row["Subject"],
        "_response_text": response_text,
        "_elapsed_time": elapsed,
    }
    ret_dict.update(base64_log_dict)
    with open(row["output_file"], "a") as f:
        f.write(json.dumps(ret_dict)+"\n")
    return ret_dict


def item_handler_few_shot_base64(row: dict) -> dict:
    global SAMPLER, FEW_SHOT_EXAMPLES

    base64_log_dict = {}

    # create the prompt
    # prompt = format_multichoice_question(row)
    prompt = format_multichoice_question_no_cot(row)
    base64_log_dict["_original_prompt"] = prompt

    # translate to base64
    prompt = english_to_base64(prompt)

    # get the response
    prompt_messages = []
    for example in FEW_SHOT_EXAMPLES:
        p, t = format_multichoice_question_no_cot(example), format_multichoice_answer_no_cot(example)
        p, t = english_to_base64(p), english_to_base64(t)
        prompt_messages += [
            SAMPLER._pack_message(content=p, role="user"),
            SAMPLER._pack_message(content=t, role="assistant")
        ]
    prompt_messages.append(
        SAMPLER._pack_message(content=prompt, role="user")
    )
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
    match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
    extracted_answer = match.group(1) if match else None
    # score
    score = 1.0 if extracted_answer == row["Answer"] else 0.0
    # return the session info
    ret_dict = {
        "index": row["UniqueIndex"],
        "prompt": prompt,
        "correct_answer": row["Answer"],
        "extracted_answer": extracted_answer,
        "score": score,
        "category": subject2category.get(row["Subject"], "other"),
        "_messages": prompt_messages,
        "_subject": row["Subject"],
        "_response_text": response_text,
        "_elapsed_time": elapsed,
    }
    ret_dict.update(base64_log_dict)
    with open(row["output_file"], "a") as f:
        f.write(json.dumps(ret_dict)+"\n")
    return ret_dict


class MMLUEval(Eval):
    def __init__(
            self, 
            num_examples: int | None = None, 
            category: str | None = None,
            num_threads: int = 1
    ):
        df = pandas.read_csv("data/mmlu/mmlu_stem_250.csv")
        few_shot_df = pandas.read_csv("data/mmlu/mmlu_stem_250_few_shot.csv")
        examples = [row.to_dict() for _, row in df.iterrows()]
        few_shot_examples = [row.to_dict() for _, row in few_shot_df.iterrows()]
        # filter examples if needed
        if category:
            examples = [
                e for e in examples 
                if subject2category.get(e["Subject"], "other") == category
            ]
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
            self.examples = [e for e in self.examples if e["UniqueIndex"] not in done_indices]
            # run the examples
            results = common.map_with_progress(
                self.fn, self.examples, num_threads=self.num_threads
            )
        except Exception as e:
            results = []
            print(f"Error: {e}")
            raise
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
    

class MMLUEval_FewShot(MMLUEval):
    def __init__(
            self, 
            num_examples: int | None = None, 
            category: str | None = None,
            num_threads: int = 1,
            k: int = 1
    ):
        super().__init__(num_examples, category, num_threads)
        global FEW_SHOT_EXAMPLES
        FEW_SHOT_EXAMPLES = self.few_shot_examples[:k]
        self.fn = item_handler_few_shot


class MMLUEval_Base64(MMLUEval):
    """MMLU Eval but in Base64"""
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


class MMLUEval_FewShot_Base64(MMLUEval_FewShot):
    """MMLU Eval Few Shot but in Base64"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        global FEW_SHOT_EXAMPLES
        FEW_SHOT_EXAMPLES = self.few_shot_examples[:kwargs.get("k", 1)]
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
