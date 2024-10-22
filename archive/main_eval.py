import json
import os

from arithmetic_eval import (ArithmeticEval, ArithmeticEval_Base64,
                             ArithmeticEval_FewShot,
                             ArithmeticEval_FewShot_Base64)
from mmlu_eval import (MMLUEval, MMLUEval_Base64, MMLUEval_FewShot,
                       MMLUEval_FewShot_Base64)
from sampler.chat_completion_sampler import (OPENAI_SYSTEM_MESSAGE_API,
                                             OPENAI_SYSTEM_MESSAGE_CHATGPT,
                                             ChatCompletionSampler)

# from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS


def main():
    debug = False
    num_threads = 1
    sampler_name = "gpt-4o_assistant"
    output_root_dir = "data/model_outputs"
    temperature = 0.0
    num_examples = None
    max_tokens = 2048
    few_shot_k = 5
    n_digits = None
    
    import sys
    assert len(sys.argv) >= 2, "Please provide the eval name"
    eval_name = sys.argv[1]
    if "arithmetic" in eval_name:
        assert len(sys.argv) == 3, "Please provide the number of digits for arithmetic eval"
        try:
            n_digits = int(sys.argv[1])
        except:
            raise ValueError("The number of digits should be an integer but you provided", sys.argv[1])

    inference_args = {
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    samplers = {
        # chatgpt models:
        "gpt-4-turbo-2024-04-09_assistant": ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            **inference_args
        ),
        "gpt-4-turbo-2024-04-09_chatgpt": ChatCompletionSampler(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            **inference_args
        ),
        "gpt-4o_assistant": ChatCompletionSampler(
            model="gpt-4o-2024-08-06",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            **inference_args
        ),
        "gpt-4o_chatgpt": ChatCompletionSampler(
            model="gpt-4o-2024-08-06",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            **inference_args
        ),
        "gpt-4o-mini-2024-07-18": ChatCompletionSampler(
            model="gpt-4o-mini-2024-07-18",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            **inference_args
        ),
        # claude models:
        # "claude-3-opus-20240229_empty": ClaudeCompletionSampler(
        #     model="claude-3-opus-20240229", system_message=None,
        # ),
    }

    def get_evals(eval_name, **kwargs):
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "mmlu":
                return MMLUEval(num_examples=1 if debug else num_examples, **kwargs)
            case "mmlu_base64":
                return MMLUEval_Base64(num_examples=1 if debug else num_examples, **kwargs)
            case "mmlu_few_shot":
                return MMLUEval_FewShot(num_examples=1 if debug else num_examples, **kwargs)
            case "mmlu_few_shot_base64":
                return MMLUEval_FewShot_Base64(num_examples=1 if debug else num_examples, **kwargs)
            case "arithmetic":
                return ArithmeticEval(num_examples=1 if debug else num_examples, **kwargs)
            case "arithmetic_base64":
                return ArithmeticEval_Base64(num_examples=1 if debug else num_examples, **kwargs)
            case "arithmetic_few_shot":
                return ArithmeticEval_FewShot(num_examples=1 if debug else num_examples, **kwargs)
            case "arithmetic_few_shot_base64":
                return ArithmeticEval_FewShot_Base64(num_examples=1 if debug else num_examples, **kwargs)
            case _:
                raise Exception(f"Unrecoginized eval type: {eval_name}")

    eval_config = {
        "mmlu": {"category": "stem", "num_threads": num_threads},
        "mmlu_base64": {"category": "stem", "num_threads": num_threads},
        "mmlu_few_shot": {"category": "stem", "num_threads": num_threads, "k": few_shot_k},
        "mmlu_few_shot_base64": {"category": "stem", "num_threads": num_threads, "k": few_shot_k},
        "arithmetic": {"n_digits": n_digits, "op": "addition", "num_threads": num_threads},
        "arithmetic_base64": {"n_digits": n_digits, "op": "addition", "num_threads": num_threads},
        "arithmetic_few_shot": {"n_digits": n_digits, "op": "addition", "num_threads": num_threads, "k": few_shot_k},
        "arithmetic_few_shot_base64": {"n_digits": n_digits, "op": "addition", "num_threads": num_threads, "k": few_shot_k},
    }
    
    debug_suffix = "_DEBUG" if debug else ""
    sampler = samplers[sampler_name]
    eval_obj = get_evals(eval_name, **eval_config.get(eval_name, {}))
    suffix = ""
    for k, v in eval_config.get(eval_name, {}).items():
        suffix += f"_{k}={v}"
    for k, v in inference_args.items():
        suffix += f"_{k}={v}"
    
    output_fp = os.path.join(
        output_root_dir, f"{sampler_name}_{eval_name}{suffix}{debug_suffix}.jsonl"
    )
    result = eval_obj(sampler, output_fp)


if __name__ == "__main__":
    main()
