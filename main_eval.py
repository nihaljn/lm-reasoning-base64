import json
import os

from mmlu_eval import MMLUEval, MMLUEval_Base64
from sampler.chat_completion_sampler import (OPENAI_SYSTEM_MESSAGE_API,
                                              OPENAI_SYSTEM_MESSAGE_CHATGPT,
                                              ChatCompletionSampler)

# from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS


def main():
    debug = False
    num_threads = 3
    sampler_name = "gpt-4o_assistant"
    output_root_dir = "data/model_outputs"
    temperature = 0.5
    num_examples = None
    max_tokens = 2048
    eval_name = "mmlu_base64"

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
            case _:
                raise Exception(f"Unrecoginized eval type: {eval_name}")

    eval_config = {
        "mmlu": {"category": "stem", "num_threads": num_threads},
        "mmlu_base64": {"category": "stem", "num_threads": num_threads}
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
