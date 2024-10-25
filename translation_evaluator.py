import json
import re

from evaluator import EvaluatorBase
from translate import english_to_base64


class TranlsationEvaluator_Base64ToEnglish(EvaluatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # check if there is an out file path
        self.out_path = kwargs.get("out_path", None)
        if self.out_path:
            self.out_file = open(self.out_path, "w")
        
        # templates
        self.instruction_template = "Decode the following text from base64. " \
        "You should put your response within <translation> tags " \
        "and say nothing else.\n\n{prompt}"
        self.user_template = "Solve the following arithmetic problem. The only " \
        "line of your response should be of the following format: 'Answer: " \
        "$NUMBER' (without quotes) where NUMBER is the answer to the problem.\n\n" \
        "{prompt}"
        self.answer_template = "<translation>{answer}</translation>"

    
    def preprocess_user_message(self, message: str) -> str:
        """Preprocess user message"""
        message = self.user_template.format(prompt=message)
        base64 = english_to_base64(message)
        return self.instruction_template.format(prompt=base64)
    

    def preprocess_asst_message(self, message: str) -> str:
        """Preprocess assistant message"""
        raise NotImplementedError


    def postprocess(self, message: str) -> dict:
        """Postprocess model output"""
        # extract answer
        start_index = message.find("<translation>")
        end_index = message.find("</translation>")
        if start_index == -1 or end_index == -1:
            extracted_answer = ""
        else:
            extracted_answer = message[start_index+len("<translation>"):end_index]
            extracted_answer = extracted_answer.strip()
        return {"extracted_answer": extracted_answer}


    def score(self, hypothesis: dict, target: str) -> float:
        """Score the response"""
        hypothesis = hypothesis.get("extracted_answer").strip()
        target = self.user_template.format(prompt=target).strip()
        if hypothesis is None:
            raise ValueError("No 'extracted_answer' found in", hypothesis)
        return 1.0 if hypothesis == target else 0.0
        

    def _before_exit(self, ret_dict: dict) -> dict:
        """Perform any final operations before returning"""
        # deal with any complex types
        ret_dict["target"] = self.user_template.format(prompt=ret_dict["target"])
        if self.out_path:
            with open(self.out_path, "a") as f:
                f.write(json.dumps(ret_dict)+"\n")
