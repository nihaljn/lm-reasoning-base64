import json
import re

from evaluator import EvaluatorBase


class ArithmeticEvaluator(EvaluatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # check if there is an out file path
        self.out_path = kwargs.get("out_path", None)
        if self.out_path:
            self.out_file = open(self.out_path, "w")
        
        # templates
        self.user_template = "Solve the following arithmetic problem. The only " \
        "line of your response should be of the following format: 'Answer: " \
        "$NUMBER' (without quotes) where NUMBER is the answer to the problem.\n\n" \
        "{prompt}"
        self.answer_regex = r"(?i)Answer\s*:\s*([^\n]+)"
        self.answer_template = "Answer: {answer}"

    
    def preprocess_user_message(self, message: str) -> str:
        """Preprocess user message"""
        return self.user_template.format(prompt=message)
    

    def preprocess_asst_message(self, message: str) -> str:
        """Preprocess assistant message"""
        return self.answer_template.format(answer=message)


    def postprocess(self, message: str) -> dict:
        """Postprocess model output"""
        match = re.search(self.answer_regex, message)
        extracted_answer = match.group(1) if match else None
        return {"extracted_answer": extracted_answer}
    

    def _before_exit(self, ret_dict: dict) -> dict:
        """Perform any final operations before returning"""
        # deal with any complex types
        print(ret_dict)
        if self.out_path:
            with open(self.out_path, "a") as f:
                f.write(json.dumps(ret_dict)+"\n")
