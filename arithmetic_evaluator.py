import json
import re

from evaluator import EvaluatorBase
from translate import english_to_base64, base64_to_english


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
        extracted_answer = match.group(1) if match else ""
        return {"extracted_answer": extracted_answer}
    

    def _before_exit(self, ret_dict: dict) -> dict:
        """Perform any final operations before returning"""
        # deal with any complex types
        if self.out_path:
            with open(self.out_path, "a") as f:
                f.write(json.dumps(ret_dict)+"\n")


class ArithmeticEvaluator_Base64(ArithmeticEvaluator):
    """Arithmetic Evaluator with base64 encoding"""
    def preprocess_user_message(self, message: str) -> str:
        """Preprocess user message to base64"""
        english_msg = self.user_template.format(prompt=message)
        try:
            base64 = english_to_base64(english_msg)
        except:
            raise ValueError(f"Failed to convert message to base64: {english_msg}")
        return base64
    

    def preprocess_asst_message(self, message: str) -> str:
        """Preprocess assistant message to base64"""
        english_msg = self.answer_template.format(answer=message)
        try:
            base64 = english_to_base64(english_msg)
        except:
            raise ValueError(f"Failed to convert message to base64: {english_msg}")
        return base64
    
    
    def postprocess(self, message: str) -> dict:
        """Postprocess model output expecting a base64 response"""
        try:
            english_response = base64_to_english(message) # expect base64 response
        except Exception as e:
            # keep the response in-tact for record
            english_response = message + f"\n\nFailed: {e}"
        match = re.search(self.answer_regex, english_response)
        extracted_answer = match.group(1) if match else ""
        return {"extracted_answer": extracted_answer}
