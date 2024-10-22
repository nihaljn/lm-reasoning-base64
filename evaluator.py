from time import time

from sampler.chat_completion_sampler import ChatCompletionSampler


class EvaluatorBase:
    def __init__(
        self,
        model_name: str,
        system_message: str = None,
        rpm_limit: float = None,
        few_shot_examples: list[dict] = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        **kwargs
    ):
        """Base class for evaluators"""
        self.completion_sampler = ChatCompletionSampler(
            model=model_name,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.rpm_limit = rpm_limit
        self.few_shot_examples = few_shot_examples
     
     
    def preprocess_user_message(self, message: str) -> str:
        """Preprocess user message"""
        return message
    

    def preprocess_asst_message(self, message: str) -> str:
        """Preprocess assistant message"""
        return message
    

    def preprocess(self, messages: list[str]):
        """Preprocess messages"""
        preprocessed_messages = []
        for message in messages:
            # preprocess user message
            if message["role"] == "user":
                msg = self.preprocess_user_message(message["content"])
            # preprocess assistant message
            elif message["role"] == "assistant":
                msg = self.preprocess_asst_message(message["content"])
            else:
                raise ValueError(f"Invalid role: {message['role']}")
            preprocessed_messages.append(msg)
        return preprocessed_messages
    

    def infer(self, messages: list[str]) -> dict:
        """Get model output"""
        # get model output
        ct = time()
        response = self.completion_sampler(messages)
        elapsed = time() - ct
        # sleep to maintain the rate limit
        if self.rpm_limit:
            sleep_time = 60 / self.rpm_limit
            if elapsed < sleep_time:
                sleep_time -= elapsed
                time.sleep(sleep_time)
        return {"response_text": response, "elapsed_time": elapsed}
    

    def postprocess(self, message: str) -> dict:
        """Postprocess model output"""
        return {"extracted_answer": message}
    

    def score(self, hypothesis: dict, target: str) -> float:
        """Score the response"""
        hypothesis = hypothesis.get("extracted_answer")
        if hypothesis is None:
            raise ValueError("No 'extracted_answer' found in", hypothesis)
        return 1.0 if hypothesis == target else 0.0
    

    def _before_exit(self, ret_dict: dict) -> dict:
        """Perform any final operations before returning"""
        return ret_dict


    def evaluate(
        self, 
        prompt: str, 
        target: str, 
        metadata: dict, 
        few_shot_examples: list[dict] | None = None
    ):
        """Evaluate the model on the given input.
        If few_shot_examples is provided, it will be used,
        else the class-level few_shot_examples will be used."""
        # input messages
        this_message = {"role": "user", "content": prompt}
        if not few_shot_examples and self.few_shot_examples:
            few_shot_examples = self.few_shot_examples
        if few_shot_examples:
            messages = few_shot_examples + [this_message]
        else:
            messages = [this_message]
        ret_dict = {"original_messages": messages}
        
        # preprocess messages
        preprocessed_messages = self.preprocess(messages)
        ret_dict["preprocessed_messages"] = preprocessed_messages
        
        # get model output
        response = self.infer(preprocessed_messages)
        ret_dict.update(response)
        
        # postprocess output
        hypothesis = self.postprocess(response)
        ret_dict.update(hypothesis)

        # score the response
        score = self.score(hypothesis, target)
        ret_dict["score"] = score

        ret_dict.update({"target": target, "metadata": metadata})
        return self._before_exit(ret_dict)
