from typing import Optional, List


class ModelWrapper:
    def __init__(self, config: dict):
        """A wrapper class for a model with a configuration. configs should tell you how to create the model object.
        For example, if we use a zero shot method through openrouter, pass in all necessary parameters in config.
        """
        self.config = config

    def __repr__(self):
        return f"ModelWrapper(config={self.config})"

    def run(self, test_data: List, **kwargs) -> List:
        """Run the model on the test data. If train/dev data is provided, use it for training/fine-tuning. Eg. for few-shot learning,
        use train/dev data for in-context examples.
        Additional kwargs can be passed in for model-specific parameters.
        """
        pass


from mdg.methods.few_shot import *
from mdg.methods.zero_shot import *
