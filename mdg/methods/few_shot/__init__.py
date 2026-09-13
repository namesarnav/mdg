from mdg.methods import ModelWrapper
from typing import List, Optional


class FewShotModelWrapper(ModelWrapper):
    """
    Method wrapper for few-shot models
    """

    def run(self, test_data: List, train_data: Optional[List] = None, dev_data: Optional[List] = None, **kwargs) -> List:
        """Run the model on the test data. If train/dev data is provided, use it for training/fine-tuning. Eg. for few-shot learning,
        use train/dev data for in-context examples.
        Additional kwargs can be passed in for model-specific parameters.
        """
        pass
   
import mdg.methods.few_shot.openrouter_based 


