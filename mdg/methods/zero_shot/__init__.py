from mdg.methods import ModelWrapper
from typing import List, Optional


class ZeroShotModelWrapper(ModelWrapper):
    """
    Method wrapper for zero-shot models
    """

    def run(self, test_data: List, **kwargs) -> List:
        """Run the model on the test data. Additional kwargs can be passed in for model-specific parameters."""
        pass


from mdg.methods.zero_shot.openrouter_based import Llama31OpenRouterWrapperTimex
from mdg.methods.zero_shot.mock_method import Llama31OpenRouterWrapperTimexMock
from mdg.methods.zero_shot.openrouter_based import Llama31OpenRouterWrapperTLink