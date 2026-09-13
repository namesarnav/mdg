from mdg.methods.zero_shot.openrouter_based import OpenRouterBasedWrapper
from mdg.datamodels import TimeDocument
from typing import List
from mdg.registry import register, MODEL_WRAPPER


@register(_type=MODEL_WRAPPER, _name="llama-3.1-openrouter-zero-shot-timex-mock")
class Llama31OpenRouterWrapperTimexMock(OpenRouterBasedWrapper):
    """
    Method wrapper for Llama 3.1 models via OpenRouter. Takes in a list of TimeDocuments and returns a list of TimeDocuments.
    """

    def __init__(self, config: dict):
        super().__init__(name="Llama 3.1", config=config)

    def run(self, test_data: List[TimeDocument], **kwargs) -> List[TimeDocument]:
        """Run the model on the test data. Additional kwargs can be passed in for model-specific parameters."""
        return test_data
