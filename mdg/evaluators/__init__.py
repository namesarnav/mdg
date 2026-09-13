from typing import List, Optional
from mdg.datamodels import (
    TimeExpression,
    TrajectorExpression,
    LandmarkExpression,
    SpIndicatorExpression,
    Result,
    CompositeResult,
    TimeDocument,
    SpatialDocument,
)


class Evaluator:
    def __init__(self, config, **kwargs):
        self.config = config

    def run(self, gold_data: List, predicted_data: List, metric: Optional[str], **kwargs) -> Result | CompositeResult:
        """
        Run the evaluation.
        :param gold_data: List of gold standard documents
        :param predicted_data: List of predicted documents
        :param metric: Specific metric to compute (if any)
        :param kwargs: Additional arguments
        :return: Result or list of Results (if you are returning )
        """
        pass


class ExtractionEvaluator(Evaluator):
    """
    Evaluator for extraction datasets
    """

    def run(
        self,
        gold_data: List[TimeExpression] | List[TrajectorExpression] | List[LandmarkExpression] | List[SpIndicatorExpression],
        predicted_data: (
            List[TimeExpression] | List[TrajectorExpression] | List[LandmarkExpression] | List[SpIndicatorExpression]
        ),
    ):
        pass


class ExtractionEvaluatorDoc(Evaluator):
    """
    Evaluator for extraction datasets, at the document level
    """

    def run(
        self, gold_data: List[TimeDocument] | List[SpatialDocument], predicted_data: List[TimeDocument] | List[SpatialDocument]
    ):
        pass


from mdg.evaluators.temporal_evals import *
from mdg.evaluators.spatial_evals import *
from mdg.evaluators.causal_evals import *
