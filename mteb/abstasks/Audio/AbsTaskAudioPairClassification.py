from __future__ import annotations

import logging
from collections import Counter, defaultdict

from typing import Any
from datasets import Dataset

from ..encoder_interface import Encoder
from ..evaluation.evaluators import AudioPairClassificationEvaluator
from ..load_results.task_results import ScoresDict
from .AbsTask import AbsTask
from .TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class AbsTaskPairClassification(AbsTask):
    """Abstract class for AudioPairClassificationTasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for pairwise pair classification.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        audio1: datasets.Audio
        audio2: datasets.Audio
        label: int
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _evaluate_subset(
        self,
        model: AudioEncoder,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, str] = {},
        **kwargs,
    ) -> ScoresDict:
        data_split = dataset[0]
        logging.getLogger(
            "sentence_transformers.evaluation.PairClassificationEvaluator"
        ).setLevel(logging.WARN)
        evaluator = AudioPairClassificationEvaluator(
            data_split["audio1"],
            data_split["audio2"],
            data_split["labels"],
            task_name=self.metadata.name,
            **kwargs,
        )
        scores = evaluator.compute_metrics(model, encode_kwargs=encode_kwargs)

        self._add_main_score(scores)
        return scores

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> None:
        pass
    
    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for classification"""
        raise NotImplementedError("This method should be overridden by subclasses")
