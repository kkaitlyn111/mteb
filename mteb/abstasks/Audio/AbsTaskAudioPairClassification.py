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


class AudioPairClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for AudioPairClassification

    Attributes:
        num_samples: Number of audio samples
        total_duration: Total audio duration in seconds

        min_duration1: Minimum audio clip duration
        avg_duration1: Average audio clip duration
        max_duration1: Maximum audio clip duration
        sample_rate1: Audio sample rate

        min_duration2: Minimum audio clip duration
        avg_duration2: Average audio clip duration
        max_duration2: Maximum audio clip duration
        sample_rate2: Audio sample rate

        unique_labels: Number of unique labels
        labels: dict of label frequencies
    """

    num_samples: int
    total_duration: float

    min_duration1: float
    avg_duration1: float
    max_duration1: float
    sample_rate1: int

    min_duration2: float
    avg_duration2: float
    max_duration2: float
    sample_rate2: int

    unique_labels: int
    labels: dict[str, dict[str, int]]


class AbsTaskAudioPairClassification(AbsTask):
    """Abstract class for AudioPairClassificationTasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for pairwise pair classification.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        audio1: datasets.Audio
        audio2: datasets.Audio
        label: int
    """

    abstask_prompt = "Retrieve audio that are semantically similar to the given audio."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _evaluate_subset(
        self,
        model: Encoder,
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
        if hf_subset:
            dataset = self.dataset[hf_subset][split]
            if isinstance(dataset, list):
                dataset = dataset[0]
        elif compute_overall:
            dataset = defaultdict(list)
            for hf_subset in self.metadata.eval_langs:
                cur_dataset = self.dataset[hf_subset][split]
                if isinstance(cur_dataset, list):
                    cur_dataset = cur_dataset[0]
                for key, value in cur_dataset.items():
                    dataset[key].extend(value[0] if len(value) == 1 else value)
        else:
            dataset = self.dataset[split]
    
    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for classification"""
        raise NotImplementedError("This method should be overridden by subclasses")
