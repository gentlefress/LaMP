import evaluate
import numpy as np

from typing import Callable, Dict
from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core import MetricForLanguageGeneration
from nlgmetricverse.metrics._core.utils import requirement_message
try:
    from itertools import izip as zip
except ImportError:
    pass
_LICENSE= """ """
_DESCRIPTION = """\
MAUVE is a library built on PyTorch and HuggingFace Transformers to measure the gap between neural text and human text with the 
eponymous MAUVE measure.
MAUVE summarizes both Type I and Type II errors measured softly using Kullback–Leibler (KL) divergences.
For details, see the MAUVE paper: https://arxiv.org/abs/2102.01454 (Neurips, 2021).
This metrics is a wrapper around the official implementation of MAUVE:
https://github.com/krishnap25/mauve
"""

_CITATION = """\
@inproceedings{pillutla-etal:mauve:neurips2021,
title={MAUVE: Measuring the Gap Between Neural Text and Human Text using Divergence Frontiers},
author={Pillutla, Krishna and Swayamdipta, Swabha and Zellers, Rowan and Thickstun, John and Welleck, Sean and Choi, Yejin and
Harchaoui, Zaid},
booktitle = {NeurIPS},
year = {2021}
}
"""

_KWARGS_DESCRIPTION = """
Calculates MAUVE scores between two lists of generated text and reference text.
Args:
    predictions: list of generated text to score. Each predictions should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each reference should be a string with tokens separated by spaces.
Optional Args:
    num_buckets: the size of the histogram to quantize P and Q. Options: 'auto' (default) or an integer
    pca_max_data: the number data points to use for PCA dimensionality reduction prior to clustering. If -1, use all the data. Default -1
    kmeans_explained_var: amount of variance of the data to keep in dimensionality reduction by PCA. Default 0.9
    kmeans_num_redo: number of times to redo k-means clustering (the best objective is kept). Default 5
    kmeans_max_iter: maximum number of k-means iterations. Default 500
    featurize_model_name: name of the model from which features are obtained. Default 'gpt2-large' Use one of ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'].
    device_id: Device for featurization. Supply a GPU id (e.g. 0 or 3) to use GPU. If no GPU with this id is found, use CPU
    max_text_length: maximum number of tokens to consider. Default 1024
    divergence_curve_discretization_size: Number of points to consider on the divergence curve. Default 25
    mauve_scaling_factor: "c" from the paper. Default 5.
    verbose: If True (default), print running time updates
    seed: random seed to initialize k-means cluster assignments.
Returns:
    mauve: MAUVE score, a number between 0 and 1. Larger values indicate that P and Q are closer
    frontier_integral: Frontier Integral, a number between 0 and 1. Smaller values indicate that P and Q are closer
    reduced_scores: list of MAUVE scores for each prediction-reference pair
Examples:
    >>> from nlgmetricverse import NLGMetricverse, load_metric
    >>> predictions = ["There is a cat on the mat.", "Look! a wonderful day."]
    >>> references = ["The cat is playing on the mat.", "Today is a wonderful day"]
    >>> scorer = NLGMetricverse(metrics=load_metric("mauve"))
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    "mauve": {
        "score": 0.0040720962619612555,
        "reduced_scores": [
            0.0040720962619612555,
            0.0040720962619612555
        ]
    }
"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MauvePlanet(MetricForLanguageGeneration):
    def __init__(
            self,
            resulting_name: str = None,
            compute_kwargs: Dict = None,
            p_features=None,
            q_features=None,
            p_tokens=None,
            q_tokens=None,
            num_buckets: str='auto',
            pca_max_data: int=-1,
            kmeans_explained_var: float=0.9,
            kmeans_num_redo: int=5,
            kmeans_max_iter: int=500,
            featurize_model_name: str="gpt2-large",
            device_id: int=-1,
            max_text_length: int=1024,
            divergence_curve_discretization_size: int=25,
            mauve_scaling_factor: int=5,
            _seed = 25,
            verbose: bool=True,
            **kwargs,
    ):

        self.p_features=p_features
        self.q_features=q_features
        self.p_tokens=p_tokens
        self.q_tokens=q_tokens
        self.num_buckets =num_buckets
        self.pca_max_data=pca_max_data
        self.kmeans_explained_var=kmeans_explained_var
        self.kmeans_num_redo=kmeans_num_redo
        self.kmeans_max_iter=kmeans_max_iter
        self.featurize_model_name=featurize_model_name
        self.device_id=device_id
        self.max_text_length=max_text_length
        self.divergence_curve_discretization_size=divergence_curve_discretization_size
        self.mauve_scaling_factor=mauve_scaling_factor
        self.verbose=verbose
        self._seed=_seed
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _download_and_prepare(self, dl_manager) -> None:
        """
        Import the computation of mauve score from mauve-text library.
        """
        global mauve
        try:
            import mauve
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="Mauve", package_name="mauve-text"))
          
        
    def _info(self):
        return MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/krishnap25/mauve",
            inputs_description=_KWARGS_DESCRIPTION,
            upper_bound=1,
            lower_bound=0,
            features=self._default_features,
            codebase_urls=["https://github.com/krishnap25/mauve"],
            reference_urls=[
                "https://arxiv.org/abs/2102.01454",
                "https://github.com/krishnap25/mauve",
            ],
            license=_LICENSE,
        )
    def compute_mauve(
        self,
        predictions,
        references,
    ):
        scores = []
        for pred, ref in zip(predictions, references):
            out = mauve.compute_mauve(
                p_text=predictions,
                q_text=references,
                p_features=self.p_features,
                q_features=self.q_features,
                p_tokens=self.p_tokens,
                q_tokens=self.q_tokens,
                num_buckets=self.num_buckets,
                pca_max_data=self.pca_max_data,
                kmeans_explained_var=self.kmeans_explained_var,
                kmeans_num_redo=self.kmeans_num_redo,
                kmeans_max_iter=self.kmeans_max_iter,
                featurize_model_name=self.featurize_model_name,
                device_id=self.device_id,
                max_text_length=self.max_text_length,
                divergence_curve_discretization_size=self.divergence_curve_discretization_size,
                mauve_scaling_factor=self.mauve_scaling_factor,
                verbose=self.verbose,
                seed=self._seed,
            )
            scores.append(out.mauve)
        return scores


    def _compute_single_pred_single_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):
        """
        Compute the mauve score for a single prediction and a single reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a single text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            segment_scores (bool, optional): Whether to return scores per instance.
        """
        scores = self.compute_mauve(predictions=predictions, references=references)
        return {
            "score":  float(np.mean(scores)),
        }

    def _compute_single_pred_multi_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):
        """
        Compute the mauve score for a single prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            segment_scores (bool, optional): Whether to return scores per instance.
        """
        reduced_scores = []
        for refs, pred in zip(references, predictions):
            extended_preds = [pred for _ in range(len(refs))]
            reduced_scores.append(reduce_fn(self.compute_mauve(extended_preds, refs)))
        return {
            "score":  float(np.mean(reduced_scores)), "reduced_scores" : reduced_scores
        }

    def _compute_multi_pred_multi_ref(
        self,
            predictions: EvaluationInstance,
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):
        """
        Compute the mauve score for multiple prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a multiple text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            segment_scores (bool, optional): Whether to return scores per instance.
        """
        reduced_scores = []
        for preds, refs in zip(predictions, references):
            scores= []
            for pred in preds:
                scores.append(self._compute_single_pred_multi_ref(
                        predictions=[pred],
                        references=[refs],
                        reduce_fn=reduce_fn,
                )["reduced_scores"])
            reduced_scores.append(reduce_fn(scores))
        return {
             "score":  float(np.mean(reduced_scores)), "reduced_scores" : reduced_scores
        }