# coding=utf-8
# Copyright 2021 Open Business Software Solutions, The HuggingFace evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Nubia metric. The part of this file is adapted from metric implementations
of evaluate package. See https://github.com/huggingface/evaluate/blob/master/metrics/ """
from typing import Callable, Dict, List

import evaluate
import numpy as np

from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core.utils import requirement_message
from nlgmetricverse.metrics._core import MetricForLanguageGeneration

_CITATION = """\
@misc{kane2020nubia,
    title={NUBIA: NeUral Based Interchangeability Assessor for Text Generation},
    author={Hassan Kane and Muhammed Yusuf Kocyigit and Ali Abdalla and Pelkins Ajanoh and Mohamed Coulibali},
    year={2020},
    eprint={2004.14667},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
NUBIA is a SoTA evaluation metric for text generation. It stands for NeUral Based Interchangeability Assessor. In addition to returning 
an interchangeability score, NUBIA also returns scores for semantic relation, contradiction, irrelevancy, logical 
agreement, and grammaticality.
Nubia is composed of three modules.
- The first is **neural feature extraction**. The three main neural features that power the metric are semantic similarity, 
  logical inference, and sentence legibility. These are extracted by exposing layers from powerful (pretrained) language 
  models: RoBERTa STS for semantic similarity, RoBERTa MNLI for logical inference, and GPT-2 for sentence legibility.
- The second module is the **aggregator**. This module is trained to approximate a function mapping input neural features to 
  a quality score that reflects how interchangeable the sentences are. The objective is to come as close as possible to human evaluation.
- The final module is **calibration**. This is necessary because the aggregator is not bound between 0 and 1, nor does a regressed 
  score comparing a reference sentence with itself always ouput 1. So to calibrate, the output is normalized against the score of the 
  reference sentence compared with itself, and bound between 0 and 1.
"""

_KWARGS_DESCRIPTION = """\
Args:
    predictions: list of predictions to score. Each prediction should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each reference should be a string with tokens separated by spaces.
    segment_scores: If True, return scores for each segment.
Returns:
    'nubia_score': nubia score.
    'semantic_relation': semantic relation score.
    'irrelevancy': irrelevancy score.
    'contradiction': contradiction score.
    'logical_agreement': logical agreement score.
    'segment_scores': segment scores.

Examples:
    >>> from nlgmetricverse import NLGMetricverse, load_metric
    >>> predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
    >>> references = ["Peace at home, peace in the world.", "The cat is playing on the mat."]
    >>> scorer = NLGMetricverse(metrics=load_metric("nubia"))
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    {"nubia": {'nubia_score': 0.9504227034094436, 'semantic_relation': 4.672990322113037/5.0, 'irrelevancy': 0.5306123290210962, 'contradiction': 0.26220036670565605, 'logical_agreement': 99.20719265937805, 'segment_scores': False}}
"""

_LICENSE = """\
Copyright (c) 2020 wl-research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NubiaPlanet(MetricForLanguageGeneration):
    def _download_and_prepare(self, dl_manager):
        global nubia_score
        try:
            import nubia_score
        except ModuleNotFoundError:
            raise ModuleNotFoundError(requirement_message(path="Nubia", package_name="nubia-score"))
        else:
            super(NubiaPlanet, self)._download_and_prepare(dl_manager)
        self.scorer = nubia_score.Nubia()

    # Method to provide information about the Nubia metric.
    def _info(self):
        return MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/wl-research/nubia",
            inputs_description=_KWARGS_DESCRIPTION,
            upper_bound=1,
            lower_bound=0,
            features=self._default_features,
            codebase_urls=["https://github.com/wl-research/nubia"],
            reference_urls=[
                "https://github.com/wl-research/nubia",
                "https://aclanthology.org/2020.evalnlgeval-1.4",
            ],
            license=_LICENSE,
        )

    def _compute_single_pred_single_ref(
            self,
            predictions: EvaluationInstance,
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
         
    ):
        """
        Compute the nubia score for a single prediction and a single reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a single text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            segment_scores (bool, optional): Whether to return scores per instance. Defaults to False.
        """
        scores=[] 
        irrelevancy_scores=[]
        semantic_relation=[]
        logical_agreement=[]
        contradictions=[]
        for pred, ref in zip(predictions, references):
          score= self.scorer.score(pred, ref,get_features=True)
          scores.append(score['nubia_score'])
          irrelevancy_scores.append(score['features']['irrelevancy'])
          contradictions.append(score['features']['contradiction'])
          semantic_relation.append(score['features']['semantic_relation'])
          logical_agreement.append(score['features']['logical_agreement'])

        if not segment_scores:
            scores = float(np.mean(scores))
            irrelevancy_scores = float(np.mean(irrelevancy_scores))
            semantic_relation = float(np.mean(semantic_relation))
            logical_agreement = float(np.mean(logical_agreement))
            contradictions = float(np.mean(contradictions))


        return {
           
            "nubia_score":  scores,
            "semantic_relation": semantic_relation,
            "irrelevancy": irrelevancy_scores,
            "contradiction": contradictions,
            "logical_agreement": logical_agreement,
             "segment_scores": segment_scores,
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
        We combine our prediction/reference pairs and use their single prediction/reference function.
        """
        final_scores=[]
        irrelevancy_scores=[]
        semantic_relation=[]
        logical_agreement=[]
        contradictions=[]
        for pred, refs in zip(predictions, references):
          scores=[]
          features={"contradiction":[],"semantic_relation":[],"irrelevancy":[],"logical_agreement":[]}
          for ref in refs:
              score= self.scorer.score(pred, ref,get_features=True)
              scores.append(score['nubia_score'])
              for feature in features:
                features[feature].append(score['features'][feature])
          final_scores.append(float(reduce_fn(scores)))
          irrelevancy_scores.append(float(reduce_fn(features["irrelevancy"])))
          semantic_relation.append(float(reduce_fn(features["semantic_relation"])))
          logical_agreement.append(float(reduce_fn(features["logical_agreement"])))
          contradictions.append(float(reduce_fn(features["contradiction"])))
        if not segment_scores:
            final_scores = float(np.mean(final_scores))
            irrelevancy_scores = float(np.mean(irrelevancy_scores))
            semantic_relation = float(np.mean(semantic_relation))
            logical_agreement = float(np.mean(logical_agreement))
            contradictions = float(np.mean(contradictions))

       
        return {
            
            "nubia_score":  final_scores,
            "semantic_relation": semantic_relation,
            "irrelevancy": irrelevancy_scores,
            "contradiction": contradictions,
            "logical_agreement": logical_agreement,
             "segment_scores": segment_scores,
        }

    def _compute_multi_pred_multi_ref(
            self,
            predictions: EvaluationInstance,
            references: EvaluationInstance,
            reduce_fn: Callable = None,
            batch_size: int = 4,
            segment_scores: bool = False,
            **kwargs,
    ):
        """
        Like Single Pred/Multi Ref, we pre-combine all possible prediction/reference
        pairs into a list of single prediction/single reference pairs.
        """
        final_scores=[]
        irrelevancy_scores=[]
        semantic_relation=[]
        logical_agreement=[]
        contradictions=[]
        for preds, refs in zip(predictions, references):
            scores=[]       
            features={"contradiction":[],"semantic_relation":[],
            "irrelevancy":[],"logical_agreement":[],"r_contradiction":[],
            "r_semantic_relation":[],"r_irrelevancy":[],"r_logical_agreement":[]}
            for pred in preds:
                reduced_scores=[]
                features["contradiction"] = []
                features["semantic_relation"] = []
                features["irrelevancy"] = []
                features["logical_agreement"] = []
                for ref in refs:
                    score= self.scorer.score(pred, ref,get_features=True)
                    reduced_scores.append(score['nubia_score'])
                    reduced_scores.append(self.scorer.score(pred,ref))
                    for feature in features:
                        if not feature.startswith("r_"):
                             features[feature].append(score['features'][feature])
                scores.append(float(reduce_fn(reduced_scores)))
                features["r_contradiction"].append(float(reduce_fn(features["contradiction"])))
                features["r_semantic_relation"].append(float(reduce_fn(features["semantic_relation"])))
                features["r_irrelevancy"].append(float(reduce_fn(features["irrelevancy"])))
                features["r_logical_agreement"].append(float(reduce_fn(features["logical_agreement"])))
            final_scores.append(float(reduce_fn(scores)))
            irrelevancy_scores.append(float(reduce_fn(features["r_irrelevancy"])))
            semantic_relation.append(float(reduce_fn(features["r_semantic_relation"])))
            logical_agreement.append(float(reduce_fn(features["r_logical_agreement"])))
            contradictions.append(float(reduce_fn(features["r_contradiction"])))  

        if not segment_scores:
            final_scores = float(np.mean(final_scores))
            irrelevancy_scores = float(np.mean(irrelevancy_scores))
            semantic_relation = float(np.mean(semantic_relation))
            logical_agreement = float(np.mean(logical_agreement))
            contradictions = float(np.mean(contradictions))

        return {
            "nubia_score":  final_scores,
            "semantic_relation": semantic_relation,
            "irrelevancy": irrelevancy_scores,
            "contradiction": contradictions,
            "logical_agreement": logical_agreement,
            "segment_scores": segment_scores,
        }