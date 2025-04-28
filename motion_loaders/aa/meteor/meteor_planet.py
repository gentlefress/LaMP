# coding=utf-8
# Copyright 2020 Open Business Software Solutions, The HuggingFace evaluate Authors.
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
""" METEOR metric. The part of this file is adapted from HuggingFace's
evaluate package implementation of METEOR metric. See
https://github.com/huggingface/evaluate/blob/master/metrics/meteor/meteor.py
"""

from typing import Callable, Dict, Tuple

import evaluate
import numpy as np
from nltk import __version__ as NLTK_VERSION
from nltk.translate import meteor_score
from packaging.version import Version

if Version(NLTK_VERSION) < Version("3.6.6"):
    raise EnvironmentError(
        f"Version constraints does not hold for 'nltk', expected version >=3.6.6, got {NLTK_VERSION}."
    )

from nlgmetricverse.collator import Collator
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core import MetricForLanguageGeneration
from nlgmetricverse.tokenizer import DefaultTokenizer, TokenizerWrapper

_CITATION = """\
@inproceedings{banarjee2005,
  title     = {{METEOR}: An Automatic Metric for {MT} Evaluation with Improved Correlation with Human Judgments},
  author    = {Banerjee, Satanjeev  and Lavie, Alon},
  booktitle = {Proceedings of the {ACL} Workshop on Intrinsic and Extrinsic Evaluation Measures for Machine Translation
                and/or Summarization},
  month     = jun,
  year      = {2005},
  address   = {Ann Arbor, Michigan},
  publisher = {Association for Computational Linguistics},
  url       = {https://www.aclweb.org/anthology/W05-0909},
  pages     = {65--72},
}
"""

_DESCRIPTION = """\
METEOR (Metric for Evaluation of Translation with Explicit ORdering) is an automatic metric originally designed to
address some of the issues found in BLEU and has been widely used for evaluating machine translation models.
Compared to BLEU, which only measures precision, METEOR is based on the harmonic mean of the unigram precision and
recall, in which recall is weighted higher than precision. It is based on a generalized concept of unigram matching
between the machine-produced translation and human-produced reference translations. METEOR has several variants that
extend exact word matching that most of the metrics in this category do not include, such as stemming and
WordNet-based synonym matching (if English is the target). These variants address the problem of reference translation
variability, allowing for morphological variants and synonyms to be recognized as valid translations. The metric has
been found to produce good correlation with human judgments at the sentence or segment level (Agarwal & Lavie, 2008).
This differs from BLEU in that METEOR is explicitly designed to compare at the sentence level rather than the corpus
level. Once all generalized unigram matches between the two strings have been found, METEOR computes a score for this
matching using a combination of unigram-precision, unigram-recall, and a measure of fragmentation that is designed to
directly capture how well-ordered the matched words in the machine translation are in relation to the reference. To
take into account longer n-gram matches, a penalty factor is introduced: the longer the adjacent mappings between the
candidate and the reference, the fewer chunks there are (a translation that is identical to the reference will give
just one chunk). The penalty has the effect of reducing the harmonic mean by up to 50% if there are no bigram or
longer matches.
- precision: $P=\frac{m}{w_t}$, where $m$ is the number of unigrams in the hypothesis that are also found in the
  reference, and $w_t$ is the number of unigrams in the hypothesis
- recall: $R=\frac{m}{w_r}$, where $w_r$ is the number of unigrams in the reference
- harmonic mean: $F_{mean}=\frac{10PR}{R+9P}$, with recall weighted 9 times more than precision
- penalty: $p=0.5(\frac{c}{u_m})^3$, where $c$ is the number of chunks, and $u_m$ is the number of unigrams that have
  been mapped. $\frac{c}{m}$ is also known as fragmentation fraction. The exponential value determines the functional
  relation between fragmentation and the penalty; it is also known as beta.
- final score: $M=F_{mean}(1-p)$
To calculate a score over a whole corpus, or collection of segments, the aggregate values for P, R and p are taken and
then combined using the same formula. The algorithm also works for comparing a candidate translation against more than
one reference translations. In this case the algorithm compares the candidate against each of the references and selects
the highest score (f_reduce=max).
Example:
- reference: "the cat sat on the mat"
- hypothesis: "on the mat sat the cat"
- P = 1, R = 1, F_mean = 1.0000
- p = 0.5*(6/6)^3 = 0.5000
- M = 1.0000*(1-0.5000) = 0.5000
"""

_KWARGS_DESCRIPTION = """
Computes METEOR score of translated segments against one or more references.
Args:
    predictions (list): list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    references (list): list of reference for each prediction. Each reference
        should be a string with tokens separated by spaces.
    alpha (float): parameter for controlling relative weights of precision and recall. Default: 0.9
    beta (float): parameter for controlling shape of penalty as a function of fragmentation. Default: 3
    gamma (float): relative weight assigned to fragmentation penalty. Default: 0.5
Returns:
    score (float): METEOR score.
Examples:
    >>> scorer = NLGMetricverse(metrics=load_metric("meteor"))
    >>> predictions = ["the cat sat on the mat"]
    >>> references = ["on the mat sat the cat"]
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    { "total_items": 1, "empty_items": 0, "meteor": { "score": 0.5 } }
"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MeteorPlanet(MetricForLanguageGeneration):
    def __init__(self, resulting_name: str = None, compute_kwargs: Dict = None, **kwargs):
        self.should_change_resulting_name = resulting_name is None
        self.tokenizer = DefaultTokenizer()
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _info(self):
        return MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            upper_bound=1,
            lower_bound=0,
            features=self._default_features,
            codebase_urls=["https://github.com/nltk/nltk/blob/develop/nltk/translate/meteor_score.py"],
            reference_urls=[
                "https://www.nltk.org/api/nltk.translate.html#module-nltk.translate.meteor_score",
                "https://en.wikipedia.org/wiki/METEOR",
            ],
        )

    def _download_and_prepare(self, dl_manager):
        import nltk

        nltk.download("wordnet", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    def _preprocess(self, predictions: Collator, references: Collator) -> Tuple[Collator, Collator]:
        tokenizer_wrapper = TokenizerWrapper(self.tokenizer)
        return tokenizer_wrapper.tokenize(predictions, references)

    def _compute_single_pred_single_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, alpha=0.9, beta=3, gamma=0.5
    ):
        """
        Compute the meteor score for a single prediction and a single reference.
        Args:
            predictions (Collator): A Collator containing a single text sample for prediction.
            references (Collator): A Collator containing a single text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            alpha (float): parameter for controlling relative weights of precision and recall. Default: 0.9
            beta (float): parameter for controlling shape of penalty as a function of fragmentation. Default: 3
            gamma (float): relative weight assigned to fragmentation penalty. Default: 0.5
        """
        scores = [
            meteor_score.single_meteor_score(ref, pred, alpha=alpha, beta=beta, gamma=gamma)
            for ref, pred in zip(references, predictions)
        ]
        return {"score": self._reduce_scores(scores, reduce_fn=np.mean)}

    def _compute_single_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, alpha=0.9, beta=3, gamma=0.5
    ):
        """
        Compute the meteor score for a single prediction and multiple reference.
        Args:
            predictions (Collator): A Collator containing a single text sample for prediction.
            references (Collator): A Collator containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            alpha (float): parameter for controlling relative weights of precision and recall. Default: 0.9
            beta (float): parameter for controlling shape of penalty as a function of fragmentation. Default: 3
            gamma (float): relative weight assigned to fragmentation penalty. Default: 0.5
        """
        scores = [
            meteor_score.meteor_score(references=ref, hypothesis=pred, alpha=alpha, beta=beta, gamma=gamma)
            for ref, pred in zip(references, predictions)
        ]
        return {"score": self._reduce_scores(scores, reduce_fn=np.mean)}

    def _compute_multi_pred_multi_ref(
        self, predictions: Collator, references: Collator, reduce_fn: Callable = None, alpha=0.9, beta=3, gamma=0.5
    ):
        """
        Compute the meteor score for multiple prediction and multiple reference.
        Args:
            predictions (Collator): A Collator containing a multiple text sample for prediction.
            references (Collator): A Collator containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            alpha (float): parameter for controlling relative weights of precision and recall. Default: 0.9
            beta (float): parameter for controlling shape of penalty as a function of fragmentation. Default: 3
            gamma (float): relative weight assigned to fragmentation penalty. Default: 0.5
        """
        scores = []
        for pred, ref in zip(predictions, references):
            score = [
                meteor_score.meteor_score(references=ref, hypothesis=p, alpha=alpha, beta=beta, gamma=gamma)
                for p in pred
            ]
            reduced_score = reduce_fn(score)
            scores.append(reduce_fn(reduced_score))
        return {"score": self._reduce_scores(scores, reduce_fn=np.mean)}
