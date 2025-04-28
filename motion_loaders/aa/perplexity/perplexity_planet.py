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
""" Perplexity metric. The part of this file is adapted from Perplexity implementation
of evaluate package. See
https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py """

import torch
import evaluate
import numpy as np
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Callable, Dict

from evaluate import logging
from nlgmetricverse.metrics import EvaluationInstance
from nlgmetricverse.utils.metric_info import MetricInfo
from nlgmetricverse.metrics._core import MetricForLanguageGeneration

_CITATION = """\
@article{jelinek1977perplexity,
  title={Perplexity—a measure of the difficulty of speech recognition tasks},
  author={Jelinek, Fred and Mercer, Robert L and Bahl, Lalit R and Baker, James K},
  journal={The Journal of the Acoustical Society of America},
  volume={62},
  number={S1},
  pages={S63--S63},
  year={1977},
  publisher={Acoustical Society of America}
}
"""

_DESCRIPTION = """
Given a model and an input text sequence, perplexity measures how likely the model is to generate the input text sequence.
So, perplexity is a common metric for directly assessing how well a selection of text matches the distribution of text that the 
input model was trained on.
Perplexity is defined as the exponentiated average negative log-likelihood of a sequence. If we have a tokenized sequence 
X = (x_0, x_1, ..., x_t) then, the perplexity of X is, PPL(X)=exp( (1/t)summation{t}and{i} log p_{theta}(x_i|x_{<i}))
where log p_{theta}(x_i|x_{<i}) is the log-likelihood of the ith token conditioned on the preceding tokens x_{<i} according to our model.
So, the final perplexity is obtained by averaging all the obtained log-likelihoods (geometric mean).

Intuitively, it can be thought of as an evaluation of the model's ability to predict uniformly among the set of specified tokens in a corpus.
Importantly, this means that the tokenization procedure has a direct impact on a model's perplexity which should always be taken into consideration when comparing different models.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: List of str or List of List of str. Each prediction should be a string with tokens separated by spaces.
    references: List of str or List of List of str. Each prediction should be a string with tokens separated by spaces.
    batch_size: int, default 16. The batch size to use for computing perplexity.

Returns:
    'perplexities': List of float. The perplexity score for each prediction.
    'mean_perplexity': float. The mean perplexity score for all predictions.

Examples:
    >>> from nlgmetricverse import NLGMetricverse, load_metric
    >>> predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
    >>> references = ["Peace at home, peace in the world.", "The cat is playing on the mat."]
    >>> scorer = NLGMetricverse(metrics=load_metric("perplexity"))
    >>> scores = scorer(predictions=predictions, references=references)
    >>> print(scores)
    {"perplexity": {"perplexities": [16.518978118896484, 16.501928329467773], "mean_perplexity": 16.51045322418213 }}
"""

_LICENSE= """ """

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PerplexityPlanet(MetricForLanguageGeneration):
    def __init__(
            self,
            resulting_name: str = None,
            compute_kwargs: Dict = None,
            model_id = 'gpt2',
            add_start_token: bool = False,
            device = "cpu",
            **kwargs,
    ):
        self.model_id = model_id
        self.add_start_token = add_start_token
        self.device = device
        super().__init__(resulting_name=resulting_name, compute_kwargs=compute_kwargs, **kwargs)

    def _download_and_prepare(self, dl_manager) -> None:
        """
        Downloads and import the computation of Perplexity score from the implementation
        of Perplexity computation. The code is sourced from a specific
        commit on the master branch, in order to keep things stable. See
        https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py

        """
        if self.device is not None:
            assert self.device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
            if self.device == "gpu":
                self.device = "cuda"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.pretrained_model = self.model.to(self.device)
        self.tokenizer =  AutoTokenizer.from_pretrained(self.model_id)
        
    def _info(self):
        return MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage="https://github.com/huggingface/evaluate/blob/main/metrics/perplexity",
            inputs_description=_KWARGS_DESCRIPTION,
            upper_bound=300,
            lower_bound=0,
            features=self._default_features,
            codebase_urls=["https://github.com/huggingface/evaluate/blob/main/metrics/perplexity"],
            reference_urls=[
                "https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/",
                "https://huggingface.co/docs/transformers/perplexity",
            ],
            license=_LICENSE,
        )
    def perplexity_score(self, predictions: EvaluationInstance, batch_size: int = 16):
        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if self.add_start_token:
            # leave room for <BOS> token to be added:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = self.pretrained_model.config.max_length - 1
        else:
            max_tokenized_len = self.pretrained_model.config.max_length

        encodings = self.tokenizer(
            predictions,
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        # check that each input is long enough:
        if self.add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if self.add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(self.device)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.pretrained_model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp2(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}
    def _compute_single_pred_single_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            batch_size: int = 16,
            **kwargs,
    ):
        """
        Compute the perplexity score for a single prediction and a single reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a single text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            batch_size (int, optional): The batch size to use for computing perplexity.
        """
        scores = self.perplexity_score(predictions=predictions, batch_size=batch_size)
        return scores

    def _compute_single_pred_multi_ref(
          self,
            predictions: EvaluationInstance, 
            references: EvaluationInstance,
            batch_size: int = 16,
            reduce_fn: Callable = None,
            **kwargs,
    ):  
        """
        Compute the perplexity score for a single prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a single text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            batch_size (int, optional): The batch size to use for computing perplexity.
        """
        scores = self.perplexity_score(predictions=predictions, batch_size=batch_size)
        return scores

    def _compute_multi_pred_multi_ref(
        self,
            predictions: EvaluationInstance,
            references: EvaluationInstance,
            batch_size: int = 16,
            reduce_fn: Callable = None,
            segment_scores: bool = False,
            **kwargs,
    ):
        """
        Compute the perplexity score for multiple prediction and multiple reference.
        Args:
            predictions (EvaluationInstance): A EvaluationInstance containing a multiple text sample for prediction.
            references (EvaluationInstance): A EvaluationInstance containing a multiple text sample for reference.
            reduce_fn (Callable, optional): A function to apply reduction to computed scores.
            batch_size (int, optional): The batch size to use for computing perplexity.
        """
        inputList = []
        for prediction in predictions:
            inputList += prediction
        scores = self.perplexity_score(predictions=inputList, batch_size=batch_size)
        return scores