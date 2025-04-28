# Metric Card for Nubia

## Metric Description
NUBIA is a SoTA evaluation metric for text generation. It stands for NeUral Based Interchangeability Assessor. In addition to returning an interchangeability score, NUBIA also returns scores for semantic relation, contradiction, irrelevancy, logical agreement, and grammaticality.

<p align="center">
  <img src="../../../figures/metrics/nubia/nubia.png" width="100%" title="BERTScore recall illustration" alt="">
</p>

Nubia is composed of three modules.
- The first is **neural feature extraction**. The three main neural features that power the metric are semantic similarity, logical inference, and sentence legibility. These are extracted by exposing layers from powerful (pretrained) language models: RoBERTa STS for semantic similarity, RoBERTa MNLI for logical inference, and GPT-2 for sentence legibility.
- The second module is the **aggregator**. This module is trained to approximate a function mapping input neural features to a quality score that reflects how interchangeable the sentences are. The objective is to come as close as possible to human evaluation.
- The final module is **calibration**. This is necessary because the aggregator is not bound between 0 and 1, nor does a regressed score comparing a reference sentence with itself always ouput 1. So to calibrate, the output is normalized against the score of the reference sentence compared with itself, and bound between 0 and 1.

### Inputs
- **predictions** (`list`): list of predictions to score. Each prediction should be a string with tokens separated by spaces.
- **references** (`list`): list of reference for each prediction. Each reference should be a string with tokens separated by spaces.
- **segment_scores** (`bool`): whether to return the scores for each segment of the input text. Defaults to `False`.

### Outputs
- **score** (`float`): the average Nubia score for the text input in the list.
- **semantic_relation** (`float`): the average semantic relation score for the text input in the list.
- **contradiction** (`float`): the average contradiction score for the text input in the list.
- **irrelevancy** (`float`): the average irrelevancy score for the text input in the list.
- **logical_agreement** (`float`): the average logical agreement score for the text input in the list.
- **segment_scores** (`list`): the scores for each segment of the input text. Only returned if `segment_scores` is set to `True`.

### Results from popular papers

## Bounds
Nubia's scores can be any value in <img src="https://render.githubusercontent.com/render/math?math={[0,1]}##gh-light-mode-only">, apart from `semantic_relation`, which can be any value in <img src="https://render.githubusercontent.com/render/math?math={[0,5]}##gh-light-mode-only">, `logical_agreement`, which can be any value in <img src="https://render.githubusercontent.com/render/math?math={[0,100]}##gh-light-mode-only"> and `segment_scores`, which can be True or False.

## Example
```python
from nlgmetricverse import NLGMetricverse, load_metric
predictions=["He agreed to a proposal of mine."]
references=["He gave his agreement to my proposal."]
scorer = NLGMetricverse(metrics=load_metric("nubia"))
scores = scorer(predictions=predictions, references=references, reduce_fn=REDUCTION_FUNCTION)
print(scores)
{
  "nubia": {
    'nubia_score': 0.9504227034094436,
    'semantic_relation': 4.672990322113037/5.0,
    'irrelevancy': 0.5306123290210962,
    'contradiction': 0.26220036670565605,
    'logical_agreement': 99.20719265937805,
    'segment_scores': False
  }
}
```

## Limitations and bias

## Citation
```bibtex
@misc{kane2020nubia,
    title={NUBIA: NeUral Based Interchangeability Assessor for Text Generation},
    author={Hassan Kane and Muhammed Yusuf Kocyigit and Ali Abdalla and Pelkins Ajanoh and Mohamed Coulibali},
    year={2020},
    eprint={2004.14667},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

## Further References