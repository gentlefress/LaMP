# Metric Card for Recall

## Metric Description
This metric is derived from Modified Unigram Precision as a recall metric so that it will compute across references and not across predictions unlike precision. The computation is similar, however, we call this recall since there is no measure called "modified unigram recall".
Recall is the fraction of the common unigrams between the prediction and the references among the reference tokens. It can be computed with:
Recall = # of matching tokens / # of reference tokens

### Inputs
-  **predictions** (`list`): list of predictions to score. Each predictions should be a string with tokens separated by spaces.
-  **references** (`list`): list of reference for each prediction. Each reference should be a string with tokens separated by spaces.

### Outputs
-  **recall**: A dictionary containing the computed recall metric score, that is stored under the key "score".

### Results from popular papers

## Bounds
Recall scores can be any value in <img src="https://render.githubusercontent.com/render/math?math={[0, 1]}##gh-light-mode-only">.

## Examples
```python
from nlgmetricverse import NLGMetricverse, load_metric
predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
references = ["Peace at home, peace in the world.", "The cat is playing on the mat."]
scorer = NLGMetricverse(metrics=load_metric("recall"))
scores = scorer(predictions=predictions, references=references)
print(scores)
{
  "recall": {
    'score': 0.6571428571428571,
  }
}
```

## Limitations and bias

## Citation
```bibtex
@inproceedings{papineni2002bleu,
  title={Bleu: a method for automatic evaluation of machine translation},
  author={Papineni, Kishore and Roukos, Salim and Ward, Todd and Zhu, Wei-Jing},
  booktitle={Proceedings of the 40th annual meeting of the Association for Computational Linguistics},
  pages={311--318},
  year={2002}
}
```

## Further References