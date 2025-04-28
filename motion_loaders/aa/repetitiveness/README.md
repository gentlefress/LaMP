# Metric card for Repetitiveness

## Metric description
The [repetition problem](https://github.com/fuzihaofzh/repetition-problem-nlg) has been observed in nearly all text generation models. This problem is, unfortunately, caused by the traits of our language itself. There exists too many words predicting the same word as the subsequent word with high probability. Consequently, it is easy to go back to that word and form repetitions.

The Repetitiveness metric evaluates how many n-grams are repeated on average in the hypothesis sentences, the result is normalized by the length of the sentence.

### Inputs
-  **predictions** (`list` of `str`): List of generated text predictions.
-  **references** (`list` of `str`): List of reference texts for comparison.

### Outputs
-  **repetitiveness**: A dictionary containing the computed Repetitiveness metric score. The score is stored under the key "score".

### Results from popular papers

## Bounds
The Repetitiveness score can be any value in <img src="https://render.githubusercontent.com/render/math?math={[0,1]}##gh-light-mode-only">.

## Example
```python
from nlgmetricverse import NLGMetricverse, load_metric
predictions = ["Peace in the dormitory, peace in the world.", "There is a cat on the mat."]
references = ["Peace at home, peace in the world.", "The cat is playing on the mat."]
scorer = NLGMetricverse(metrics=load_metric("repetitiveness"))
scores = scorer(predictions=predictions, references=references)
print(scores)
{
  "repetitiveness": {
    'score': 0.85
  }
}
```

## Limitations and bias

## Citation
```bibtex
@inproceedings{fu2020a,
  title={A Theoretical Analysis of the Repetition Problem in Text Generation.},
  author={Fu, Zihao and Lam, Wai and So, Anthony Man-Cho and Shi, Bei },
  booktitle={Thirty-Fifth AAAI Conference on Artificial Intelligence},
  year={2021}
}
```

## Further References