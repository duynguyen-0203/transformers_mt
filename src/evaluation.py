import string
from typing import List

from datasets import load_metric
from tqdm import tqdm


class Evaluator:
    def __init__(self, dataset, tokenizer, decoder):
        self._metric = load_metric('bleu')
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._decoder = decoder

        self.predictions = []
        self.references = []

        self._convert_dataset()

    def compute_scores(self):
        assert len(self.predictions) == len(self.references)
        print(self.predictions)
        print(self.references)
        results = self._metric.compute(predictions=self.predictions, references=self.references)

        return results

    def _convert_dataset(self):
        for sample in tqdm(self._dataset.samples):
            self.references.append(convert_tokens(self._tokenizer.tokenize(sample.target, add_special_tokens=False)))
            self.predictions.append(convert_tokens(self._decoder.translate(sample.source)))

def convert_tokens(tokens: List[str]):
    res = []
    for i, token in enumerate(tokens):
        if token.startswith(chr(9601)):
            res.append(token[1:])
        elif token in string.punctuation:
            res.append(token)
        elif len(res) != 0:
            res[-1] += token
        else:
            res.append(token)

    return res
