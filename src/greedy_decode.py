import string
from typing import List, Union

import torch


class GreedyDecoder:
    def __init__(self, model, tokenizer, device: torch.device, max_length: int):
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length

    def decode(self, src: torch.Tensor):
        """

        :param src: size (1, seq_length)
        :return:
        """
        src = src.to(self._device)
        num_tokens = src.shape[1]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(self._device)

        memory = self._model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(self._tokenizer.bos_token_id).type(torch.long).to(self._device)

        for i in range(self._max_length - 1):
            tgt_mask = self._model.generate_square_subsequent_mask(ys.size(1)).type(torch.bool)
            out = self._model.decode(ys, memory, tgt_mask)
            prob = self._model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word == self._tokenizer.eos_token_id:
                break

        return ys[0].tolist()

    def translate(self, dataset: Union[str, List[str]]):
        if type(dataset) == str:
            list_sentences = [dataset]
        else:
            list_sentences = dataset
        list_res = []

        for sentence in list_sentences:
            src = torch.tensor([self._tokenizer.encode(sentence, add_special_tokens=True)])
            ys = self.decode(src)
            list_res.append(convert_tokens([self._tokenizer.convert_ids_to_tokens(i, skip_special_tokens=True)
                                            for i in ys]))

        if type(dataset) == str:
            return list_res[0]
        else:
            return list_res


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
