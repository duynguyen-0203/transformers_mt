from .entities import Dataset


class Reader:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def read(self, src_path: str, tgt_path: str, data_name: str, max_length: int):
        dataset = Dataset(data_name, self._tokenizer, max_length)

        src_data = read_text(src_path)
        tgt_data = read_text(tgt_path)
        assert len(src_data) == len(tgt_data)

        for (src_sent, tgt_sent) in zip(src_data, tgt_data):
            dataset.add_sample(src_sent.strip(), tgt_sent.strip())

        return dataset


def read_text(path: str):
    with open(path, mode='r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    if lines[-1] == '':
        lines.pop()

    return lines
