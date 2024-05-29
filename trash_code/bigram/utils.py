from typing import Tuple, Dict

import torch


class BiGram:
    def __init__(self, chars=None,text = None,**kwargs,) -> None:
        self.text = text
        self.batch_size = kwargs.get("batch_size", None)
        self.block_size = kwargs.get("block_size", None)
        self.device = kwargs.get("device", None)
        if chars is None:
            chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: "".join([self.itos[i] for i in l])

    def train_test_splits(self, text=None, split_ratio: int = 0.9) -> Tuple:
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(split_ratio * len(data))
        train_data = data[:n]
        val_data = data[n:]
        return train_data, val_data

    def get_batch(self, split=None):
        train_data, val_data = self.train_test_splits(self.text)
        data = train_data if split=="train" else val_data
        ix = torch.randint(len(data)-self.block_size,(self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x,y = x.to(self.device),y.to(self.device)
        return x,y


    def print_params(self):
        for k,v in self.__dict__.items():
            if k not in ["stoi", "itos", "encode", "decode","text"]:
                print(f"{k} = {v}")




if __name__ == '__main__':
    pass
    # b = BiGram(text="santhosdsafadf2342342bh",batch_size=5,block_size=2)
    # b.print_params()
    # b.get_batch(text='sant2fasfadfhosh')
    # b.train_test_splits()
    # b.get_batch()