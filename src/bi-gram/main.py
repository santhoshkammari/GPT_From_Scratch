import time

import torch

torch.manual_seed((1337))

with open('../../data/input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])


def train_test_splits(text=None, split_ratio: int = 0.9):
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(split_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data

def get_batch(data):
    idx = torch.randint(len(data)-BLOCKSIZE,(BATCHSIZE,1))
    x = torch.stack([data[i:i+BLOCKSIZE] for i in idx])
    y = torch.stack([data[i+1:i+BLOCKSIZE+1] for i in idx])
    return x,y

class BiGram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size,N_EMB)
        self.head = torch.nn.Linear(N_EMB,vocab_size)
    def forward(self,idx,targets = None):
        embs: torch.Tensor = self.token_embedding_table(idx) #(B,T,C)
        logits:torch.Tensor = self.head(embs)

        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C) # (B*T,C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits,targets)
        else:
            loss = None

        return logits,loss

    def generate(self,idx,max_new_tokens = 10):
        ans = []
        for _ in range(max_new_tokens):
            logits,loss = model(idx)
            logits = logits[:,-1,:] # (B,C)
            probs = torch.nn.functional.softmax(logits,dim=-1)
            idx = torch.multinomial(probs,num_samples=1)
            time.sleep(0.05)
            print(decode([idx.item()]),end="",flush=True)

            ans.append(idx.item())
        return ans






if __name__ == '__main__':
    BATCHSIZE = 8
    BLOCKSIZE = 16
    N_EMB = 128
    N_TRAINING = 150

    train_data,val_data = train_test_splits(text)
    x,y = get_batch(train_data)

    model = BiGram()
    logits,loss = model(x,y)

    opt = torch.optim.AdamW(params=model.parameters(),
                            lr=1e-1)

    ntrains_loss = []
    for ntrains in [3000]: #range(100,5000,50):
        print(f"## {ntrains} ###")
        losses = []
        for _ in range(ntrains):
            x,y = get_batch(train_data)
            logits,loss = model(x,y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        ntrains_loss.append(loss.item())


    print(f"Loss : {loss}")
    import matplotlib.pyplot as plt

    plt.plot(range(len(ntrains_loss)), ntrains_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.savefig('name.png')

    tokens = model.generate(torch.tensor([[1]]),max_new_tokens=500)
    # print(decode(tokens))









