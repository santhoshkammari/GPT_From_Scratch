import torch
import torch.nn as nn
from torch.nn import functional as F

from trash_code.bigram.configs.bigram_config import BATCH_SIZE, BLOCK_SIZE, MAX_ITERS, EVAL_INTERVAL, LEARNING_RATE, DEVICE, \
    EVAL_ITERS, EMBEDDING_SIZE

from trash_code.bigram.utils import BiGram

# hyperparameters
batch_size = BATCH_SIZE
block_size = BLOCK_SIZE
max_iters = MAX_ITERS
eval_interval = EVAL_INTERVAL
learning_rate = LEARNING_RATE
device = DEVICE if torch.cuda.is_available() else "cpu"
eval_iters = EVAL_ITERS
n_embd = EMBEDDING_SIZE
# --------------

torch.manual_seed((1337))

with open('data/input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

print("################################")
print(f"{chars=}")
print(f"{len(chars)=}")
print("################################")




class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)

    def forward(self, idx, targets=None):
        """
        return logits, loss
        """
        tok_emb = self.token_embedding_table(idx)
        logits = self.lm_head(tok_emb)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx
def main(input_text = None):
    model = BigramLanguageModel(vocab_size).eval()
    m = model.to(device)
    sample = BiGram(
        text=text,
        block_size=block_size,
        batch_size=batch_size,
        device=device
    )
    sample.print_params()
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-1)

    generated_text = sample.decode(m.generate(idx = torch.zeros((1,1),dtype=torch.long).to(device), max_new_tokens=300)[0].tolist())

    print(generated_text)

    for steps in range(50):  # increase number of steps for good results...

        xb, yb = sample.get_batch('train')

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"loss = {loss.item()}")

    # xb, yb = sample.get_batch(
    #     split='train'
    # )
    # logits,loss = m(xb,yb)

    generated_text = sample.decode(m.generate(idx = torch.zeros((1,1),dtype=torch.long).to(device), max_new_tokens=300)[0].tolist())
    print('================================================================')
    print(generated_text)

    return generated_text



if __name__ == '__main__':
    main()