from collections import Counter

import torch

weights = torch.tensor([0.2,0.2,0.6],dtype=torch.float)

print(weights.tolist())
for _ in range(100):
    ans = []
    for __ in range(1):
        ans.append(",".join([str(i) for i in torch.multinomial(weights,1,replacement=True).tolist()]))
    print(_+1,Counter(ans))
