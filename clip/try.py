import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

a = torch.rand((7,8))
a_v = torch.rand((7,5))
# b = torch.rand(8)
b = torch.rand((7,2))
print("a:",a,"\nb:",b)
c = torch.cosine_similarity(a,b)
print("c:",c)
distance = torch.cosine_similarity(a,b)
print(a_v[torch.topk(distance, k=3)[1],:])
