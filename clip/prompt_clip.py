import torch
import torch.nn as nn
import torch.nn.functional as F

# num_prompt 代表提示池中提示的数目
# num_select 代表选择嵌入x的数目
# prompt_dim prompt的value的维度
class PromptClip(nn.Module):
    def __init__(self, num_prompt, num_select, prompt_dim):
        super(PromptClip, self).__init__()
        self.num_prompt = num_prompt
        self.num_select = num_select
        self.prompt_dim = prompt_dim

        # 使用torch.nn.Parameter将不可训练的tensor转换为可训练的tensor并在该类中进行注册
        self.prompt_k = nn.Parameter(torch.rand((self.num_prompt,512),requires_grad=True))
        self.prompt_v = nn.Parameter(torch.rand((self.num_prompt,prompt_dim),requires_grad=True))
        # for i in range(self.num_prompt):
        #     k_t = torch.rand(512)
        #     v_t = torch.rand(prompt_dim)
        #     self.prompt_pool[k_t] = v_t
    
    def forward(self, x):
        # 计算余弦相似度
        distance = torch.cosine_similarity(self.prompt_k,x)
        # 选取相似度最大的num_select（k）个提示键对应的v并返回，[0]代表返回相似度值，[1]代表返回下标 # 不拉直了
        # 同时返回相似度值
        return self.prompt_v[torch.topk(distance, k=self.num_select)[1],:], torch.topk(distance, k=self.num_select)[0].sum()/self.num_select

    def get_prompt(self):
        # print("prompt_k:",self.prompt_k)
        # print("prompt_v:",self.prompt_v)
        return self.prompt_k, self.prompt_v



