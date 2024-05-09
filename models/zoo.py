import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np
import copy
import clip

def ortho_penalty(t, angle=0):
    temp_matrix = (t @t.T - torch.eye(t.shape[0]).cuda())**2  # t 已经归一化过
    result = torch.where(temp_matrix>angle, temp_matrix, 0).mean()
    
    return result

# father class
class PromptModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def gram_schmidt(self, vv, pt):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)


        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)

        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def cal_logits_cosine(self, x, temperature, test_feature=None):

        image_features = x / x.norm(dim=-1, keepdim=True)

        if test_feature is not None:
            logits = (image_features.half() @ test_feature.T.half().cuda()) / temperature
        else:
            logits = (image_features.half() @ self.text_features.T.half().cuda()) / temperature

        return logits


# ================================================================ Our method =====================================================
class MSP(PromptModel):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim=768, text_features=None, args=None):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        
        self.text_features = text_features
        self._init_smart(emb_d, prompt_param)
        self.n_classes = text_features.shape[0]

        self.adap_angle = torch.nn.Parameter(torch.FloatTensor([0.1]), requires_grad=True)


        pt = int(self.n_classes / self.n_tasks) * self.lp_size
        pt_task = int(self.task_pool_size / (self.n_tasks))
        # prompt init
        for e in self.e_layers:
            e_l = self.lp_length
            p = tensor_prompt(self.lp_size*self.n_classes, e_l, self.emb_d)
            k = tensor_prompt(self.lp_size*self.n_classes, self.key_d)
            a = tensor_prompt(self.lp_size*self.n_classes, self.key_d)
            p = self.gram_schmidt(p, pt)
            k = self.gram_schmidt(k, pt)
            a = self.gram_schmidt(a, pt)
            setattr(self, f'lp_{e}',p)
            setattr(self, f'lk_{e}',k)
            setattr(self, f'la_{e}',a)


        if self.task_pool_size > 0:
            for e in self.g_layers:
                p_t = tensor_prompt(self.task_pool_size, self.tp_length, emb_d)
                k_t = tensor_prompt(self.task_pool_size, self.key_d)
                a_t = tensor_prompt(self.task_pool_size, self.key_d)
                p_t = self.gram_schmidt(p_t, pt_task)
                k_t = self.gram_schmidt(k_t, pt_task)
                a_t = self.gram_schmidt(a_t, pt_task)
                setattr(self, f'tp_{e}',p_t)
                setattr(self, f'tk_{e}',k_t)
                setattr(self, f'ta_{e}',a_t)


    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.lp_size = int(prompt_param[0])
        self.lp_length = int(prompt_param[1])
        self.e_layers = [0,1,2,3,4]  

        # self.e_layers = [0,1] 
        self.g_layers = [3,4,5,6]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]
        self.cos_angle = prompt_param[3] 
        self.select_num = int(prompt_param[4]) 

        self.task_pool_size = int(prompt_param[5]) 
        self.tp_length = int(prompt_param[6]) 
    
    def process_task_count(self):
        self.task_count += 1

        pt = int(self.n_classes / self.n_tasks) * self.lp_size
        pt_task = int(self.task_pool_size / (self.n_tasks))
        for e in self.e_layers:
            K = getattr(self,f'lk_{e}')
            A = getattr(self,f'la_{e}')
            P = getattr(self,f'lp_{e}')
            k = self.gram_schmidt(K, pt)
            a = self.gram_schmidt(A, pt)
            p = self.gram_schmidt(P, pt)
            setattr(self, f'lp_{e}',p)
            setattr(self, f'lk_{e}',k)
            setattr(self, f'la_{e}',a)

        if self.task_pool_size > 0:
            for e in self.g_layers:
                K_t = getattr(self,f'tk_{e}')
                A_t = getattr(self,f'ta_{e}')
                P_t = getattr(self,f'tp_{e}')
                k_t = self.gram_schmidt(K_t, pt_task)
                a_t = self.gram_schmidt(A_t, pt_task)
                p_t = self.gram_schmidt(P_t, pt_task)
                setattr(self, f'tp_{e}',p_t)
                setattr(self, f'tk_{e}',k_t)
                setattr(self, f'ta_{e}',a_t)


    

    def forward(self, x_querry, l, x_block, train=False, task_id=None):

        # e prompts
        e_valid = False
        g_valid = False

        sim_loss = 0

        if l in self.g_layers or l in self.e_layers:
            
            # task_prompt
            if self.task_pool_size > 0 and l in self.g_layers:
                g_valid = True
                K_t = getattr(self,f'tk_{l}') 
                A_t = getattr(self,f'ta_{l}') 
                p_t = getattr(self,f'tp_{l}')

                pt_task = int(self.task_pool_size / (self.n_tasks))
                s_t = int(self.task_count * pt_task) 
                f_t = int((self.task_count + 1) * pt_task) 

                # freeze/control past tasks
                if train:
                    if self.task_count > 0:
                        K_t = torch.cat((K_t[:s_t].detach().clone(),K_t[s_t:f_t]), dim=0)
                        A_t = torch.cat((A_t[:s_t].detach().clone(),A_t[s_t:f_t]), dim=0)
                        p_t = torch.cat((p_t[:s_t].detach().clone(),p_t[s_t:f_t]), dim=0)
                    else:
                        K_t = K_t[s_t:f_t]
                        A_t = A_t[s_t:f_t]
                        p_t = p_t[s_t:f_t]
                else:
                    K_t = K_t[0:f_t]
                    A_t = A_t[0:f_t]
                    p_t = p_t[0:f_t]
                        


                a_querry_t = torch.einsum('bd,kd->bkd', x_querry, A_t) 
                q_t = nn.functional.normalize(a_querry_t, dim=2) 
                n_K_t = nn.functional.normalize(K_t, dim=1)

                # k_size = n_K_t.shape[0]
                mask_text_features_t = []
                for i in range(self.task_count+1):

                    tf_start = int(self.n_classes / self.n_tasks)*self.task_count
                    tf_end = int(self.n_classes / self.n_tasks)*(self.task_count + 1)
                    tmp_text_feature = self.text_features[tf_start:tf_end ,:].to(torch.float32).cuda()

                    tp_start = i*pt_task
                    tp_end = (i+1)*pt_task
                    tmp_text_mask = n_K_t[tp_start:tp_end , :]

                    temp_mask_text_features_t = torch.mean(torch.einsum('kd,cd->kcd', tmp_text_mask, tmp_text_feature), dim=1) #'kcd->kd' 所有c类文本特征的均值
                    mask_text_features_t.append(temp_mask_text_features_t)

                mask_text_features_t = torch.cat(mask_text_features_t, dim=0) 
                aq_k_t = torch.einsum('bkd,kd->bk', q_t, mask_text_features_t)
                P_t = torch.einsum('bk,kld->bld', aq_k_t, p_t)



            # label prompt
            if l in self.e_layers:
                e_valid = True
                # B, C = x_querry.shape

                K = getattr(self,f'lk_{l}') 
                A = getattr(self,f'la_{l}') 
                p = getattr(self,f'lp_{l}')

                pt = int(self.n_classes / self.n_tasks) * self.lp_size 
                s = int(self.task_count * pt) 
                f = int((self.task_count + 1) * pt)
                
                # freeze/control past tasks
                if train:
                    if self.task_count > 0:
                        K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0) 
                        A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                        p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)

                    else:
                        K = K[s:f]
                        A = A[s:f]
                        p = p[s:f]

                else:
                    K = K[0:f]
                    A = A[0:f]
                    p = p[0:f]


                a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
                q = nn.functional.normalize(a_querry, dim=2)
                n_K = nn.functional.normalize(K, dim=1) 
                mask_text_features = torch.einsum('kd,kd->kd', n_K, self.text_features[:int(self.n_classes / self.n_tasks)*(self.task_count + 1),:].repeat_interleave(self.lp_size, dim=0).to(torch.float32).cuda()) # 对所有见过的文本标签分别和掩码k做逐元素乘积
                aq_k = torch.einsum('bkd,kd->bk', q, mask_text_features)
                
                if train:
                    if self.task_count > 0:
                        sim_loss = torch.mean(aq_k[:,:s])-torch.mean(aq_k[:,s:f])
                    else:
                        sim_loss = -torch.mean(aq_k[:,s:f])
                else:
                    sim_loss = 0

                if self.select_num > int(aq_k.shape[1]):
                    select_num = int(aq_k.shape[1])
                else:
                    select_num = self.select_num


                mask2zero = aq_k.topk(select_num, dim=1, largest=True, sorted=True)[0][:,-1] 
                aq_k [aq_k < mask2zero[:,None]] = 0
                P_ = torch.einsum('bk,kld->bld', aq_k , p)





                    
            # select prompts
            if e_valid:  
                if g_valid:
                    i_l = int(self.lp_length/2)
                    i_t = int(self.tp_length/2)
                    Ek = torch.cat((P_[:,:i_l,:], P_t[:,:i_t,:]), dim = 1)
                    Ev = torch.cat((P_[:,i_l:,:], P_t[:,i_t:,:]), dim = 1)
                else:
                    i = int(self.lp_length/2)
                    Ek = P_[:,:i,:]
                    Ev = P_[:,i:,:]
                p_return = [Ek, Ev]
                
            else:
                if g_valid:
                    i_t = int(self.tp_length/2)
                    Ek = P_t[:,:i_t,:]
                    Ev = P_t[:,i_t:,:]
                    p_return = [Ek, Ev]
                else:
                    loss = 0
                    p_return = None
                    # print("------------------------error------------------------------")
            
            # p_return = [Ek, Ev]



            # ortho penalty
            if train and e_valid and self.ortho_mu > 0:
                loss = ortho_penalty(A, self.adap_angle) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1), self.adap_angle ) * self.ortho_mu
            else:
                loss = sim_loss
        
        
        else: 
            loss = 0
            p_return = None


        return p_return, loss, x_block





    
    def cal_logits_cosine(self, x, temperature, test_feature=None):

        # 归一化
        image_features = x / x.norm(dim=-1, keepdim=True)

        if test_feature is not None:
            logits = (image_features.half() @ test_feature.T.half().cuda()) / temperature
        else:
            logits = (image_features.half() @ self.text_features.T.half().cuda()) / temperature # 计算余弦相似度

        return logits
    


def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p    

class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None, clip_encoder=True, cfg=None):
        super(ViTZoo, self).__init__()

        # get last layer
        # self.last = nn.Linear(512, num_classes)
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.clip_label_dict = cfg['clip_label_dict']
        self.semantic_label_dict = cfg['semantic_label_dict']
        self.temperature = cfg['temperature']
        self.args = cfg['args']
        self.use_clip_encoder = clip_encoder
        self.clip_type = self.args.clip_type

        # get feature encoder
        if clip_encoder:
            zoo_model, self.clip_preproess = clip.load(self.args.clip_type, "cuda") # clip encoder ViT-B/32、ViT-L/14
            if self.args.clip_type == "ViT-B/32":  # 512->768  12层
                self.clip_key_d = 512
                self.clip_emb_d = 768 
            elif self.args.clip_type == "ViT-B/16":  # 512->768  12层
                self.clip_key_d = 512
                self.clip_emb_d = 768 
            elif self.args.clip_type == "ViT-L/14":  # 768->1024  24层
                self.clip_key_d = 768
                self.clip_emb_d = 1024 
            else:
                self.clip_key_d = 768
                self.clip_emb_d = 768
            
        else:
            self.clip_key_d = 768
            if pt:
                zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12,
                                            num_heads=12, ckpt_layer=0,
                                            drop_path_rate=0
                                            )
                from timm.models import vit_base_patch16_224
                load_dict = vit_base_patch16_224(pretrained=True).state_dict()
                del load_dict['head.weight']; del load_dict['head.bias']
                zoo_model.load_state_dict(load_dict)

        # feature encoder changes if transformer vs resnet
        self.feat = zoo_model 
        

        # classifier
        if not self.args.use_label_encoder:
            self.last = nn.Linear(self.clip_key_d, num_classes)
        else:
            self.last = nn.Linear(self.clip_key_d, num_classes)
        
        

        # create prompting module
        if clip_encoder:
            with torch.no_grad():
                text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.clip_label_dict]).cuda()
                # dataparallel
                text_features = self.feat.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                # image_features  = image_features / image_features.norm(dim=-1, keepdim=True)

                if self.semantic_label_dict is not None:
                    semantic_text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.semantic_label_dict]).cuda()
                    # dataparallel
                    self.semantic_text_features = self.feat.encode_text(semantic_text_inputs)
                    self.semantic_text_features /= self.semantic_text_features.norm(dim=-1, keepdim=True)
                else:
                    self.semantic_text_features = None
            
            if self.prompt_flag == 'l2p':
                self.prompt = L2P(self.clip_emb_d, prompt_param[0], prompt_param[1], key_dim=self.clip_key_d, text_features=text_features, args = self.args)
            elif self.prompt_flag == 'dual':
                self.prompt = DualPrompt(self.clip_emb_d, prompt_param[0], prompt_param[1], key_dim=self.clip_key_d, text_features=text_features, args = self.args)
            elif self.prompt_flag == 'coda':
                self.prompt = CodaPrompt(self.clip_emb_d, prompt_param[0], prompt_param[1], key_dim=self.clip_key_d, text_features=text_features, args = self.args)
            elif self.prompt_flag == 'msp':
                self.prompt = MSP(self.clip_emb_d, prompt_param[0], prompt_param[1], key_dim=self.clip_key_d, text_features=text_features, args = self.args)
            else:
                self.prompt = None

        else:
            if self.prompt_flag == 'l2p':
                self.prompt = L2P(768, prompt_param[0], prompt_param[1])
            elif self.prompt_flag == 'dual':
                self.prompt = DualPrompt(768, prompt_param[0], prompt_param[1])
            elif self.prompt_flag == 'coda':
                self.prompt = CodaPrompt(768, prompt_param[0], prompt_param[1], args = self.args)
            else:
                self.prompt = None
        

        
        
    # pen: get penultimate features    
    def forward(self, x, pen=False, train=False, text_encoder=None, semantic_test = False): # 这里正向传播 上一级在prompt.py
        

        if self.prompt is not None:
            if self.use_clip_encoder:
                self.feat.eval()
                with torch.no_grad():
                    q, _ = self.feat.encode_image(x) 
                
                if self.args.semantic_exp_pure_clip: 
                    out = q.to(torch.float32)
                else:
                    out, prompt_loss = self.feat.encode_image(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id) # 这里feat是vit.py的visionTransformer，将q(x)进行匹配寻找prompt并嵌入中间向量得到最后的输出
                    out = out.to(torch.float32)


            else:
                
                with torch.no_grad():
                    q, _ = self.feat(x) 
                    q = q[:,0,:] 
                
                # if not self.args.config == 'configs/MNIST10_prompt.yaml':  # 可去
                #     out, prompt_loss = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
                #     out = out[:,0,:]
                # else:
                out = q
                prompt_loss = torch.tensor(0).cuda()
            # out = q.to(torch.float32)
            # prompt_loss = torch.zeros((1,), requires_grad=True).cuda()
        else:
            out, _ = self.feat(x)
            out = out[:,0,:]

        out = out.view(out.size(0), -1)


        if semantic_test:
            if self.use_clip_encoder: 
                out = self.prompt.cal_logits_cosine(out, self.temperature, test_feature=self.semantic_text_features)
            else:
                out = self.last(out)
        else:
            if self.args.use_label_encoder: 
                out = self.prompt.cal_logits_cosine(out, self.temperature) 
            else:
                out = self.last(out)




        if self.prompt is not None and train:
            return out, prompt_loss
        else:
            return out
            

            if self.prompt is not None and train:
                return out, prompt_loss
            else:
                return out

            
def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None, clip_encoder=True, cfg=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, clip_encoder=clip_encoder, cfg=cfg)

