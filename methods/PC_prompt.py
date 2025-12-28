import argparse
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math



import torch
from torch import nn
from torchvision import models
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models import create_model
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
# class ResModel(nn.Module):
#     def __init__(self):
#         super(ResModel, self).__init__()

#         self.premodel = models.resnet18(pretrained=True)
#         self.model = nn.Sequential(*list(self.premodel.children())[:-1])          ###[:-2]
#         out_chann = 512   ###  2048x4x4
#         print ("resnet18 model loaded")
#         self.compress = nn.Linear(out_chann, 768, bias=True)   ## False


#     def forward(self, x_input):
#         x = self.model(x_input)       #####  25 512 7 7
#         # print ('000:',x.shape)

#         x_comp = self.compress(x)      ###  25  2048

#         return x_comp

# class vitembeding(nn.Module):
#     def __init__(self):
#         super(vitembeding, self).__init__()

#         self.original_model = create_model(
#             'vit_base_patch16_224',
#             pretrained=True,
#             num_classes=1000,
#             drop_rate=0.0,
#             drop_path_rate=0.0,
#             drop_block_rate=None,
#         )


#     def forward(self, x_input):

#         for p in self.original_model.parameters():
#             p.requires_grad = False

#         x = self.original_model.forward_features(x_input)     
#         # print('x shape',x.shape)
#         x = x[:,0,:]
#         return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class PromptGenerator(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),   ###  1,64,1024x3
            nn.Linear(patch_dim, dim),        ###  1,64,1024
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))     ###  1,65,1024
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # self.input_embedding = vitembeding()  ### ResModel()

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        ##############prompt
        self.num_p = 16
        
        tl_vectors = torch.empty(
            1,
            1,
            # dtype=self.dtype,
            # device=self.device,
        )                                ####   [256,768]
        torch.nn.init.normal_(tl_vectors, std=0.02)
        self.tl_vectors = torch.nn.Parameter(tl_vectors)
        
        # self.tl_vectors = nn.Parameter(torch.randn((256, 768,)))
        # nn.init.uniform_(self.tl_vectors, -1, 1)
        
        self.acti_softmax =  nn.Softmax(dim=-1)
        self.acti_Sig = nn.Sigmoid()
               
#         self.input_frozen = FrozenVIT()
        
        self.pre_out = nn.Linear(768, self.num_p*2*256)
        
    def forward(self, x, maben=None):
        # print('img shape',img.shape)
        # x = self.to_patch_embedding(img)      ### 1,64 ,1024
        # with torch.no_grad():
        #     x = self.input_embedding(img)       ### b 1 768
        #     img_embedding = x.unsqueeze(1)
        # x = x.unsqueeze(1)
        b,_,embed_dim = x.shape
        if maben ==None:
            prompt_tokens = repeat(self.tl_vectors, 'n d -> b n d', b = b)
        else:
            prompt_tokens = repeat(maben, 'n d -> b n d', b = b)
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)     ## 1,1,1024
        
        # x = torch.cat((cls_tokens, x), dim=1)         
        x = torch.cat((x, prompt_tokens), dim=1)          ###### b,257,7768
        # print(x.shape)
        # x += self.pos_embedding[:, :(n + 1)]      ##  1,65,1024

        # x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        
        corr_w = self.pre_out(x)
        
        split_logits = corr_w.reshape(
            b,
            self.num_p*2,   #### 32
           256          ###   256
        )
        mixture_coeffs = self.acti_softmax(
            split_logits
        )
        if maben ==None:
            pgn_prompts = torch.einsum(
            'bom,mv->bov',
            [mixture_coeffs, self.tl_vectors]
        ) 
        else:
            pgn_prompts = torch.einsum(
            'bom,mv->bov',
            [mixture_coeffs, maben]
        )
        
        return pgn_prompts#, img_embedding
        # return self.mlp_head(x)


class WeightGenerator(nn.Module):
    def __init__(self, *,  num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.acti_softmax =  nn.Softmax(dim=-1)
        self.acti_Sig = nn.Sigmoid()
                
        # self.pre_out = nn.Linear(768, self.num_classes)
        
        self.head_size = 768
        self.key_w  = nn.Linear(768, self.head_size)
        self.query_w = nn.Linear(768, self.head_size)
        
    def forward(self, input_embed ,prompt_embed):
        
        x_embed = input_embed # .unsqueeze(1)
        b,_,wh = x_embed.shape
        
        key = self.key_w(x_embed)
#         key = x_embed
        # print(key.size())
        
        query = self.query_w(prompt_embed)
#         query = prompt_embed
        # print(query.size())
        
        Q = query
        K = torch.transpose(key, 1, 2)
        V = prompt_embed
        
        scores = torch.matmul(Q, K)
        scores = (scores / math.sqrt(self.head_size)).squeeze()
        probs = torch.sigmoid(scores)
        
#         ttt_max, ttindex_max = torch.max(probs, dim=-1)
#         ttt_max_B = repeat(ttt_max, 'b -> b n', n =25) 
#         probs_ = probs/ttt_max_B

#         probs = repeat(probs, 'b n -> b n d', d = 768)
        probs = repeat(probs, '...   -> ... n',  n = wh)
        prompt_embedding = torch.mul(V, probs)

        return prompt_embedding


#     return parser
def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim)
#     init_linear_layer(linear, std=std)
    return linear

class MID_EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        # self.lay_pgn_config = {
        #     "non_linearity": "gelu_new",
        #     "input_dim": 768,
        #     "task_embedding_dim": 768,
        #     "device": 0,
        #     "unique_hyper_net_layer_norm": True,
        #     "reduction_factor": 32
        # }
        
#         parser = argparse.ArgumentParser('DualPrompt configs', parents=[get_args_parser()])
#         args = parser.parse_args()
#         self.lay_pgn_module = MetaLayersAdapterController( args   )
        
        self.pgn_path = None      ### prompt generate network
        self.wgn_path = None      ### weight generate network
        self.pgn_settings = {
            'image_size' : 224,
            'patch_size' : 16,
            'num_classes': 25,
            'dim' : 768,   ####1024
            'depth' : 3,    ###5
            'heads' : 12,        ###12
            'mlp_dim' : 768,   ### 768*4
            'dropout' : 0.0,
            'emb_dropout' : 0.0
        }
#         self.wgn_settings = {
#             'image_size' : 224,
#             'patch_size' : 16,
#             'num_classes': 25,
#             'dim' : 768,   ####1024
#             'depth': 4,
#             'heads' : 12,
#             'mlp_dim' : 768*4,
#             'dropout' : 0.0,
#             'emb_dropout' : 0.0
#         }
        self.wgn_settings = {
            'num_classes': 25
        }
        self.pgn_module = PromptGenerator(    **self.pgn_settings    ).cuda()
        self.pgn_module_1 = PromptGenerator(    **self.pgn_settings    ).cuda()
        # self.pgn_module_2 = PromptGenerator(    **self.pgn_settings    ).cuda()
        # self.pgn_module_3 = PromptGenerator_3(    **self.pgn_settings    ).cuda()
        self.wgn_module = WeightGenerator(   **self.wgn_settings    ).cuda()
        
#         if self.pgn_path:
#             self.load_pgn_module(self.pgn_path,self.pgn_settings)
#         else:
#             self.build_pgn_module(self.pgn_settings) 
            
#         if self.wgn_path:
#             self.load_wgn_module(self.wgn_path, self.wgn_settings)
#         else:
#             self.build_wgn_module(self.wgn_settings)
        
#         if self.prompt_pool:
#             # user prefix style
#             if self.use_prefix_tune_for_e_prompt:
#                 assert embed_dim % self.num_heads == 0
#                 if self.same_key_value:
#                     prompt_pool_shape = (self.num_layers, 1, self.pool_size, self.length, 
#                                         self.num_heads, embed_dim // self.num_heads)

#                     if prompt_init == 'zero':
#                         self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
#                     elif prompt_init == 'uniform':
#                         self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
#                         nn.init.uniform_(self.prompt, -1, 1)
#                     self.prompt = self.prompt.repeat(1, 2, 1, 1, 1, 1)
#                 else:
#                     #######################################################
#                     prompt_pool_shape = (self.num_layers, 2, self.pool_size, self.length, 
#                                         self.num_heads, embed_dim // self.num_heads)
# #                     prompt_pool_shape = (pool_size, embed_dim)
#                     if prompt_init == 'zero':
#                         self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
#                     elif prompt_init == 'uniform':
#                         self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
#                         nn.init.uniform_(self.prompt, -1, 1)
#             else:
#                 prompt_pool_shape=(self.num_layers, self.pool_size, self.length, embed_dim)
#                 if prompt_init == 'zero':
#                     self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
#                 elif prompt_init == 'uniform':
#                     self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
#                     nn.init.uniform_(self.prompt, -1, 1)
                    
        # if using learnable prompt keys
#         eval_t = locals()
        if prompt_key:
      
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
#                 self.task_key_norm = torch.zeros(key_shape)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean 
            
        self.maben = torch.empty(
            256,                        #### 
            768,
            # dtype=self.dtype,
#             device=self.device,
        ).cuda()                                ####   [256,768]
        torch.nn.init.normal_(self.maben, std=0.02)
        self.maben = torch.nn.Parameter(self.maben) 
#         self.tl_vectors = torch.nn.Parameter(self.maben) 
        
#         self.pgn_module_Gfc_1 = nn.Sequential(
#             linear_layer(768, 128),
#             nn.ReLU(),
#             linear_layer(128,768)) 
        
#         self.pgn_module_Gfc_2 = nn.Sequential(
#             linear_layer(768, 128),
#             nn.ReLU(),
#             linear_layer(128,768)) 
        
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    def load_pgn_module(self, pgn_path,pgn_settings):
        pgn_module  = PromptGenerator(     **pgn_settings    )

        pgn_module.load_state_dict(
            state_dict=torch.load(pgn_path)
        )
        self.pgn_module = pgn_module

    def build_pgn_module(self,pgn_settings):
        self.pgn_module = PromptGenerator(    **pgn_settings    ).cuda()

        
    def load_wgn_module(self, wgn_path,wgn_settings):
        wgn_module  = WeightGenerator(    **wgn_settings    )

        wgn_module.load_state_dict(
            state_dict=torch.load(wgn_path)
        )
        self.wgn_module = wgn_module

    def build_wgn_module(self,wgn_settings):
        self.wgn_module = WeightGenerator(   **wgn_settings    ).cuda()
    def forward(self, x_embed,task_id=-1,task_key_norm= None, prompt_mask=None, cls_features=None,trainable=False):
        out = dict()
        batch_size = x_embed.shape[0]
       
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")
                
#             task_key_norm =task_key_norm.cuda()
#             if  trainable:
#                 self.prompt_key_ = self.prompt_key
#                 prompt_key_norm = self.l2_normalize(self.prompt_key_, dim=-1) # Pool_size, C
# #                 print('prompt_key type', self.prompt_key.dtype)
                
#             else:
# #                 prompt_key_norm = torch.tensor(task_key_norm, dtype = torch.float32)
#                 prompt_key_norm = task_key_norm
#                 print('task_key_norm type', task_key_norm.dtype)
                
            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1) # Pool_size, C
        
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1) # B, C
            
            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
            similarity = similarity.t() # B, pool_size
            
            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
            out['similarity'] = similarity
            out['idx_pred'] = idx
            
#             if self.batchwise_prompt and trainable:
#                 prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
#                 # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
#                 # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
#                 # Unless dimension is specified, this will be flattend if it is not already 1D.
#                 if prompt_id.shape[0] < self.pool_size:
#                     prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
#                     id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
#                 _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
#                 major_prompt_id = prompt_id[major_idx] # top_k
#                 # expand to batch
#                 idx = major_prompt_id.expand(x_embed.shape[0], -1).contiguous() # B, top_k
                
#                 out['idx_pred'] = idx

                # print('idx[0]',idx[0])
            # if prompt_mask is not None and trainable:
            if  trainable:
                idx = prompt_mask # B, top_k
                #print('when testing, this should not be used')    ## no
                
#             idx = prompt_mask
            out['idx_gt'] = prompt_mask
            out['prompt_idx'] = idx
            
            #############
            
            pgn_prompts_  = self.pgn_module(x_embed, self.maben)
#             out['prompt_l0'] = pgn_prompts_
#             batched_prompt = rearrange(pgn_prompts, 'b (l n e) (h d) -> b l n e h d',l=3,n=2, h = self.num_heads).permute(1, 0, 2,3,4,5)
            # print('shape:',pgn_prompts.shape)
            pgn_prompts_ = self.wgn_module(cls_features, pgn_prompts_)                            #### wgn
    #         pgn_prompts = self.wgn_module(x_embed, pgn_prompts)
            pgn_prompts=rearrange(pgn_prompts_.unsqueeze(0), 'l b (n e) d -> l b n e d', n = 2)
            # print(pgn_prompts.shape)


            pgn_prompts_1  = self.pgn_module_1(x_embed, self.maben)
#             out['prompt_l1'] = pgn_prompts_1
            pgn_prompts_1 = self.wgn_module(cls_features, pgn_prompts_1)                            #### wgn
#             pgn_prompts_1 = self.pgn_module_Gfc_1(pgn_prompts_)
            pgn_prompts_1=rearrange(pgn_prompts_1.unsqueeze(0), 'l b (n e) d -> l b n e d', n = 2)
            
            # pgn_prompts_2  = self.pgn_module_2(x_embed, self.maben)
#             out['prompt_l2'] = pgn_prompts_2
            # pgn_prompts_2 = self.wgn_module(cls_features, pgn_prompts_2)                           #### wgn
#             pgn_prompts_2 = self.pgn_module_Gfc_2(pgn_prompts_)
            # pgn_prompts_2=rearrange(pgn_prompts_2.unsqueeze(0), 'l b (n e) d -> l b n e d', n = 2)
            
            # pgn_prompts_3  = self.pgn_module_3(cls_features, self.maben)
            # pgn_prompts_3 = self.wgn_module(cls_features, pgn_prompts_3)
#             #pgn_prompts_2 = self.pgn_module_Gfc_2(pgn_prompts_)
            # pgn_prompts_3 = rearrange(pgn_prompts_3.unsqueeze(0), 'l b (n e) (h d) -> l b n e h d', n = 2, h = self.num_heads)
        
            # batched_prompt  = torch.cat([pgn_prompts, pgn_prompts_1,pgn_prompts_2,pgn_prompts_3], dim=0)
            batched_prompt  = torch.cat([pgn_prompts, pgn_prompts_1], dim=0)
          
            # if self.use_prefix_tune_for_e_prompt:
            #     batched_prompt_raw = self.prompt[:,:,idx]  # num_layers,2, (B, top_k,) length, C
            #     num_layers, dual, batch_size, top_k, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
                
            #     batched_prompt = batched_prompt_raw.reshape(
            #         num_layers, batch_size, dual, top_k * length, num_heads, heads_embed_dim
            #     )
            # else:
            #     batched_prompt_raw = self.prompt[:,idx]
            #     num_layers, batch_size, top_k, length, embed_dim = batched_prompt_raw.shape
            #     batched_prompt = batched_prompt_raw.reshape(
            #         num_layers, batch_size, top_k * length, embed_dim
            #     )

            batched_key_norm = prompt_key_norm[idx] # B, top_k, C

            out['selected_key'] = batched_key_norm
            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm
#             if task_id>0:
#                 task_key_norm = prompt_key_norm[:task_id+1]
#                 base_labels = torch.zeros(1,(task_id+1)).cuda()#.scatter(1, idx[0], 1) ##.expand(batch_size, -1)        ####[B, poolsize]
#                 base_labels = base_labels.index_fill(1, idx[0], 1)
#                 q_labels = torch.ones(batch_size, 1).cuda() 

#                 s = (q_labels @ base_labels > 0).float()     ####### [B, poolsize]
#                 # print('s shape:',s.shape)
#                 inner_product = x_embed_norm @ task_key_norm.t()      #####[B,poolsize]

#                 likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product
#                 # likelihood_loss = (1 -s)* inner_product - s * inner_product #+ inner_product.clamp(min=0)    ### 83.75
#                 lossDP = likelihood_loss.mean()                ### 0.6889
#             else:
#                 lossDP = 0.0
                
            # Put pull_constraint loss calculation inside
            lossDP =0.0
            weight_former = 1.0
            
#             if task_id> 0 and task_key_norm!= None:
# #                 if  trainable:
                    
# #                     x_xils = torch.ones(768)*int(task_id)
# #                     x_xils = x_xils.long().cuda()
# #                     y_xils = torch.arange(0,768).long().cuda()

# #                     new_value = prompt_key_norm[int(task_id)]
# #                     new_value = new_value.detach()

# #                     index = (
# #                             x_xils, 
# #                             y_xils,
# #                         )
# #                     task_key_norm.index_put_(index, new_value.cuda())
                
# #                     if task_id>0:

#                 former_idx = torch.arange(0,task_id).cuda()
#                 former_idx_batch =  former_idx.expand(x_embed.shape[0], -1).contiguous()
#                 former_key_norm = prompt_key_norm[former_idx_batch]

#                 x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
#                 sim = former_key_norm * x_embed_norm # B, task_id, C
#                 lossDP = torch.sum(sim) / x_embed.shape[0] # Scalar
#                 weight_former = 1.0/task_id

            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
            
            out['reduce_sim'] =  reduce_sim# - weight_former*lossDP
#             out['reduce_sim'] = - lossDP
        # else:
        #     # user prefix style
        #     if self.use_prefix_tune_for_e_prompt:
        #         assert embed_dim % self.num_heads == 0
        #         if self.same_key_value:
        #             prompt_pool_shape = (self.num_layers, 1, self.length, 
        #                                 self.num_heads, embed_dim // self.num_heads)
        #             if self.prompt_init == 'zero':
        #                 self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        #             elif self.prompt_init == 'uniform':
        #                 self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
        #                 nn.init.uniform_(self.prompt, -1, 1)
        #             self.prompt = self.prompt.repeat(1, 2, 1, 1, 1)
        #         else:
        #             prompt_pool_shape = (self.num_layers, 2, self.length, 
        #                                 self.num_heads, embed_dim // self.num_heads)
        #             if self.prompt_init == 'zero':
        #                 self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        #             elif self.prompt_init == 'uniform':
        #                 self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, length, num_heads, embed_dim // num_heads
        #                 nn.init.uniform_(self.prompt, -1, 1)
        #         batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1, -1)
        #     else:
        #         prompt_pool_shape = (self.num_layers, self.length, embed_dim)
        #         if self.prompt_init == 'zero':
        #             self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        #         elif self.prompt_init == 'uniform':
        #             self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
        #             nn.init.uniform_(self.prompt, -1, 1)
        #         batched_prompt = self.prompt.unsqueeze(0).expand(-1, x_embed.shape[0], -1, -1)
        
        out['batched_prompt'] = batched_prompt

        return out



class M2Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value

        self.pooling_layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        self.pgn_path = None      ### prompt generate network
        self.wgn_path = None      ### weight generate network
        self.pgn_settings = {
            'image_size' : 224,
            'patch_size' : 16,
            'num_classes': 25,
            'dim' : 768,   ####1024
            'depth' : 3,    ###5
            'heads' : 12,        ###12
            'mlp_dim' : 768,   ### 768*4
            'dropout' : 0.0,
            'emb_dropout' : 0.0
        }

        self.wgn_settings = {
            'num_classes': 25
        }
        self.pgn_module = PromptGenerator(    **self.pgn_settings    ).cuda()
        self.pgn_module_1 = PromptGenerator(    **self.pgn_settings    ).cuda()
        self.wgn_module = WeightGenerator(   **self.wgn_settings    ).cuda()
        
        if prompt_key:
      
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=[0, 2])
            self.prompt_key = prompt_mean 
            
        self.maben = torch.empty(
            256,                        #### 
            768,
            # dtype=self.dtype,
#             device=self.device,
        ).cuda()                                ####   [256,768]
        torch.nn.init.normal_(self.maben, std=0.02)
        self.maben = torch.nn.Parameter(self.maben) 
        
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    def load_pgn_module(self, pgn_path,pgn_settings):
        pgn_module  = PromptGenerator(     **pgn_settings    )

        pgn_module.load_state_dict(
            state_dict=torch.load(pgn_path)
        )
        self.pgn_module = pgn_module

    def build_pgn_module(self,pgn_settings):
        self.pgn_module = PromptGenerator(    **pgn_settings    ).cuda()

        
    def load_wgn_module(self, wgn_path,wgn_settings):
        wgn_module  = WeightGenerator(    **wgn_settings    )

        wgn_module.load_state_dict(
            state_dict=torch.load(wgn_path)
        )
        self.wgn_module = wgn_module

    def build_wgn_module(self,wgn_settings):
        self.wgn_module = WeightGenerator(   **wgn_settings    ).cuda()
    def forward(self, x_embed, task_id=-1,task_key_norm= None, prompt_mask=None, cls_features=None,trainable=False):
        out = dict()
        batch_size = x_embed.shape[0]
       
        if x_embed.shape[1] == 4096:
            x_embed = rearrange(x_embed, 'b (h w) c -> b h w c', h = int(math.sqrt(x_embed.shape[1])))
            x_embed = x_embed.permute(0, 3, 1, 2)
            x_embed = self.pooling_layers(x_embed)
            # print(x_embed.shape)
            # x_embed = x_embed.permute(0, 2, 3, 1).contiguous().view(batch_size, 8 * 8, 96)
            x_embed = rearrange(x_embed.permute(0, 2, 3, 1).contiguous(), 'b h w c -> b (h w) c')
            x_embed = x_embed.view(batch_size, 32, 8, 96)
            x_embed = rearrange(x_embed, 'b h w c -> b h (w c)')
            # print(x_embed.shape)

        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")
                
                
            prompt_key_norm = self.l2_normalize(self.prompt_key, dim=-1) # Pool_size, C
        
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1) # B, C
            
            similarity = torch.matmul(prompt_key_norm, x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
            similarity = similarity.t() # B, pool_size

            (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
            out['similarity'] = similarity
            out['idx_pred'] = idx

            if  trainable:
                idx = prompt_mask # B, top_k
                #print('when testing, this should not be used')    ## no
                
#             idx = prompt_mask
            out['idx_gt'] = prompt_mask
            out['prompt_idx'] = idx
            
            #############
            
            pgn_prompts_  = self.pgn_module(x_embed, self.maben)
            pgn_prompts_ = self.wgn_module(cls_features, pgn_prompts_)                            #### wgn
            pgn_prompts = rearrange(pgn_prompts_.unsqueeze(0), 'l b (n e) d -> l b n e d', n = 2)

            pgn_prompts_1  = self.pgn_module_1(x_embed, self.maben)
            pgn_prompts_1 = self.wgn_module(cls_features, pgn_prompts_1)                            #### wgn
            pgn_prompts_1=rearrange(pgn_prompts_1.unsqueeze(0), 'l b (n e) d -> l b n e d', n = 2)
            
            batched_prompt  = torch.cat([pgn_prompts, pgn_prompts_1], dim=0)

            batched_key_norm = prompt_key_norm[idx] # B, top_k, C

            out['selected_key'] = batched_key_norm
            out['prompt_key_norm'] = prompt_key_norm
            out['x_embed_norm'] = x_embed_norm

            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
            
            out['reduce_sim'] =  reduce_sim# - weight_former*lossDP
        out['batched_prompt'] = batched_prompt

        return out
