import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from einops import rearrange, repeat

from .models import *
from .nets.prompt_learner import *
from torch.nn import MultiheadAttention
from .nets.HTSAT import HTSAT_Swin_Transformer
from methods import esc_fig
import math
import numpy as np
import logging
from torch.cuda.amp import autocast
from einops import rearrange

### VGGSound
# from nets import Resnet_VGGSound

from .nets.helper import do_mixup, interpolate

# from nets.ast_models import ASTModel
from .nets.my_vit import VisionTransformer
from methods.prompt import VisualPrompt, AudioPrompt
from methods.PC_prompt import MID_EPrompt, M2Prompt
from methods.dualprompt import VisualEPrompt
from data import generate_category_list


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# class Midadapter(nn.Module):
#     def __init__(self,
#                  d_model=None,
#                  bottleneck=None,
#                  dropout=0.0,
#                  init_option="bert",
#                  adapter_scalar="1.0",
#                  adapter_layernorm_option="in"):
#         super().__init__()
#         self.n_embd = d_model
#         self.down_size = bottleneck

#         #_before
#         self.adapter_layernorm_option = adapter_layernorm_option

#         self.adapter_layer_norm_before = None
#         if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
#             self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

#         if adapter_scalar == "learnable_scalar":
#             self.scale = nn.Parameter(torch.ones(1))
#         else:
#             self.scale = float(adapter_scalar)

#         self.down_proj = nn.Linear(self.n_embd, self.down_size)
#         self.non_linear_func = nn.ReLU()
#         self.up_proj = nn.Linear(self.down_size, self.n_embd)

#         self.dropout = dropout
#         if init_option == "bert":
#             raise NotImplementedError
#         elif init_option == "lora":
#             with torch.no_grad():
#                 nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
#                 nn.init.zeros_(self.up_proj.weight)
#                 nn.init.zeros_(self.down_proj.bias)
#                 nn.init.zeros_(self.up_proj.bias)

#     def forward(self, x, add_residual=True, residual=None):
#         residual = x if residual is None else residual
#         if self.adapter_layernorm_option == 'in':
#             x = self.adapter_layer_norm_before(x)

#         down = self.down_proj(x)
#         down = self.non_linear_func(down)
#         down = nn.functional.dropout(down, p=self.dropout, training=self.training)
#         up = self.up_proj(down)

#         up = up * self.scale

#         if self.adapter_layernorm_option == 'out':
#             up = self.adapter_layer_norm_before(up)

#         if add_residual:
#             output = up + residual
#         else:
#             output = up

#         return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_length, d_model]
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch_size, seq_length, d_model]
        return x + self.pe[:, :x.size(1), :]




class RNNEncoder(nn.Module):
    def __init__(self, audio_dim, video_dim, d_model, num_layers):
        super(RNNEncoder, self).__init__()

        self.d_model = d_model
        # self.audio_rnn = nn.LSTM(audio_dim, int(d_model / 2), num_layers=num_layers, batch_first=True,
        #                          bidirectional=True, dropout=0.2)
        # self.visual_rnn = nn.LSTM(video_dim, d_model, num_layers=num_layers, batch_first=True, bidirectional=True,
        #                           dropout=0.2)
        
        # ------------------------ Transformer -------------------
        self.audio_pos_encoding = PositionalEncoding(audio_dim)
        self.visual_pos_encoding = PositionalEncoding(video_dim)

        self.audio_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=audio_dim, 
                nhead=8,
                dim_feedforward=int(d_model * 2),
                dropout=0.2,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.audio_proj = nn.Linear(audio_dim, d_model)

        self.visual_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=video_dim, 
                nhead=8,
                dim_feedforward=d_model * 2,
                dropout=0.2,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.visual_proj = nn.Linear(video_dim, d_model)


    def forward(self, audio_feature, visual_feature):
        # ------------------------ Transformer -------------------
        audio_input = self.audio_pos_encoding(audio_input)
        audio_output = self.audio_transformer(audio_input)
        audio_output = self.audio_proj(audio_output)

        video_input = self.visual_pos_encoding(visual_feature)
        video_output = self.visual_transformer(video_input) 
        video_output = self.visual_proj(video_output)


        # audio_output, _ = self.audio_rnn(audio_feature)
        # video_output, _ = self.visual_rnn(visual_feature)
        return audio_output, video_output


class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature


class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feature, memory_feature):
        query_feature = self.affine_matrix(query_feature)
        output = self.decoder(query_feature, memory_feature)

        return output


class CAS_Module(nn.Module):
    def __init__(self, d_model, num_class=28):
        super(CAS_Module, self).__init__()
        self.d_model = d_model
        self.num_class = num_class
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=self.num_class + 1, kernel_size=1, stride=1, padding=0,
                      bias=False)
        )

    def forward(self, content):
        content = content.permute(0, 2, 1)

        out = self.classifier(content)
        out = out.permute(0, 2, 1)
        return out


class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        # self.affine_concat = nn.Linear(2*256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(d_model, 1)  # start and end
        self.event_classifier = nn.Linear(d_model, 28)
        # self.cas_model = CAS_Module(d_model=d_model, num_class=28)

    # self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        max_fused_content, _ = fused_content.transpose(1, 0).max(1)
        logits = self.classifier(fused_content)
        # scores = self.softmax(logits)
        class_logits = self.event_classifier(max_fused_content)
        # class_logits = self.event_classifier(fused_content.transpose(1,0))
        # sorted_scores_base,_ = class_logits.sort(descending=True, dim=1)
        # topk_scores_base = sorted_scores_base[:, :4, :]
        # class_logits = torch.mean(topk_scores_base, dim=1)
        class_scores = class_logits

        return logits, class_scores


class WeaklyLocalizationModule(nn.Module):
    def __init__(self, input_dim):
        super(WeaklyLocalizationModule, self).__init__()

        self.hidden_dim = input_dim  # need to equal d_model
        self.classifier = nn.Linear(self.hidden_dim, 1)  # start and end
        self.event_classifier = nn.Linear(self.hidden_dim, 29)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        fused_content = fused_content.transpose(0, 1)
        max_fused_content, _ = fused_content.max(1)
        # confident scores
        is_event_scores = self.classifier(fused_content)
        # classification scores
        raw_logits = self.event_classifier(max_fused_content)[:, None, :]
        # fused
        fused_logits = is_event_scores.sigmoid() * raw_logits
        # Training: max pooling for adapting labels
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        return is_event_scores.squeeze(), raw_logits.squeeze(), event_scores


class AudioVideoInter(nn.Module):
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(AudioVideoInter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.video_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, video_feat, audio_feat):
        # video_feat, audio_feat: [10, batch, 256]
        global_feat = video_feat * audio_feat
        memory = torch.cat([audio_feat, video_feat], dim=0)
        mid_out = self.video_multihead(global_feat, memory, memory)[0]
        output = self.norm1(global_feat + self.dropout(mid_out))

        return output


class CMBS(nn.Module):
    def __init__(self, config):
        super(CMBS, self).__init__()
        self.config = config
        self.beta = 0.4
        self.spatial_channel_att = New_Audio_Guided_Attention(self.beta)
        self.video_input_dim = 512
        self.audio_input_dim = 128

        self.video_fc_dim = 512
        self.d_model = 256

        self.v_fc = nn.Linear(1536, self.video_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_input_dim, d_model=self.d_model,
                                                            feedforward_dim=1024)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_input_dim, d_model=self.d_model,
                                                         feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model,
                                                            feedforward_dim=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.d_model, d_model=self.d_model,
                                                         feedforward_dim=1024)
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=self.audio_input_dim, video_dim=self.video_input_dim,
                                                 d_model=self.d_model, num_layers=1)

        self.audio_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.video_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )

        self.AVInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.VAInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.localize_module = SupvLocalizeModule(self.d_model)
        self.video_norm = nn.LayerNorm(self.d_model)
        self.audio_norm = nn.LayerNorm(self.d_model)
        self.audio_cas = nn.Linear(self.d_model, 28)
        self.video_cas = nn.Linear(self.d_model, 28)
        self.alpha = 0.1
        self.gamma = 0.3

    def forward(self, visual_feature, audio_feature):
        # [batch, 10, 512]

        # optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
        audio_rnn_input = audio_feature
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # spatial-channel attention
        visual_feature = self.spatial_channel_att(visual_feature, audio_feature)
        visual_rnn_input = visual_feature

        audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer(audio_rnn_input, visual_rnn_input)
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 256]
        visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 512]

        # audio query
        video_key_value_feature = self.video_encoder(visual_encoder_input1)
        audio_query_output = self.audio_decoder(audio_encoder_input1, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_encoder_input1)
        video_query_output = self.video_decoder(visual_encoder_input1, audio_key_value_feature)

        audio_gate = self.audio_gated(audio_key_value_feature)
        video_gate = self.video_gated(video_key_value_feature)

        audio_visual_gate = audio_gate * video_gate

        video_query_output = video_query_output + audio_gate * video_query_output * self.alpha
        audio_query_output = audio_query_output + video_gate * audio_query_output * self.alpha

        video_cas = self.video_cas(video_query_output)  # [10, 32, 28]
        audio_cas = self.audio_cas(audio_query_output)
        video_cas = video_cas.permute(1, 0, 2)
        audio_cas = audio_cas.permute(1, 0, 2)
        sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        topk_scores_video = sorted_scores_video[:, :4, :]
        score_video = torch.mean(topk_scores_video, dim=1)
        sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        topk_scores_audio = sorted_scores_audio[:, :4, :]
        score_audio = torch.mean(topk_scores_audio, dim=1)  # [32, 28]

        # event_visual_gate = score_video.sigmoid()
        # event_audio_gate = score_audio.sigmoid()

        av_score = (score_video + score_audio) / 2

        video_query_output = self.AVInter(video_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, video_query_output)

        is_event_scores, event_scores = self.localize_module((video_query_output + audio_query_output) / 2)
        event_scores = event_scores + self.gamma * av_score
        # event_scores = event_scores + self.gamma * (event_visual_gate * event_audio_gate) * event_scores

        return is_event_scores, event_scores, audio_visual_gate, av_score
    

    # Question
class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):
        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        self.lstm.flatten_parameters()
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature



class AvqaDownTaskHandler(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a1 = nn.Linear(768, 768)
        self.fc_a2 = nn.Linear(768, 768)

        self.fc_v1 = nn.Linear(768, 768)
        self.fc_v2 = nn.Linear(768, 768)

        self.fc_gl=nn.Linear(768+768, 768)
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(768+768, 512)
        # self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        # self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        # self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 2)
        # self.relu4 = nn.ReLU()

        self.attn_a = nn.MultiheadAttention(768, 4, dropout=0.1)
        self.attn_v = nn.MultiheadAttention(768, 4, dropout=0.1)
        self.fc_fusion = nn.Linear(768+768, 768)

        self.linear11 = nn.Linear(768, 768)
        self.dropout1 = nn.Dropout(0.1)
        self.linear12 = nn.Linear(768, 768)

        self.linear21 = nn.Linear(768, 768)
        self.dropout2 = nn.Dropout(0.1)
        self.linear22 = nn.Linear(768, 768)
        self.norm1 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)
        self.dropout3 = nn.Dropout(0.1)
        self.dropout4 = nn.Dropout(0.1)
        self.norm3 = nn.LayerNorm(768)
        self.fc_ans = nn.Linear(768, 42)

        self.question_encoder = QstEncoder(93, 768, 768, 1, 768)

    def forward(self, visual, audio, question):

        # visual: 60, 60, 768  B, T, C
        # audio: 60, 64, 768   B, T, C

        qst_feature = self.question_encoder(question)
        xq = qst_feature.unsqueeze(0)

        ## audio features  [2*B*T, 128] =============================================
        audio_feat = F.relu(self.fc_a1(audio))
        audio_feat = self.fc_a2(audio_feat)
        audio_feat = nn.functional.normalize(audio_feat, dim=-1)

        ## viusal features  [2*B*T, 128] =================================================
        vis_feat = F.relu(self.fc_v1(visual))
        vis_feat = self.fc_v2(vis_feat)
        vis_feat = nn.functional.normalize(vis_feat, dim=-1)
        
        x2_va = torch.matmul(vis_feat, audio_feat.transpose(-1,-2))
        # print(x2_va.shape)
        x2_p = F.softmax(x2_va, dim=-1)                                     # [B*T, 1, HxW]
        # print(x2_p.shape, vis_feat.shape)
        visual_feat_grd = torch.matmul(x2_p, audio_feat)  
        visual_feat_grd_after_grounding_posi = visual_feat_grd.squeeze()    # [B*T, C]   

        visual_gl = torch.cat((visual, visual_feat_grd_after_grounding_posi),dim=-1)
        visual_feat_grd = self.tanh(visual_gl)
        visual_feat_grd_posi = self.fc_gl(visual_feat_grd)              # [B*T, C]

        # print(audio_feat.shape, visual_feat_grd_posi.shape)
        # feat = torch.cat((audio_feat, visual_feat_grd_posi), dim=-1)    # [B*T, C*2], [B*T, 3072]

        # feat = F.relu(self.fc1(feat))       # (3072, 512)
        # feat = F.relu(self.fc2(feat))       # (512, 256)
        # feat = F.relu(self.fc3(feat))       # (256, 128)
        # out_match_posi = self.fc4(feat)     # (128, 2)
        out_match_posi = 0

        B = xq.shape[1]
        visual_feat_grd_be = visual_feat_grd_posi.view(B, -1, 768)   # [B, T, 512]
        visual_feat_grd = visual_feat_grd_be.permute(1,0,2)
        
        ## attention, question as query on visual_feat_grd
        visual_feat_att = self.attn_v(xq, visual_feat_grd, visual_feat_grd, attn_mask=None, key_padding_mask=None)[0].squeeze(0)
        src = self.linear12(self.dropout1(F.relu(self.linear11(visual_feat_att))))
        visual_feat_att = visual_feat_att + self.dropout2(src)
        visual_feat_att = self.norm1(visual_feat_att)

        # attention, question as query on audio
        audio_feat_be=audio_feat.view(B, -1, 768)
        audio_feat = audio_feat_be.permute(1, 0, 2)
        audio_feat_att = self.attn_a(xq, audio_feat, audio_feat, attn_mask=None,key_padding_mask=None)[0].squeeze(0)
        src = self.linear22(self.dropout3(F.relu(self.linear21(audio_feat_att))))
        audio_feat_att = audio_feat_att + self.dropout4(src)
        audio_feat_att = self.norm2(audio_feat_att)
        

        feat = torch.cat((audio_feat_att+audio_feat_be.mean(dim=-2).squeeze(), visual_feat_att+visual_feat_grd_be.mean(dim=-2).squeeze()), dim=-1)
        feat = self.tanh(feat)
        feat = self.fc_fusion(feat)

        ## fusion with question
        combined_feature = torch.mul(feat, qst_feature)
        combined_feature = self.tanh(combined_feature)
        out_qa = self.fc_ans(combined_feature)              # [batch_size, ans_vocab_size]

        return out_qa, out_match_posi



# ------------------------------------- AVS --------------------------------------------------

class UpBlock(nn.Module):
    """可扩展的上采样模块（显式输出维度标记）"""
    def __init__(self, in_dim, out_dim, scale=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.out_dim = out_dim  # 关键属性声明

    def forward(self, x):
        return self.conv(self.upsample(x))

class VisualFeatureDecoder(nn.Module):
    """最终输出224x224的视觉解码器"""
    def __init__(self, 
                 input_dim=768, 
                 spatial_shape=None,
                 up_scales=[2,2,2,2,2],  # 5次上采样 (7 -> 224)
                 feat_dim=768):          # 新增特征维度参数
        super().__init__()
        
        # 空间重建验证
        self.spatial_shape = spatial_shape
        self.require_auto_shape = spatial_shape is None
        
        # 特征投影
        self.spatial_projection = nn.Conv2d(input_dim, input_dim//2, kernel_size=1)
        
        # 上采样模块
        self.up_blocks = nn.ModuleList()
        current_dim = input_dim//2  # 初始维度384
        for scale in up_scales:
            next_dim = max(current_dim//2, 64)
            self.up_blocks.append(
                UpBlock(current_dim, next_dim, scale)
            )
            current_dim = next_dim
        
        # 特征适配层 (保证输出维度768)
        self.feat_adapter = nn.Sequential(
            nn.Conv2d(current_dim, feat_dim, 3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU()
        )
        
        # 分类头 (保持原任务)
        # self.head = nn.Conv2d(feat_dim, 20, 1)  # 20 classes

    def _validate_input(self, seq_len):
        """验证输入是否符合空间展开要求"""
        num_patches = seq_len - 1  # 扣除CLS token
        if self.require_auto_shape:
            sqrt_patch = int(math.sqrt(num_patches))
            assert sqrt_patch ** 2 == num_patches, \
                f"Cannot auto infer spatial shape from {num_patches} patches (not a perfect square)"
            self.spatial_shape = (sqrt_patch, sqrt_patch)
        else:
            assert self.spatial_shape[0] * self.spatial_shape[1] == num_patches, \
                f"Given spatial_shape {self.spatial_shape} ({self.spatial_shape[0]*self.spatial_shape[1]} patches)" \
                f"doesn't match input {num_patches} patches"

    def forward(self, visual_seq):
        B, seq_len, C = visual_seq.shape
        self._validate_input(seq_len)  # 验证7x7输入
        
        # 特征重建
        cls_token = visual_seq[:, 0]
        patch_feats = visual_seq[:, 1:].view(B, *self.spatial_shape, C)
        spatial_feats = patch_feats.permute(0,3,1,2)  # BxCxHxW
        
        # 特征处理
        spatial_feats = self.spatial_projection(spatial_feats)
        for up_block in self.up_blocks:
            spatial_feats = up_block(spatial_feats)
        
        # 输出处理
        final_feats = self.feat_adapter(spatial_feats)  # [B,768,224,224]
        # masks = self.head(final_feats)                   # [B,20,224,224]
        return final_feats, cls_token  # 返回特征、分类结果、CLS


class AVSDownTaskHandler(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 视觉解码器 (输出768维特征)
        self.visual_decoder = VisualFeatureDecoder(feat_dim=512)
        
        # 音频处理
        self.audio_proj = nn.Linear(768, 512)
        
        # 跨模态融合
        self.cross_attn = CrossAttention(query_dim=768, context_dim=512)
        self.fusion_conv = nn.Conv2d(512, 1, 1)  # 最终输出通道为1

    def forward(self, visual_seq, audio_features):
        # 维度转换
        visual_seq = visual_seq.transpose(0, 1)  # [B,50,768]
        
        # 视觉解码
        vis_feats, cls_token = self.visual_decoder(visual_seq)
        
        # 音频增强
        audio_ctx = self.audio_proj(audio_features)  # [B,64,768]
        
        # 跨模态注意力
        fused_cls = self.cross_attn(
            query=cls_token.unsqueeze(1),  # [B,1,768]
            context=audio_ctx              # [B,64,768]
        )  # -> [B,1,768]
        
        # 空间融合
        B, C, H, W = vis_feats.shape
        spatial_vis = vis_feats.view(B, C, H*W).permute(0,2,1)  # [B,HW,768]
        fusion_weights = torch.matmul(
            spatial_vis,          # [B,224^2,768]
            fused_cls.transpose(1,2)  # [B,768,1]
        ).view(B, 1, H, W)  # [B,1,224,224]
        
        # 最终输出 (可选项：与分类结果融合)
        final_output = self.fusion_conv(vis_feats * fusion_weights)  # [B,1,224,224]


        return final_output, [], [] # [B,224,224]
    

# ----------------- 关键组件实现（增强鲁棒性）-----------------
class CrossAttention(nn.Module):
    """动态维度适应注意力"""
    def __init__(self, query_dim, context_dim, heads=8):
        super().__init__()
        self.heads = heads
        
        self.W_q = nn.Linear(query_dim, context_dim)
        self.W_kv = nn.Linear(context_dim, 2*context_dim)
        self.proj = nn.Linear(context_dim, context_dim)
        
        self.dim = context_dim // heads

    def forward(self, query, context):
        """处理不同长度的上下文"""
        B = query.size(0)
        
        # Query投影
        q = self.W_q(query).view(B, -1, self.heads, self.dim)
        
        # Key-Value对生成
        kv = self.W_kv(context)
        k, v = torch.chunk(kv, 2, dim=-1)
        k = k.view(B, -1, self.heads, self.dim)
        v = v.view(B, -1, self.heads, self.dim)
        
        # 注意力计算
        attn = torch.einsum('bqhd,bkhd->bqkh', q, k) / (self.dim ** 0.5)
        attn = torch.softmax(attn, dim=2)
        
        # 聚合Value
        out = torch.einsum('bqkh,bkhd->bqhd', attn, v)
        out = out.contiguous().view(B, -1, self.heads*self.dim)
        return self.proj(out)  # [B, seq, query_dim]
# ------------------------------------- AVS --------------------------------------------------


class AudioVisualContrastive(nn.Module):
    def __init__(self, logit_scale):
        super().__init__()
        self.fc_a1 = nn.Linear(512, 512)
        self.logit_scale = logit_scale.exp()

    def forward(self, video, audio):
        bs = audio.size(0) // 10
        audio = self.fc_a1(audio)
        video = video.view(bs, 10, -1)
        audio = audio.view(bs, 10, -1)
        video, audio = video.mean(dim=1), audio.mean(dim=1)
        video = video / video.norm(dim=-1, keepdim=True)
        audio = audio / audio.norm(dim=-1, keepdim=True)
        logits_audio_image = self.logit_scale * audio @ video.t()  # [B, B]
        logits_image_audio = self.logit_scale * video @ audio.t()  # [B, B]

        return logits_audio_image, logits_image_audio


class AudioAdapter(nn.Module):
    def __init__(self):
        super().__init__()

        self.alpha = 0.6
        self.d_model = 256
        self.audio_gated = nn.Sequential(
            nn.Linear(self.d_model, 1),
            nn.Sigmoid()
        )
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model,
                                                            feedforward_dim=1024)
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=128, video_dim=768, d_model=256, num_layers=1)

        #self.audio_encoder = VideoAudioAttentionAdapter()

    def forward(self, x, audio):
        bs = x.size(0) // 10
        x = x.view(bs, 10, -1)  # [B, 10, 768]
        x = x.permute(1, 0, 2)  # [10, B, 768]
        audio = audio.view(bs, 10, -1)  # [B, 10, 128]
        # audio query
        audio_rnn_output1 = self.audio_visual_rnn_layer.audio_rnn(audio)[0]
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [10, B, 256]
        audio_key_value_feature = self.audio_encoder(audio_encoder_input1)

        audio_gate = self.audio_gated(audio_key_value_feature)

        x = x + audio_gate * x * self.alpha

        x = x.permute(1, 0, 2)  # [B, 10, 768]
        x = x.view(bs * 10, -1)

        audio_key_value_feature = audio_key_value_feature.permute(1, 0, 2).contiguous()  # [B, 10, 256]
        audio_key_value_feature = audio_key_value_feature.view(bs * 10, -1)

        return x, audio_key_value_feature


class VisualAdapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, input_dim, output_dim, adapter_kind, task_name, dim_list=None, layer_idx=0, reduction_factor=16, opt=None ,use_bn=True, use_gate=True, num_tk=87, conv_dim_in=0, conv_dim_out=0, linear_in=0, linear_out=0):
        super().__init__()
        self.adapter_kind = adapter_kind
        self.use_bn = use_bn
        self.is_multimodal = opt["is_multimodal"]
        self.opt = opt
        self.fc_caption = nn.Linear(512, 192)
        self.num_tk = num_tk
        self.conv_adapter = nn.Conv2d(conv_dim_in, conv_dim_out, kernel_size=1)  
        self.fc = nn.Linear(linear_in, linear_out)
        d_model = linear_out // 2
        self.fc_affine_audio_1 = nn.Linear(linear_out, linear_out)
        self.fc_affine_video_1 = nn.Linear(linear_out, linear_out)
        self.fc_affine_bottleneck = nn.Linear(linear_out, d_model)
        self.fc_affine_video_2 = nn.Linear(linear_out, d_model)
        self.fc_affine_audio_2 = nn.Linear(linear_out, d_model)     
        self.fc_affine_v_s_att = nn.Linear(d_model, 1) 
        self.fc_tanh = nn.Tanh()
        self.fc_softmax = nn.Softmax(dim=-1)
        self.fc_affine_v_c_att = nn.Linear(d_model, linear_out)

        self.temporal_gated = nn.Sequential(
                        nn.Linear(linear_out, 1),
                        nn.Sigmoid()
                    )
        
        if task_name == "AVS":
            self.frame_num = 5
        else:
            self.frame_num = 10


        if use_gate:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = None

        if adapter_kind == "bottleneck" and self.is_multimodal:
            self.down_sample_size = input_dim // reduction_factor
            ### -----> attetnion 
            # self.cm1_att = nn.MultiheadAttention(embed_dim=self.down_sample_size, num_heads=1)

            # self.cm2_att = nn.MultiheadAttention(embed_dim=self.down_sample_size, num_heads=1)




            # self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, self.down_sample_size)))
            # self.my_tokens = nn.Parameter(torch.zeros((self.opt.num_tokens, input_dim)))


            self.my_tokens = nn.Parameter(torch.rand((num_tk, input_dim)))

            # self.ln_z = nn.LayerNorm(self.down_sample_size)
            # self.ln_tk = nn.LayerNorm(self.down_sample_size)

            # self.mapping = nn.Conv2d(input_dim, input_dim, 1, groups=self.opt.num_conv_group, bias=False)
            

            self.gate_tk = nn.Parameter(torch.ones(1))


            self.gate_av = nn.Parameter(torch.zeros(1))
    

            
            

            ### <------

            self.activation = nn.ReLU(inplace=True)
            # self.down_sampler = nn.Linear(input_dim, self.down_sample_size, bias=False)
            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt["num_conv_group"], bias=False)
            
            # self.down_sampler_vis = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt.num_conv_group, bias=False)

            # self.up_sampler = nn.Linear(self.down_sample_size, output_dim, bias=False)
            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt["num_conv_group"], bias=False)

            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)
            
            ### -------> yb: add
            if self.opt["is_before_layernorm"]:
                self.ln_before = nn.LayerNorm(output_dim)
            if self.opt["is_post_layernorm"]:
                self.ln_post = nn.LayerNorm(output_dim)
            ### <---------

        elif adapter_kind == "bottleneck":
            self.down_sample_size = input_dim // reduction_factor
            self.activation = nn.ReLU(inplace=True)
            
            # self.down_sampler = nn.Linear(input_dim, self.down_sample_size, bias=False)
            self.down_sampler = nn.Conv2d(input_dim, self.down_sample_size, 1, groups=self.opt["num_conv_group"], bias=False)
            nn.init.zeros_(self.down_sampler) # yb:for lora

            # self.up_sampler = nn.Linear(self.down_sample_size, output_dim, bias=False)
            self.up_sampler = nn.Conv2d(self.down_sample_size, output_dim, 1, groups=self.opt["num_conv_group"], bias=False)

            if use_bn:
                self.bn1 = nn.BatchNorm2d(self.down_sample_size)
                self.bn2 = nn.BatchNorm2d(output_dim)

            ### -------> yb: add
            if self.opt["is_before_layernorm"]:
                self.ln_before = nn.LayerNorm(output_dim)
            if self.opt["is_post_layernorm"]:
                self.ln_post = nn.LayerNorm(output_dim)
            ### <---------

        elif adapter_kind == "basic":
            self.activation = nn.ReLU(inplace=True)
            # self.conv = nn.Conv2d(input_dim, output_dim, 1, bias=False)
            self.conv = nn.Linear(input_dim, output_dim, bias=False)

            if use_bn:
                # self.bn = nn.BatchNorm2d(output_dim)
                self.bn = nn.BatchNorm1d(output_dim)

        else:
            raise NotImplementedError

    def forward(self, x, vis_token=None, caption=None, is_temporal=False):
        vis_token = self.conv_adapter(vis_token.transpose(2, 1))
        vis_token = self.fc(vis_token.squeeze(-1))
        vis_token = vis_token.permute(0, 2, 1).unsqueeze(-1)

        # vis_token = vis_token.squeeze(-1)
        # vis_token = vis_token.transpose(2, 1)
        # vis_token = self.fc(vis_token)
        # hw = int(math.sqrt(vis_token.size(1)))
        # vis_token = vis_token.view(vis_token.size(0), hw, hw, -1)
        # vis_token = F.interpolate(rearrange(vis_token, 'BF w h c -> BF c w h'), mode='bicubic',size=[int(math.sqrt(self.conv_dim_out)), int(math.sqrt(self.conv_dim_out))])
        # BF, C, _, _ = vis_token.size()
        # vis_token = vis_token.view(BF, C, -1).unsqueeze(-1)
        
        spatial_att_maps = None
        temporal_att_maps = None
    
        if self.adapter_kind == "bottleneck" and self.is_multimodal:
        ### -------> high dim att
            if caption == None:
                rep_token = repeat(self.my_tokens, 't d -> b t d', b=x.size(0))
            else:
                caption = self.fc_caption(caption)
                rep_token = rearrange(caption, 'b l d -> (b l) d')
                rep_token = repeat(caption, 'b d -> b t d', t = self.num_tk)               

            att_v2tk = torch.bmm(rep_token, vis_token.squeeze(-1))

            att_v2tk = F.softmax(att_v2tk, dim=-1)
            rep_token_res = torch.bmm(att_v2tk, vis_token.squeeze(-1).permute(0,2,1))


            rep_token = rep_token + rep_token_res
            

            att_tk2x = torch.bmm(x.squeeze(-1).permute(0,2,1), rep_token.permute(0,2,1))

            att_tk2x = F.softmax(att_tk2x, dim=-1)
            x_res = torch.bmm(att_tk2x, rep_token).permute(0,2,1).unsqueeze(-1)


            x = x + self.gate_av*x_res.contiguous()
            
            # ============================== Channel Attention ====================================    
            audio = vis_token.mean(dim=2).squeeze(-1) # [B*10, dim]
            audio_query_1 = F.relu(self.fc_affine_audio_1(audio)).unsqueeze(-2)  
            video_query_1 = F.relu(self.fc_affine_video_1(x.squeeze(-1).permute(0, 2, 1))) # [*, grid ** 2, width]       
            audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2) #  [*, width] 
            audio_video_query = F.relu(self.fc_affine_bottleneck(audio_video_query_raw))
            channel_att_maps = self.fc_affine_v_c_att(audio_video_query).sigmoid().reshape(x.size(0), 1, -1)      
            c_att_visual_feat = (x.squeeze(-1).permute(0, 2, 1) * (channel_att_maps + 1)) # [B*10, 36, 768]  

            # ============================== Spatial Attention =====================================
            # channel attended visual feature: [batch * 10, 36, v_dim]
            c_att_visual_query = F.relu(self.fc_affine_video_2(c_att_visual_feat))
            audio_query_2 = F.relu(self.fc_affine_audio_2(audio)).unsqueeze(-2)
            audio_video_query_2 = c_att_visual_query * audio_query_2
            spatial_att_maps_tmp = self.fc_affine_v_s_att(audio_video_query_2) 
            spatial_att_maps_sigmoid = spatial_att_maps_tmp.transpose(2, 1).sigmoid()
            spatial_att_maps_sigmoid = spatial_att_maps_sigmoid.transpose(2, 1)
            spatial_att_maps = self.fc_softmax(self.fc_tanh(spatial_att_maps_tmp).transpose(2, 1))
            c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat)

            # ============================== Temporal Attention =====================================
            audio = audio.view(audio.size(0) // self.frame_num, self.frame_num, -1)
            temporal_att_maps = self.temporal_gated(audio).unsqueeze(-1)
            
            alpha, beta = self.opt["alpha"], self.opt["beta"]
            # x = x.squeeze(-1).permute(0, 2, 1) * (alpha * channel_att_maps + beta * spatial_att_maps_sigmoid + 1 - alpha)
            # x = x.permute(0, 2, 1).unsqueeze(-1)

            gamma = self.opt["gamma"]
            x = x.squeeze(-1).permute(0, 2, 1)
            bs = x.size(0) // self.frame_num
            x = x.view(bs, self.frame_num, x.size(-2), x.size(-1)) # [B, 10, HxW, C]
            channel_att_maps_tmp = channel_att_maps.view(bs, self.frame_num, channel_att_maps.size(-2), channel_att_maps.size(-1))
            spatial_att_maps_sigmoid_tmp = spatial_att_maps_sigmoid.view(bs, self.frame_num, spatial_att_maps_sigmoid.size(-2), spatial_att_maps_sigmoid.size(-1))                
            x = x * (alpha * channel_att_maps_tmp + beta * spatial_att_maps_sigmoid_tmp + gamma * temporal_att_maps + 1 - alpha)
            x = rearrange(x, 'b t h c -> (b t) h c')
            x = x.permute(0, 2, 1).unsqueeze(-1)
            # <----------
            
            if self.opt["is_before_layernorm"]:
                x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

            z = self.down_sampler(x)
            

            if self.use_bn:
                # z = self.bn1(rearrange(z, 'N C L -> N L C') )
                # z = rearrange(z, 'N L C -> N C L')

                z = self.bn1(z)

            z = self.activation(z)
            output = self.up_sampler(z)
            if self.use_bn:
                # output = self.bn2(rearrange(output, 'N C L -> N L C') ) 
                # output = rearrange(output, 'N L C -> N C L')
                output = self.bn2(output)
    
        elif self.adapter_kind == "bottleneck":

            if self.opt["is_before_layernorm"]:
                x = self.ln_before(x.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

            z = self.down_sampler(x)

            if self.use_bn:
                z = self.bn1(z)
            # z = self.activation(z)
            output = self.up_sampler(z)
            if self.use_bn:
                output = self.bn2(output)
            

        elif self.adapter_kind == "basic":
            output = self.conv(x)
            if self.use_bn:
                output = self.bn(rearrange(output, 'N C L -> N L C') )
                output = rearrange(output, 'N L C -> N C L')


        if self.opt["is_post_layernorm"]:
            output = self.ln_post(output.squeeze(-1).permute(0,2,1)).permute(0,2,1).unsqueeze(-1)

        if self.gate is not None:
            output = self.gate * output
    

        return output, spatial_att_maps, temporal_att_maps


class MMIL_Net(nn.Module):
    def __init__(self, args, task, is_weak):
        super(MMIL_Net, self).__init__()

        self.args = args
        
        self.task_name = task

        clip_model = load_clip_to_cpu(self.args)
        clip_model.float()
        print("Building custom CLIP")

        # if not args["is_open_text"]:
        if self.task_name != "AVQA":
            self.classnames, _ = generate_category_list(self.args["data_path"], self.task_name)
            self.prompt_learner = PromptLearner(self.args, self.classnames, clip_model, is_weak)
            self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.text_encoder = TextEncoder(clip_model)
        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.clip_adapter = ClipAdapter(512, 4)
        self.clip_adapter_text = ClipAdapter(512, 4)
        # self.CMBS = CMBS(self.args)
        # self.audio_adapter = AudioAdapter()
        # 这里换成HTSAT
        self.htsat = HTSAT_Swin_Transformer(
            spec_size=esc_fig.htsat_spec_size,
            patch_size=esc_fig.htsat_patch_size,
            in_chans=1,
            num_classes=esc_fig.classes_num,
            window_size=esc_fig.htsat_window_size,
            config=esc_fig,
            depths=esc_fig.htsat_depth,
            embed_dim=esc_fig.htsat_dim,
            patch_stride=esc_fig.htsat_stride,
            num_heads=esc_fig.htsat_num_head
        )

        self.audio_projection = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 512)
            )
        
        checkpoint_path = os.path.join(esc_fig.checkpoint_path, esc_fig.checkpoint)
        tmp = torch.load(checkpoint_path, map_location='cpu')
        audio_projection_list = {k[24:]:v for k, v in tmp['state_dict'].items() if 'audio_projection' in k}
        self.audio_projection.load_state_dict(audio_projection_list)
        
        text_branch_list = {k[7:]:v for k, v in tmp['state_dict'].items() if 'text_branch' in k}
        text_transform_list = {k[7:]:v for k, v in tmp['state_dict'].items() if 'text_transform' in k}
        text_projection_list = {k[7:]:v for k, v in tmp['state_dict'].items() if 'text_projection' in k}

        # if not args["is_open_text"]:
        if self.task_name != "AVQA":
            self.clap_text_encoder = CLAPTextEncoder(self.args, self.classnames, [text_branch_list, text_transform_list, text_projection_list], is_weak=is_weak)
        # else:
        #     self.clap_text_encoder = CLAPTextEncoder(self.args, self.classnames, [text_branch_list, text_transform_list, text_projection_list])

        self.logit_scale_a = tmp['state_dict']['module.logit_scale_a']
        self.logit_scale_t = tmp['state_dict']['module.logit_scale_t']
        
        print("loading HTSAT")
        temp = dict()
        useless = ["logit_scale_a", "logit_scale_t", "patch_embed.mel_conv2d.weight", "patch_embed.mel_conv2d.bias",
                   "patch_embed.fusion_model.local_att.0.weight", "patch_embed.fusion_model.local_att.0.bias",
                   "patch_embed.fusion_model.local_att.1.weight", "patch_embed.fusion_model.local_att.1.bias",
                   "patch_embed.fusion_model.local_att.1.running_mean",
                   "patch_embed.fusion_model.local_att.1.running_var",
                   "patch_embed.fusion_model.local_att.1.num_batches_tracked",
                   "patch_embed.fusion_model.local_att.3.weight", "patch_embed.fusion_model.local_att.3.bias",
                   "patch_embed.fusion_model.local_att.4.weight", "patch_embed.fusion_model.local_att.4.bias",
                   "patch_embed.fusion_model.local_att.4.running_mean",
                   "patch_embed.fusion_model.local_att.4.running_var",
                   "patch_embed.fusion_model.local_att.4.num_batches_tracked",
                   "patch_embed.fusion_model.global_att.1.weight", "patch_embed.fusion_model.global_att.1.bias",
                   "patch_embed.fusion_model.global_att.2.weight", "patch_embed.fusion_model.global_att.2.bias",
                   "patch_embed.fusion_model.global_att.2.running_mean",
                   "patch_embed.fusion_model.global_att.2.running_var",
                   "patch_embed.fusion_model.global_att.2.num_batches_tracked",
                   "patch_embed.fusion_model.global_att.4.weight", "patch_embed.fusion_model.global_att.4.bias",
                   "patch_embed.fusion_model.global_att.5.weight", "patch_embed.fusion_model.global_att.5.bias",
                   "patch_embed.fusion_model.global_att.5.running_mean",
                   "patch_embed.fusion_model.global_att.5.running_var",
                   "patch_embed.fusion_model.global_att.5.num_batches_tracked", "logit_scale_a", "logit_scale_t"]
        for k, v in tmp['state_dict'].items():
            p = k.find(".", 7)
            if p != -1:
                if k[p + 1:] in useless:
                    continue
                temp[k[p + 1:]] = v
                # print(k[p+1:])
                if k[p + 1:] == "head.bias":
                    break
            else:
                p = k.find(".")
                if k[p + 1:] in useless:
                    continue
                temp[k[p + 1:]] = v
                # print(k[p+1:])
        # tmp = {k[10:]: v for k, v in tmp['state_dict'].items()}
        self.htsat.load_state_dict(temp, strict=True)

        self.audio_visual_contrastive_learner = AudioVisualContrastive(self.logit_scale)

        self.ViT = VisionTransformer(clip_model)

        
        self.hidden_list, self.hidden_list_a = [], []
        down_in_dim, down_in_dim_a = [], []
        down_out_dim, down_out_dim_a = [], []
        conv_dim, conv_dim_a = [], []


        ## ------------> for swin and htsat
        for idx_layer, my_blk_a in enumerate(self.htsat.layers):
            conv_dim_tmp_a = (my_blk_a.input_resolution[0] * my_blk_a.input_resolution[1])
            if my_blk_a.downsample is not None:
                down_in_dim_a.append(my_blk_a.downsample.reduction.in_features)
                down_out_dim_a.append(my_blk_a.downsample.reduction.out_features)

            for idx_layer, blk_a in enumerate(my_blk_a.blocks):
                hidden_d_size_a = blk_a.norm1.normalized_shape[0]
                self.hidden_list_a.append(hidden_d_size_a)
                conv_dim_a.append(conv_dim_tmp_a)
            

        ### ----------> for vit
        for idx_layer, my_blk in enumerate(self.ViT.transformer.resblocks) :
            hidden_d_size = my_blk.mlp.c_proj.in_features
            self.hidden_list.append(hidden_d_size)
            conv_dim_tmp = self.ViT.input_resolution
            conv_dim.append(conv_dim_tmp)

        # # ### <---------
        self.hidden_list = [768]*len(self.hidden_list)
        conv_dim = [50]*len(conv_dim)
        
        self.prompt_length = args["prompt_length"]

        
        
        # self.use_g_prompt = self.args["use_p_prompt"]
        # self.g_prompt_layer_idx = self.args["g_prompt_layer_idx"]

        #  -------------------------------- v_g_prompt ----------------------------------------
        # self.g_prompt_length = args["g_prompt_length"]
        # g_prompt_init = args['g_prompt_init']
        # num_g_prompt = len(self.g_prompt_layer_idx) if self.g_prompt_layer_idx is not None else 0
        # self.use_prefix_tune_for_g_prompt = self.args["use_prefix_tune_for_g_prompt"]

        # if not self.use_prefix_tune_for_g_prompt and not self.use_prefix_tune_for_g_prompt:
        #     self.use_g_prompt = False
        #     self.g_prompt_layer_idx = []

        # if self.use_g_prompt and self.g_prompt_length is not None and len(self.g_prompt_layer_idx) != 0:
        #     # if not self.use_prefix_tune_for_g_prompt:
        #     if not self.use_prefix_tune_for_g_prompt:
        #         g_prompt_shape=(num_g_prompt, self.g_prompt_length, args['embed_dim'])
        #         if g_prompt_init == 'zero':
        #             self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
        #         elif g_prompt_init == 'uniform':
        #             self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
        #             nn.init.uniform_(self.g_prompt, -1, 1)
        #     else:
        #         if args['same_key_value']:
        #             g_prompt_shape=(num_g_prompt, 1, self.g_prompt_length, args['embed_dim'])
        #             if g_prompt_init == 'zero':
        #                 self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
        #             elif g_prompt_init == 'uniform':
        #                 self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
        #                 nn.init.uniform_(self.g_prompt, -1, 1)
        #             self.g_prompt = self.g_prompt.repeat(1, 2, 1, 1)
        #         else:
        #             g_prompt_shape=(num_g_prompt, 2, self.g_prompt_length, args['embed_dim'])
        #             if g_prompt_init == 'zero':
        #                 self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
        #             elif g_prompt_init == 'uniform':
        #                 self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
        #                 nn.init.uniform_(self.g_prompt, -1, 1)
        # else:
        #     self.g_prompt = None
        #  -------------------------------- v_e_prompt ----------------------------------------

       

        # self.deep_audio_e_prompt = AudioPrompt(length=args['audio_prompt_length'], embed_dim=args['audio_embed_dim'], embedding_key=args['embedding_key'], prompt_init=args['prompt_init'],
        #             prompt_pool=args['prompt_pool'], prompt_key=args['prompt_key'], pool_size=args['pool_size'], top_k=args['top_k'], batchwise_prompt=args['batchwise_prompt'],
        #             prompt_key_init=args['prompt_key_init'],)

        self.is_use_shallow_adapter = args['use_shallow_adapter']
        if self.is_use_shallow_adapter:
            self.shallow_adapter_layer_idx = args['s_prompt_layer_idx']
            self.shallow_adapter_layer_num = len(self.shallow_adapter_layer_idx)
        else:
            self.shallow_adapter_layer_idx = None
            self.shallow_adapter_layer_num = None
            
        self.middle_adapter_layer_num = args['middle_adapter_layers']
        

        if self.is_use_shallow_adapter:
            self.audio_adapter_blocks_p1 = nn.ModuleList([
                VisualAdapter(input_dim=self.hidden_list_a[self.shallow_adapter_layer_idx[i]], output_dim=self.hidden_list_a[self.shallow_adapter_layer_idx[i]], 
                adapter_kind="bottleneck", task_name=self.task_name, dim_list=self.hidden_list_a, layer_idx=self.shallow_adapter_layer_idx[i],
                reduction_factor=8, 
                opt=args, use_bn=self.args['is_bn'], use_gate=self.args['is_gate'],
                num_tk=args['num_tokens'], conv_dim_in=conv_dim[self.shallow_adapter_layer_idx[i]], conv_dim_out=conv_dim_a[self.shallow_adapter_layer_idx[i]],
                linear_in=self.hidden_list[self.shallow_adapter_layer_idx[i]], linear_out=self.hidden_list_a[self.shallow_adapter_layer_idx[i]]       
                )
                for i in range(self.shallow_adapter_layer_num)])

            self.vis_adapter_blocks_p1 = nn.ModuleList([
                VisualAdapter(input_dim=self.hidden_list[self.shallow_adapter_layer_idx[i]], 
                output_dim=self.hidden_list[self.shallow_adapter_layer_idx[i]], adapter_kind="bottleneck", task_name=self.task_name,
                dim_list=self.hidden_list, layer_idx=self.shallow_adapter_layer_idx[i], reduction_factor=8, 
                opt=args, use_bn=self.args['is_bn'], use_gate=True,
                num_tk=args['num_tokens'], conv_dim_in=conv_dim_a[self.shallow_adapter_layer_idx[i]], conv_dim_out=conv_dim[self.shallow_adapter_layer_idx[i]],
                linear_in=self.hidden_list_a[self.shallow_adapter_layer_idx[i]], linear_out=self.hidden_list[self.shallow_adapter_layer_idx[i]]   
                )
                for i in range(self.shallow_adapter_layer_num)])

            self.audio_adapter_blocks_p2 = nn.ModuleList([
                VisualAdapter(input_dim=self.hidden_list_a[self.shallow_adapter_layer_idx[i]], output_dim=self.hidden_list_a[self.shallow_adapter_layer_idx[i]], adapter_kind="bottleneck", 
                task_name=self.task_name, dim_list=self.hidden_list_a, layer_idx=self.shallow_adapter_layer_idx[i], reduction_factor=8, 
                opt=args, use_bn=self.args['is_bn'], use_gate=self.args['is_gate'],
                num_tk=args['num_tokens'], conv_dim_in=conv_dim[self.shallow_adapter_layer_idx[i]], conv_dim_out=conv_dim_a[self.shallow_adapter_layer_idx[i]],
                linear_in=self.hidden_list[self.shallow_adapter_layer_idx[i]], linear_out=self.hidden_list_a[self.shallow_adapter_layer_idx[i]]
                )
                for i in range(self.shallow_adapter_layer_num)])

            self.vis_adapter_blocks_p2 = nn.ModuleList([
                VisualAdapter(input_dim=self.hidden_list[self.shallow_adapter_layer_idx[i]], output_dim=self.hidden_list[self.shallow_adapter_layer_idx[i]], adapter_kind="bottleneck", 
                task_name=self.task_name, dim_list=self.hidden_list, layer_idx=self.shallow_adapter_layer_idx[i], reduction_factor=8, 
                opt=args, use_bn=self.args['is_bn'], use_gate=True,
                num_tk=args['num_tokens'], conv_dim_in=conv_dim_a[self.shallow_adapter_layer_idx[i]], conv_dim_out=conv_dim[self.shallow_adapter_layer_idx[i]],
                linear_in=self.hidden_list_a[self.shallow_adapter_layer_idx[i]], linear_out=self.hidden_list[self.shallow_adapter_layer_idx[i]]   
                )
                for i in range(self.shallow_adapter_layer_num)])

        # TODO: 目前只进行全调优实施，还未开展冻结操作
        # TODO: 参数需与PC论文进行比对（还未比对）
        self.m_prompt_layer_idx = args["m_prompt_layer_idx"]

        self.is_use_mid_prompts = args["use_mid_prompt"]
        

        if self.is_use_mid_prompts:
            self.mid_e_prompt = M2Prompt(length=args['audio_prompt_length'], embed_dim=args['embed_dim'], 
                                            embedding_key=args['embedding_key'], prompt_init=args['prompt_init'],
                                            prompt_pool=args['prompt_pool'], prompt_key=args['prompt_key'], pool_size=args['pool_size'], 
                                            top_k=args['top_k'], batchwise_prompt=args['batchwise_prompt'],
                                            prompt_key_init=args['prompt_key_init'], num_layers=self.middle_adapter_layer_num, 
                                            use_prefix_tune_for_e_prompt=args["use_prefix_tune_for_d_prompt"],
                                            num_heads=8, same_key_value=args['same_key_value'])
            # self.ln_mid_prompt = nn.LayerNorm(args['embed_dim'])
            
        self.is_use_deep_prompts = args["use_deep_prompt"]
        self.d_prompt_layer_idx = self.args["d_prompt_layer_idx"]
        
        if self.is_use_deep_prompts:
            self.deep_visual_e_prompt = VisualEPrompt(length=args['d_prompt_length'], embed_dim=args['embed_dim'], embedding_key=args['embedding_key'], prompt_init=args['prompt_init'],
                        prompt_pool=args['prompt_pool'], prompt_key=args['prompt_key'], pool_size=args['pool_size'], top_k=args['top_k'], batchwise_prompt=args['batchwise_prompt'],
                        prompt_key_init=args['prompt_key_init'], num_layers=len(self.args["d_prompt_layer_idx"]), use_prefix_tune_for_e_prompt=args["use_prefix_tune_for_d_prompt"], same_key_value=args['same_key_value'])
            # self.ln_deep_prompt = nn.LayerNorm(args['embed_dim'])

        # --------------------------------- MIDDLE ADAPTER -------------------------------------
        # self.fc = CosineLinear(in_dim, out_dim)

        # self.middle_adapter_blocks_a_p1_list = []
        # self.middle_adapter_blocks_a_p2_list = []
        # self.middle_adapter_blocks_v_p1_list = []
        # self.middle_adapter_blocks_v_p2_list = []


        # self.middle_adapter_blocks_a_p1 = nn.ModuleList()
        # self.middle_adapter_blocks_a_p2 = nn.ModuleList()
        # self.middle_adapter_blocks_v_p1 = nn.ModuleList()
        # self.middle_adapter_blocks_v_p2 = nn.ModuleList()

        # self.get_middle_adapter()

        # --------------------------------- MIDDLE ADAPTER -------------------------------------
        
        # trainable cls token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, args['embed_dim'])) # if class_token else None
        # nn.init.normal_(self.cls_token, std=1e-6)

        if self.task_name == "AVQA":
            self.handler = AvqaDownTaskHandler()
        elif self.task_name == "AVS":
            self.handler = AVSDownTaskHandler()

        

        self.numtask = 0
        self.save_cnt = 0

    def update_fc(self):
        self.numtask +=1

    # def add_adapter_to_list(self):
    #     self.middle_adapter_blocks_a_p1_list.append(copy.deepcopy(self.middle_adapter_blocks_a_p1.requires_grad_(False)))
    #     self.middle_adapter_blocks_a_p2_list.append(copy.deepcopy(self.middle_adapter_blocks_a_p2.requires_grad_(False)))
    #     self.middle_adapter_blocks_v_p1_list.append(copy.deepcopy(self.middle_adapter_blocks_v_p1.requires_grad_(False)))
    #     self.middle_adapter_blocks_v_p2_list.append(copy.deepcopy(self.middle_adapter_blocks_v_p2.requires_grad_(False)))
    #     self.get_new_adapter()

    # def get_middle_adapter(self):

    #     self.middle_adapter_blocks_a_p1 = nn.ModuleList()
    #     self.middle_adapter_blocks_a_p2 = nn.ModuleList()
    #     self.middle_adapter_blocks_v_p1 = nn.ModuleList()
    #     self.middle_adapter_blocks_v_p2 = nn.ModuleList()

    #     for i in range(self.shallow_adapter_layer_num, self.middle_adapter_layer_num):
    #         self.middle_adapter_blocks_a_p1.append(Midadapter(d_model=self.hidden_list_a[i], bottleneck=64, dropout=0.1, init_option='lora'))
    #         self.middle_adapter_blocks_a_p2.append(Midadapter(d_model=self.hidden_list_a[i], bottleneck=64, dropout=0.1, init_option='lora'))
    #         self.middle_adapter_blocks_v_p1.append(Midadapter(d_model=self.hidden_list_a[i], bottleneck=64, dropout=0.1, init_option='lora'))
    #         self.middle_adapter_blocks_v_p2.append(Midadapter(d_model=self.hidden_list_a[i], bottleneck=64, dropout=0.1, init_option='lora'))


    def audio_padding_prompt(self, x, prompt):
        # prompt --> [batch_size, num_tokens, embed_dim]   num_tokens = 8
        # [40, 2 ,5 , 768]

        # audio feature 1 ： [40, 1024, 192] --> [40, 32, 32, 192]
        # audio feature 2 ： [40, 256, 384] --> [40, 16, 16, 384]
        # prompt : [40, 8, 768] --> [40, 8, 384]

        x = rearrange(x, 'b (h w) c -> b h w c', h = int(math.sqrt(x.shape[1])))
        b, n, _, d = x.shape
        prompt = prompt.reshape(b, n, n, -1)
        sd = prompt.shape[-1]
        x[:, :, :, :sd] += prompt
        
        x = rearrange(x, 'b h w c -> b (h w) c')
        
        return x

        # if c == 192:
        #     prompt = self.mid_audio_prompt_proj_1(prompt)
        # if c == 384:
        #     prompt = self.mid_audio_prompt_proj_2(prompt)
        
        # prompt=rearrange(prompt, 'b (n e) d -> b n e d', n = 4)
        # prompt = prompt.permute(1, 0, 2, 3)
        
        # x = rearrange(x, 'b (n n) d -> b n n d')
        # for i in range(4):
        #     x = x[:, :, i] + prompt[i]



    def get_numtask(self, task_id):
        self.numtask = task_id
        
    def get_dataset_name(self, dataset_name):
        self.task_name = dataset_name

    def clip_matching(self, visual_grd):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()
        x = self.clip_adapter(visual_grd)
        ratio = 0.2
        visual_grd = ratio * x + (1 - ratio) * visual_grd
        visual_grd = visual_grd / visual_grd.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner(visual_grd)


        text_features = self.text_encoder(prompts, tokenized_prompts)  # [n_cls, 512]
        x = self.clip_adapter_text(text_features)
        ratio = 0.2
        text_features = ratio * x + (1 - ratio) * text_features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * visual_grd @ text_features.t()
        return logits

    
    def clap_matching(self, audio_features):
        text_features = self.clap_text_encoder()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale_a * audio_features @ text_features.t()
        
        return logits
        


    def forward(self, wave, vis, question=None, train=False, mixup_lambda=None):
        
        b, t, c, w, h = vis.shape
        
        output_dict=self.av_prompt_forward(rearrange(vis, 'b t c w h -> (b t) c w h'), wave, 
                                           question=question,
                                           train=False,
                                           mixup_lambda=mixup_lambda)  # [B*10, 512]
        
        if self.task_name == "AVQA":
            return output_dict['out_qa'], output_dict['out_match_posi'], output_dict['vis_reduce_sim'], output_dict['aud_reduce_sim']
        elif self.task_name == "AVS":
            return output_dict['output_mask'], output_dict['v_map_list'], output_dict['a_fea_list']

        v_cls=output_dict['x']
        a_cls=output_dict['embedding']
        loss_audio_image=output_dict['logits_audio_image']
        loss_image_audio=output_dict['logits_image_audio']
        
        loss_vis_reduce_sim = output_dict['vis_reduce_sim']
        loss_aud_reduce_sim = output_dict['aud_reduce_sim']
        
        logits_v = self.clip_matching(v_cls)
        logits_a = self.clap_matching(a_cls)
        

        # 视频 和 音频 的 匹配分数
        #best score = 72.16
        w1 = logits_v / (logits_v + logits_a)
        w2 = 1-w1
        event_scores = 2 * logits_v + 2 * logits_a

        return event_scores, loss_audio_image, loss_image_audio, loss_vis_reduce_sim, loss_aud_reduce_sim
        

    def av_prompt_forward(self, vis, wave, mixup_lambda=None, longer_idx=None, question=None, train=False):
        
        x = self.ViT.conv1(vis)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2 ,width]

        cls_token = self.ViT.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        # x = torch.cat([self.ViT.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                        #   device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, witdh]
        # trainable cls
        x = torch.cat([cls_token, x], dim=1)  # shape = [*, grid ** 2 + 1, witdh]
        
        x = x + self.ViT.positional_embedding.to(x.dtype)
        # prompt_mask
        prompt_mask = None
        # if self.train:
        #     start = self.numtask * self.visual_e_prompt.top_k
        #     end = (self.numtask + 1) * self.visual_e_prompt.top_k
        #     single_prompt_mask = torch.arange(start, end).to(x.device)
        #     prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
        #     if end > self.visual_e_prompt.pool_size:
        #         prompt_mask = None
        

        # v_res = self.visual_e_prompt(x, prompt_mask=prompt_mask)
        # e_prompt = v_res['batched_prompt']

        if self.is_use_mid_prompts:
            mid_res_v = self.mid_e_prompt(x, task_id=self.numtask, task_key_norm=None, prompt_mask=prompt_mask, cls_features=cls_token, trainable=train)
            mid_e_prompt_v = mid_res_v['batched_prompt']
            # mid_e_prompt_v = self.ln_mid_prompt(mid_e_prompt_v)

        if self.is_use_deep_prompts:
            dp_res_v = self.deep_visual_e_prompt(x, prompt_mask=prompt_mask)
            dp_e_prompt_v = dp_res_v['batched_prompt']
            # dp_e_prompt_v = self.ln_deep_prompt(dp_e_prompt_v)

        x = self.ViT.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND


        y = wave.to(device=wave.device, non_blocking=True)
        

        y = y.view(y.size(0) * y.size(1), -1)
        y = self.htsat.spectrogram_extractor(y)  # (batch_size, 1, time_steps, freq_bins)
        y = self.htsat.logmel_extractor(y)  # (batch_size, 1, time_steps, mel_bins)
        y = y.transpose(1, 3)
        y = self.htsat.bn0(y)  # 使用修正后的输入
        y = y.transpose(1, 3)
        if self.htsat.training:
            y = self.htsat.spec_augmenter(y)
        if self.htsat.training and mixup_lambda is not None:
            y = do_mixup(y, mixup_lambda)
        y = self.htsat.reshape_wav2img(y)

        # a_res = self.audio_e_prompt(y, prompt_mask=prompt_mask)
        # y = a_res['prompted_embedding']

        
        # handle x and y
        frames_num = y.shape[2]
        y = self.htsat.patch_embed(y, longer_idx=longer_idx)
        
        if self.htsat.ape:
            y = y + self.htsat.absolute_pos_embed(y)
        y = self.htsat.pos_drop(y)
        
        if self.is_use_mid_prompts:
            mid_res_a = self.mid_e_prompt(y, task_id=self.numtask, task_key_norm=None, prompt_mask=prompt_mask, cls_features=cls_token, trainable=train)
            mid_e_prompt_a = mid_res_a['batched_prompt']

        # dp_res_a = self.deep_audio_e_prompt(y, prompt_mask=prompt_mask)
        # y = dp_res_a['prompted_embedding']
        # ------------------------------------- MID ----------------------------------------------
        # warning: !!! 一定要重新比对，重新过一遍流程
        

        # ------------------------------------- MID ----------------------------------------------
        cnt = 0
        # g_counter = -1 
        # e_counter = -1
        s_counter = 0 
        mid_counter = 0
        dp_counter = 0
        
        # with autocast(enabled=False):
        #     x = x.float()
        for idx_blk, blk in enumerate(self.htsat.layers):
            for idx_layer, layer in enumerate(blk.blocks):
                # compute audio
                attns = []

                if cnt in self.m_prompt_layer_idx and self.is_use_mid_prompts:
                    # print(mid_e_prompt_v[mid_counter].shape)
                    y = self.audio_padding_prompt(y, mid_e_prompt_a[mid_counter])
                    x = x + self.ViT.transformer.resblocks[cnt].attention(self.ViT.transformer.resblocks[cnt].ln_1(x), mid_e_prompt_v[mid_counter].permute(1, 2, 0, 3))
                    mid_counter += 1
                elif cnt in self.d_prompt_layer_idx and self.is_use_deep_prompts:
                    # 没加音频
                    x = x + self.ViT.transformer.resblocks[cnt].attention(self.ViT.transformer.resblocks[cnt].ln_1(x), dp_e_prompt_v[dp_counter].permute(1, 2, 0, 3))
                    dp_counter += 1
                else: 
                    x = x + self.ViT.transformer.resblocks[cnt].attention(self.ViT.transformer.resblocks[cnt].ln_1(x))
                

                y, attn = layer(y)
                
                if not layer.training:
                    attns.append(attn.unsqueeze(0))
                # ------------------------------------------------------ adpter -------------------------------------------------------
                
                # shallow
                if self.is_use_shallow_adapter and cnt in self.shallow_adapter_layer_idx:
                    # if cnt < self.shallow_adapter_layer_num:
                        
                    f_a = y

                    f_v = x.permute(1, 0, 2)

                    f_a_res, spatial_att_maps, temporal_att_maps = self.audio_adapter_blocks_p1[s_counter](f_a.permute(0, 2, 1).unsqueeze(-1),
                                                                    f_v.permute(0, 2, 1).unsqueeze(-1))
                    f_v_res, spatial_att_maps, temporal_att_maps = self.vis_adapter_blocks_p1[s_counter](f_v.permute(0, 2, 1).unsqueeze(-1),
                                                                    f_a.permute(0, 2, 1).unsqueeze(-1))
                
                # middle
                # elif cnt >= self.shallow_adapter_layer_num and cnt < self.shallow_adapter_layer_num + self.middle_adapter_layer_num:
                #     f_a_res = self.middle_adapter_blocks_a_p1[cnt-g_counter](f_a_res)
                #     f_v_res = self.middle_adapter_blocks_v_p1[cnt-g_counter](f_v_res)

                    f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)

                    f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                    x = f_v.permute(1, 0, 2)

                # ------------------------------------------------------ adpter -------------------------------------------------------
                x = x + self.ViT.transformer.resblocks[cnt].mlp(self.ViT.transformer.resblocks[cnt].ln_2(x))

                # ------------------------------------------------------ adpter -------------------------------------------------------
                # shallow
                if self.is_use_shallow_adapter and cnt in self.shallow_adapter_layer_idx:
                    # if cnt < self.shallow_adapter_layer_num:
                    f_v = x.permute(1, 0, 2)

                    f_a_res, spatial_att_maps, temporal_att_maps = self.audio_adapter_blocks_p2[s_counter](f_a.permute(0, 2, 1).unsqueeze(-1),
                                                                    f_v.permute(0, 2, 1).unsqueeze(-1))
                    f_v_res, spatial_att_maps, temporal_att_maps = self.vis_adapter_blocks_p2[s_counter](f_v.permute(0, 2, 1).unsqueeze(-1),
                                                                    f_a.permute(0, 2, 1).unsqueeze(-1))
                    
                    # middle
                    # elif cnt >= self.shallow_adapter_layer_num and cnt < self.shallow_adapter_layer_num + self.middle_adapter_layer_num:
                    #     f_a_res = self.middle_adapter_blocks_a_p1[cnt-g_counter](f_a_res)
                    #     f_v_res = self.middle_adapter_blocks_v_p1[cnt-g_counter](f_v_res)
                        
                    f_v = f_v + f_v_res.squeeze(-1).permute(0, 2, 1)
                    f_a = f_a + f_a_res.squeeze(-1).permute(0, 2, 1)

                    x = f_v.permute(1, 0, 2)
                    s_counter += 1
                # ------------------------------------------------------ adpter -------------------------------------------------------

                cnt += 1


            # compute audio
            if blk.downsample is not None:
                y = blk.downsample(y)
            if not blk.training:
                attn = torch.cat(attns, dim=0)
                attn = torch.mean(attn, dim=0)




        if self.task_name == "AVQA":
            x = x.permute(1, 0, 2)  # LND -> NLD
            out_qa, out_match_posi = self.handler(x, y, question)
            
            output_dict = {
            'out_qa': out_qa,
            'out_match_posi': out_match_posi, 
            'vis_reduce_sim': 0,
            'aud_reduce_sim': 0,
            }

            return output_dict
        
        elif self.task_name == "AVS":
            pred_masks, a_fea_list, v_map_list = self.handler(x, y)
            
            output_dict = {
                'output_mask': pred_masks,
                'a_fea_list': a_fea_list,
                'v_map_list': v_map_list
            }

            return output_dict
        

        y = self.htsat.norm(y)
        B, N, C = y.shape
        SF = frames_num // (2 ** (len(self.htsat.depths) - 1)) // self.htsat.patch_stride[0]
        ST = frames_num // (2 ** (len(self.htsat.depths) - 1)) // self.htsat.patch_stride[1]
        y = y.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
        B, C, F, T = y.shape
        # group 2D CNN
        c_freq_bin = F // self.htsat.freq_ratio
        y = y.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
        y = y.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
        # get latent_output
        fine_grained_latent_output = torch.mean(y, dim=2)
        fine_grained_latent_output = interpolate(fine_grained_latent_output.permute(0, 2, 1).contiguous(),
                                                 8 * self.htsat.patch_stride[1])

        latent_output = self.htsat.avgpool(torch.flatten(y, 2))
        latent_output = torch.flatten(latent_output, 1)

        # display the attention map, if needed

        y = self.htsat.tscam_conv(y)
        y = torch.flatten(y, 2)  # B, C, T

        fpx = interpolate(torch.sigmoid(y).permute(0, 2, 1).contiguous(), 8 * self.htsat.patch_stride[1])

        y = self.htsat.avgpool(y)
        y = torch.flatten(y, 1)

        output_dict = {
            'framewise_output': fpx,  # already sigmoided
            'clipwise_output': torch.sigmoid(y),
            'fine_grained_embedding': fine_grained_latent_output,
            'embedding': latent_output
        }
        #

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ViT.ln_post(x[:, 0, :])
    
        if self.ViT.proj is not None:
            x = x @ self.ViT.proj
        latent_output=self.audio_projection(latent_output)


        logits_audio_image, logits_image_audio = self.audio_visual_contrastive_learner(x, latent_output)
        
        
        output_dict = {
            'framewise_output': fpx,  # already sigmoided
            'clipwise_output': torch.sigmoid(y),
            'fine_grained_embedding': fine_grained_latent_output,
            'embedding': latent_output,
            'x':x,
            'logits_audio_image':logits_audio_image,
            'logits_image_audio':logits_image_audio,
            # 'logits_visual_prompt': v_res['batched_prompt'],
            # 'logits_audio_prompt_lr': a_res['logits_audio_prompt_lr'],
            # 'logits_audio_prompt_tb': a_res['logits_audio_prompt_tb'],
            'vis_reduce_sim': 0,
            'aud_reduce_sim': 0,
        }
        return output_dict


    def extract_vector(self, image_features, audio_features):
        image_features = self.ViT(rearrange(image_features, 'b t c w h -> (b t) c w h'))

        return image_features, audio_features
    

    # vis, wave, vis_instance_token=None, aud_instance_token=None, mixup_lambda=None, longer_idx=None
    def interface(self, image, wave, question=None, mixup_lambda=None):

        output_dict = self.av_prompt_forward(rearrange(image, 'b t c w h -> (b t) c w h'), wave, question=question, mixup_lambda=mixup_lambda)  # [B*10, 512]

        if self.task_name == "AVQA":
            return output_dict['out_qa'], output_dict['out_match_posi'], output_dict['vis_reduce_sim'], output_dict['aud_reduce_sim']
        elif self.task_name == "AVS":
            return output_dict['output_mask'], output_dict['a_fea_list'], output_dict['v_map_list']

        v_cls=output_dict['x']
        a_cls=output_dict['embedding']

        # 保存特征用于T-sne
        # save_root = os.path.join('tsne', self.task_name)
        # v = v_cls
        # a = a_cls
        # v_path = os.path.join(save_root, 'video', str(self.save_cnt)+'.npy')
        # a_path = os.path.join(save_root, 'audio', str(self.save_cnt)+'.npy')
        # np.save(v_path, v.cpu())
        # np.save(a_path, a.cpu())
        # self.save_cnt += 1

        loss_audio_image=output_dict['logits_audio_image']
        loss_image_audio=output_dict['logits_image_audio']
        
        loss_vis_reduce_sim = output_dict['vis_reduce_sim']
        loss_aud_reduce_sim = output_dict['aud_reduce_sim']

        logits_v = self.clip_matching(v_cls)
        logits_a = self.clap_matching(a_cls)
        

        # 视频 和 音频 的 匹配分数
        #best score = 72.16
        w1 = logits_v / (logits_v + logits_a)
        w2 = 1-w1
        event_scores = 2 * logits_v + 2 * logits_a

        return event_scores, loss_audio_image, loss_image_audio, loss_vis_reduce_sim, loss_aud_reduce_sim