from utils import *
import torch.nn as nn
import copy
import torch.nn.functional as F
from CrossAtten import *

# class CT_transformer(nn.Module):
#     def __init__(self,d_model=256, nhead=8,
#                  num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024, dropout=0.1,
#                  activation="relu", return_intermediate_dec=False,
#                  num_feature_levels=3, batch_size=0, num_query=0, n_points=0,
#                  two_stage_num_proposals=300):
#         super().__init__()
#         self.d_model = d_model
#         self.nhead = nhead
#         self.two_stage_num_proposals = two_stage_num_proposals
#
#         encoder_layer = CorssTransformerEncoderLayer(d_model, dim_feedforward,
#                                                           dropout, activation,
#                                                           num_feature_levels, nhead, enc_n_points)
#         self.encoder =  CorssTransformerEncoder(encoder_layer, num_encoder_layers)
#
#         decoder_layer = CrossTransformerDecoderLayer(d_model, dim_feedforward,
#                                                           dropout, activation,
#                                                           num_feature_levels, nhead, dec_n_points)
#         self.decoder =  CrossTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
#
#         self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
#
#         self.reference_points = nn.Linear(d_model, 2)
#
#     def forward(self,srcs,pos):
#         src_flatten = []
#         lvl_pos_embed_flatten = []
#         spatial_shapes = []
#         for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos)):
#             bs, c, h, w = src.shape
#             spatial_shape = (h, w)
#             spatial_shapes.append(spatial_shape)
#             src = src.flatten(2).transpose(1, 2)  ##b c h w -> b c (hw) -> b (hw) c
#             pos_embed = pos_embed.flatten(2).transpose(1, 2)  ##b c h w -> b c (hw) -> b (hw) c
#             lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)  ##猜测 对pos_embed 加上每层特征的level_embed
#             lvl_pos_embed_flatten.append(lvl_pos_embed)
#             src_flatten.append(src)
#         src_flatten = torch.cat(src_flatten, 1)
#         lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
#         spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
#         level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
#         #valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)  # b * n_level * 2
#
#         memory = self.encoder

class CT_transformer_e(nn.Module):
    def __init__(self,d_model=256, nhead=8,
                 num_encoder_layers=3, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=3, batch_size=0, num_query=0, n_points=0, outchannel= 0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = CorssTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          nhead, n_points,num_query,batch_size,outchannel)
        self.encoder = CorssTransformerEncoder(encoder_layer, num_encoder_layers, batch_size, num_query)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self.reference_points = nn.Linear(d_model, 2)

    def get_valid_ratio(self,mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1) ###b h w ->b h->b
        valid_W = torch.sum(~mask[:, 0, :], 1) ###b h w-> b w->b
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1) ##b 2
        return valid_ratio

    def forward(self,srcs, pos):
        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos)):
            bs, c, h, w = src.shape
            spatial_shape = (h,w)
            #print('ss:',spatial_shape)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  ##b c h w -> b c (hw) -> b (hw) c
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  ##b c h w -> b c (hw) -> b (hw) c
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)  ##猜测 对pos_embed 加上每层特征的level_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        memory = self.encoder(src_flatten,spatial_shapes,level_start_index,lvl_pos_embed_flatten)

        return memory


class CorssTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer,num_layers, batch_size,num_query):
        super(CorssTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_query = num_query
    @staticmethod
    def get_reference_points(spatial_shapes, num_query,batch_size,device):
        reference_points_list = []
        num_query = int(num_query)
        base_H_ , base_W_ = spatial_shapes[1]
        for index, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32,device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32,device=device))
            if H_//base_H_ == 2:  ####batch_size 从大到小
                ref_y = ref_y[0::2, 0::2]
                ref_x = ref_x[0::2, 0::2]
            elif H_/base_H_  == 1/2:
                tmp_y = torch.zeros((H_ * 2, W_ * 2),device=device)
                tmp_y[0::2, 0::2] = ref_y
                tmp_y[1::2, 1::2] = ref_y
                tmp_x = torch.zeros((H_ * 2, W_ * 2),device=device)
                tmp_x[0::2, 0::2] = ref_x
                tmp_x[1::2, 1::2] = ref_x
                ref_y = tmp_y
                ref_x = tmp_x
            shape_H = torch.zeros((batch_size, 1),device=device) + H_
            shape_W = torch.zeros((batch_size, 1),device=device) + W_
            #print(ref_y.device, shape_H.device)
            ref_y = ref_y.reshape(-1)[None] / shape_H  ##maybe b 1 b h*w
            ref_x = ref_x.reshape(-1)[None] / shape_W  ##maybe b 1

            x1 = torch.zeros((W_),device=device)
            x2 = torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32,device=device) / H_
            x2 = x2[None, None, :].repeat(batch_size, num_query, 1)

            y1 = torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32,device=device) / W_
            y1 = y1[None, None, :].repeat(batch_size, num_query, 1)
            y2 = torch.zeros((H_),device=device)

            ref_1 = ref_x[:, :, None] + x1[None, None, :]  # b h*w n1
            ref_11 = torch.stack((ref_1, y1), -1)

            ref_2 = ref_y[:, :, None] + y2[None, None, :]  # b h*w n1
            ref_22 = torch.stack((x2, ref_2), -1)

            ref = torch.cat((ref_11, ref_22), 2)  ##b h*w n1 2
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 2)  ##b h*w*level 2
        # reference_points = reference_points[:, :, None] * valid_ratios[:,None]  ##b h*w*level 1 2 ; b 1 level 2 -> b h*w*level level 2
        return reference_points  # b h*w n1+n2+n3

    def forward(self, src, spatial_shapes, level_start_index, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, self.num_query, self.batch_size, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class CorssTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_heads=8, n_points=0, num_query=0,batch_size=0,outchannel = 0):
        super().__init__()

        # self attention
        self.self_attn = CrossAtten(d_model, n_heads, n_points, num_query, out_channel= outchannel)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(outchannel)
        self.linear3 = nn.Linear(d_model,outchannel)
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.linear3(src)
        src = self.norm2(src)

        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src_query = src[:,level_start_index[1]:level_start_index[2],:]
        #print(src_query.shape)
        src2 = self.self_attn(src_query, reference_points, self.with_pos_embed(src, pos), spatial_shapes, level_start_index,
                              padding_mask)
        src = src_query + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        b,_,c = src.shape
        h,w = spatial_shapes[1]
        src = src.permute(0, 2, 1).view(b,c,h,w)
        return src


class CrossTransformerDecoder(nn.Module):
    def __init__(self,d_model=256, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=3, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        pass

    def forward(self):
        pass

class CrossTransformerDecoderLayer(nn.Module):
    def __init__(self,d_model=256, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=3, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        pass

    def forward(self):
        pass

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")