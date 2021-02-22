
import torch
from Unet_parts import *
from torch.nn import init
from utils import *
#from utils import nested_tensor_from_tensor_list
from CT_transformer import *
from position import *
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownPoolDoubleConv(64, 128)
        self.down2 = DownPoolDoubleConv(128, 256)
        self.down3 = DownPoolDoubleConv(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownPoolDoubleConv(512, 1024 // factor)
        self.up1 = UpDoubleConv(1024, 512 // factor, bilinear)
        self.up2 = UpDoubleConv(512, 256 // factor, bilinear)
        self.up3 = UpDoubleConv(256, 128 // factor, bilinear)
        self.up4 = UpDoubleConv(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# class CT_Unetv1():
#     def __init__(self, args,n_channels, n_classes, bilinear=True,position_sin=True):
#         super(CT_Unetv1, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = DownPoolDoubleConv(64, 128)
#         self.down2 = DownPoolDoubleConv(128, 256)
#         self.down3 = DownPoolDoubleConv(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = DownPoolDoubleConv(512, 1024 // factor)
#         self.up1 = UpDoubleConv(1024, 512 // factor, bilinear)
#         self.up2 = UpDoubleConv(512, 256 // factor, bilinear)
#         self.up3 = UpDoubleConv(256, 128 // factor, bilinear)
#         self.up4 = UpDoubleConv(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)
#         self.hidden_dim = args.hidden_dim
#         self.transformer0 = CT_transformer()
#         self.transformer1 = CT_transformer()
#         self.transformer2 = CT_transformer()
#         self.transformer3 = CT_transformer()
#         self.position = build_position_encoding(position_sin)
#
#
#
#
#     def get_posision(self,input):
#         pos = []
#         for x in input:
#             pos.append(self.position(x))
#         return pos
#
#     def before_transformer(self,features):
#         input_proj_list = []
#         for _ in range(3):
#             in_channels = features[_].shape[1]
#             input_proj_list.append(nn.Sequential(
#                 nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
#                 nn.GroupNorm(32, self.hidden_dim),
#             ))
#         srcs = []
#         masks = []
#         for l, feat in enumerate(features):
#             src, mask = feat.decompose()
#             srcs.append(self.input_proj[l](src))
#             masks.append(mask)
#         return srcs,masks
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#
#         input0 = [x3,x4,x5]
#         input1 = [x2,x3,x4]
#         input2 = [x1,x2,x3]
#         input3 = [x,x1,x2]
#
#         pos0 = self.get_posision(input0,self.position)
#         pos1 = self.get_posision(input1, self.position)
#         pos2 = self.get_posision(input2, self.position)
#         pos3 = self.get_posision(input3, self.position)
#
#         input0 = nested_tensor_from_tensor_list(input0)
#         input1 = nested_tensor_from_tensor_list(input1)
#         input2 = nested_tensor_from_tensor_list(input2)
#         input3 = nested_tensor_from_tensor_list(input3)
#
#         input0_scr, input0_mask = self.before_transformer(input0)
#         input1_scr, input1_mask = self.before_transformer(input1)
#         input2_scr, input2_mask = self.before_transformer(input2)
#         input3_scr, input3_mask = self.before_transformer(input3)
#
#         out0 = x4
#         out1 = self.transformer0(input0_scr,input0_mask,pos0,out0) ##xi-xi+2 for encoder ->memory ; memory and outi for decoder
#         out2 = self.transformer1(input1_scr,input1_mask,pos1,out1)
#         out3 = self.transformer2(input2_scr,input2_mask,pos2,out2)
#         out4 = self.transformer3(input3_scr,input3_mask,pos3,out3)
#
#         # x = self.up1(x5, x4)
#         # x = self.up2(x, x3)
#         # x = self.up3(x, x2)
#         # x = self.up4(x, x1)
#         logits = self.outc(out4)
#         return logits

class CT_Unetv2(nn.Module):
    def __init__(self, n_channels, n_classes, args, bilinear=True):
        super(CT_Unetv2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownPoolDoubleConv(64, 128)
        self.down2 = DownPoolDoubleConv(128, 256)
        self.down3 = DownPoolDoubleConv(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownPoolDoubleConv(512, 1024 // factor)
        self.up1 = UpDoubleConv(1024, 512 // factor, bilinear)
        self.up2 = UpDoubleConv(512, 256, bilinear)
        self.up3 = UpDoubleConv(256, 128 // factor, bilinear)
        self.up4 = UpDoubleConv(192, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.hidden_dim = args.hidden_dim
        self.shape = args.shape
        self.batch_size = args.batch_size
        self.hidden_dim = args.hidden_dim
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.num_encoder_layers = args.num_encoder_layers
        self.dim_feedforward = args.dim_feedforward
        self.dropout = args.dropout
        self.activation = args.activation
        self.num_feature_levels = args.num_feature_levels
        self.num_channels = args.num_channels_True if bilinear else args.num_channels_False
        self.num_feature_levels = args.num_feature_levels
        self.input_proj_list = []
        for _ in range(len(self.num_channels)):
            in_channels = self.num_channels[_]
            self.input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                nn.GroupNorm(32, self.hidden_dim),
            ))
        self.input_proj = nn.ModuleList(self.input_proj_list)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer0 = CT_transformer_e(d_model = self.d_model, nhead=self.nhead,num_encoder_layers = self.num_encoder_layers,
                                             dim_feedforward = self.dim_feedforward,dropout = self.dropout,activation = self.activation,
                                             num_feature_levels = self.num_feature_levels,
                                             batch_size = self.batch_size,num_query = (self.shape/8)**2,
                                             n_points = (self.shape/8)+(self.shape/4)+(self.shape/2),
                                             outchannel=self.num_channels[3])
        self.transformer1 = CT_transformer_e(d_model = self.d_model, nhead=self.nhead,num_encoder_layers = self.num_encoder_layers,
                                             dim_feedforward = self.dim_feedforward,dropout = self.dropout,activation = self.activation,
                                             num_feature_levels = self.num_feature_levels,
                                             batch_size = self.batch_size,num_query = (self.shape/4)**2,
                                             n_points = (self.shape/1)+(self.shape/2)+(self.shape/4),
                                             outchannel=self.num_channels[2])
        self.transformer2 = CT_transformer_e(d_model = self.d_model, nhead=self.nhead,num_encoder_layers = self.num_encoder_layers,
                                             dim_feedforward = self.dim_feedforward,dropout = self.dropout,activation = self.activation,
                                             num_feature_levels = self.num_feature_levels,
                                             batch_size=self.batch_size,num_query = (self.shape/2)**2,
                                             n_points = (self.shape*2)+(self.shape/1)+(self.shape/2),
                                             outchannel=self.num_channels[1])
        self.transformer3 = CT_transformer_e(d_model = self.d_model, nhead=self.nhead,num_encoder_layers = self.num_encoder_layers,
                                             dim_feedforward = self.dim_feedforward,dropout = self.dropout,activation = self.activation,
                                             num_feature_levels = self.num_feature_levels,
                                             batch_size = self.batch_size,num_query = (self.shape/1)**2,
                                             n_points = (self.shape*2)+(self.shape*2)+(self.shape/1),
                                             outchannel=self.num_channels[0])
        self.position = build_position_encoding(args=args)


    def get_posision(self, input):
        pos = []
        for x in input:
            pos.append(self.position(x))
        return pos

    def before_transformer(self, features,start):
        srcs = []
        for l, feat in enumerate(features):
            #src, mask = feat.decompose()
            srcs.append(self.input_proj[start[l]](feat))
        return srcs

    def forward(self, img):
        x = img
        x1 = self.inc(x)      #shape,3 -> shape,64
        x2 = self.down1(x1)   #shape,64 -> shape/2,128
        x3 = self.down2(x2)   #shape/2,128 -> shape/4,256
        x4 = self.down3(x3)   #shape/4,256 -> shape/8,512
        x5 = self.down4(x4)   #shape/8,512 -> shape/16,512

        # input0 = [x3, x4, x5]
        # #input0 = nested_tensor_from_tensor_list(input0)
        # pos0 = self.get_posision(input0)
        # input0_scr = self.before_transformer(input0,start=[2,3,4])
        # out0 = self.transformer0(input0_scr, pos0)

        # input1 = [x2, x3, out0]
        # #input1 = nested_tensor_from_tensor_list(input1)
        # pos1 = self.get_posision(input1)
        # input1_scr = self.before_transformer(input1,start=[1,2,3])
        # out1 = self.transformer1(input1_scr, pos1)


        x = self.up1(x5, x4)
        x = self.up2(x, x3)

        input2 = [x1, x2, x]
        #input2 = nested_tensor_from_tensor_list(input2)
        pos2 = self.get_posision(input2)
        input2_scr = self.before_transformer(input2,start=[0,1,2])
        out2 = self.transformer2(input2_scr, pos2)
        #
        #
        # #x = self.up4(out2, x1)
        #
        # input3 = [x1, x1, out2]
        # #input3 = nested_tensor_from_tensor_list(input3)
        # pos3 = self.get_posision(input3)
        # input3_scr = self.before_transformer(input3,start=[0,0,1])
        # out3 = self.transformer3(input3_scr, pos3)


        # x = self.up1(x5, x4)
        # x = self.up2(out0, x3)
        #x = self.up3(x, x2)
        x = self.up4(out2, x1)
        logits = self.outc(x)
        return logits

class Network:
    def __init__(self, args, in_channel=3, model_name='', gpu_ids=[], init_type='normal', init_gain=0.02):
        self.gpu_ids = gpu_ids
        self.init_type = init_type
        self.init_gain = init_gain
        self.args = args
        if model_name == 'unet':
            self.model = UNet(in_channel, n_classes=2, bilinear=True)
        elif model_name == 'CT_Unet':
            self.model = CT_Unetv2(in_channel, n_classes=2, args = self.args,bilinear=True)
        # elif model_name == 'CT_Unet':
        #     self.model = CT_Unetv2(in_channel, n_classes=2, bilinear=True)
        self._init_model()

    def _init_model(self):
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if self.init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, self.init_gain)
                elif self.init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=self.init_gain)
                elif self.init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif self.init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=self.init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % self.init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find(
                    'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, self.init_gain)
                init.constant_(m.bias.data, 0.0)
        self.model.to(self.gpu_ids[0])
        self.model = torch.nn.DataParallel(self.model, self.gpu_ids)
        print('initialize network with %s' % self.init_type)
        self.model.apply(init_func)