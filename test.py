import torch
import CrossAtten as ct

# def get_reference_points(spatial_shapes,n_points):
#     reference_points_list = []
#     for index, (H_, W_) in enumerate(spatial_shapes):
#         ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32),
#                                       torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32))
#         if index == 0:  ####batch_size 从大到小
#             ref_y = ref_y[0::2, 0::2]
#             ref_x = ref_x[0::2, 0::2]
#         elif index == 2:
#             tmp_y = torch.zeros((H_ * 2, W_ * 2))
#             tmp_y[0::2,0::2] = ref_y
#             tmp_y[1::2,1::2] = ref_y
#             tmp_x = torch.zeros((H_ * 2, W_ * 2))
#             tmp_x[0::2,0::2] = ref_x
#             tmp_x[1::2,1::2] = ref_x
#             ref_y = tmp_y
#             ref_x = tmp_x
#         shape_H = torch.zeros((2,1))+H_
#         shape_W = torch.zeros((2,1))+W_
#         ref_y = ref_y.reshape(-1)[None] / shape_H  ##maybe b 1 b h*w
#         ref_x = ref_x.reshape(-1)[None] / shape_W  ##maybe b 1
#
#
#         x1 = torch.zeros((W_))
#         x2 = torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32)/H_
#         x2 = x2[None,None,:].repeat(2,n_points,1)
#
#         y1= torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32)/W_
#         y1 = y1[None,None,:].repeat(2,n_points,1)
#         y2 = torch.zeros((H_))
#
#         ref_1 = ref_x[:, :, None] + x1[None, None, :]  # b h*w n1
#         ref_11 = torch.stack((ref_1,y1),-1)
#
#         ref_2 = ref_y[:, :, None] + y2[None, None, :]  # b h*w n1
#         ref_22 = torch.stack((x2,ref_2),-1)
#
#         ref = torch.cat((ref_11, ref_22), 2)  ##b h*w n1 2
#         reference_points_list.append(ref)
#     reference_points = torch.cat(reference_points_list, 2)  ##b h*w*level 2
#     # reference_points = reference_points[:, :, None] * valid_ratios[:,None]  ##b h*w*level 1 2 ; b 1 level 2 -> b h*w*level level 2
#     return reference_points  # b h*w n1+n2+n3
#
# spatial_shapes = [(8,8),(4,4),(2,2)]
# result = get_reference_points(spatial_shapes,16)
# print(result[:, :, None, :, :].repeat(1,1,8,1,1).shape)
# print(result[0,0,:,:])
# spatial_shapes = torch.as_tensor(spatial_shapes)
# level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
# print(level_start_index)
# print(level_start_index[0:1])
# print(1/2 == 1/2)
from position import *
from main import args
position = build_position_encoding(args=args)
def get_posision(input):
    pos = []
    for x in input:
        pos.append(position(x))
    return pos
ins = torch.ones((2,2,64,4,4))
pos = get_posision(ins)
print(pos[0].shape)