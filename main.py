import argparse
from utils import *
from Dataset import *
from Segmentor import *



def get_args_parser():
    parser = argparse.ArgumentParser(description='Seg:CrossTransformer', add_help=False)

    ###Seed###
    parser.add_argument('--seed', type=int, default=40, help='')

    ###Network###
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers of dataloader')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--max_epoch', type=int, default=150, help='max number of training epochs')
    parser.add_argument('--project_param', type=str, default='Project_params.yml', help='some params for project')
    parser.add_argument('--print_info_interval', type=int, default=40, help='Number of iterations between printing infos')
    parser.add_argument('--save_img_interval', type=int, default=20, help='Number of iterations between saving result img')
    parser.add_argument('--model_path', type=str, default=r'./Checkpoints', help='dir to save checkpoints')
    parser.add_argument('--log_path', type=str, default='./Results', help='log path')
    parser.add_argument('--lr_scheduler_mode', type=str, default='poly', help='lr scheduler mode')
    parser.add_argument('--optimizer', type=str, default='Adam', help='select optimizer for training, '
                                                                       'suggest using \'admaw\' until the'
                                                                       ' very final stage then switch to \'sgd\'')
    parser.add_argument('--Network', type=str, default='CT_Unet', help='')#unet CT_Unet
    parser.add_argument('--is_val', type=boolean_string, default=True, help='is evaluate the model')
    parser.add_argument('--num_channels_True', type=list, default=[64,128,256,512,512], help='')
    parser.add_argument('--num_channels_False', type=list, default=[64, 128, 256, 512, 1024], help='')

    ###DATA
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=8, help='input batch size for valuing')
    parser.add_argument('--dataset', type=str, default='generate_dep_info/train.csv', metavar='str', help='dataset name')
    parser.add_argument('--val_dataset', type=str, default='generate_dep_info/val.csv', metavar='str', help='val dataset name')
    parser.add_argument('--is_shuffle', type=boolean_string, default=True, help='is shuffle in training')

    #### transformer###
    parser.add_argument('--hidden_dim', type=int, default=256, help='')
    parser.add_argument('--shape', type=int, default=256, help='')
    parser.add_argument('--d_model', type=int, default=256, help='')
    parser.add_argument('--nhead', type=int, default=2, help='')
    parser.add_argument('--num_encoder_layers', type=int, default=1, help='')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='')
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--activation', type=str, default='relu', help='')
    parser.add_argument('--num_feature_levels', type=int, default=3, help='')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    return parser


# parser = argparse.ArgumentParser('Cross_Transformer train and evaluation script', parents=[get_args_parser()])
# args = parser.parse_args()


def main(args, config):
    torch.cuda.set_device(0)
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.empty_cache()
    dataloader = get_data_loader(args, config.params)
    network = Segmentor(args=args, config=config.params, data_loaders = dataloader)
    network.train_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cross_Transformer train and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    config = ParamsParser(args.project_param)
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args,config)


