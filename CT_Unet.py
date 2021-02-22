import os
from utils import *
class Networks(object):
    def __init__(self, args):
        os.makedirs(self.paras['model_path'], exist_ok=True)
        os.makedirs(self.paras['log_path'] + '/train', exist_ok=True )
        if args.is_val:
            os.makedirs(self.paras['log_path'] + '/val', exist_ok=True)

        self.model = Nets(in_channel=3, model_name='unet', gpu_ids=config['GPU_IDS']).model



