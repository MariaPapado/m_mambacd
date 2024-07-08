import sys
sys.path.append('/home/mariapap/Videos')

import argparse
import os
import time
from PIL import Image
import numpy as np
import cv2
from MambaCD.changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from MambaCD.changedetection.models.STMambaSCD import STMambaSCD
import imageio
from tqdm import tqdm
import MambaCD.changedetection.datasets.imutils as imutils

class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        #self.train_data_loader = make_data_loader(args)

        self.deep_model = STMambaSCD(
            output_cd = 2, 
            output_clf = 7,
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            ) 
        self.deep_model = self.deep_model.cuda()
        #self.model_save_path = os.path.join(args.model_param_path, args.dataset,
        #                                    args.model_type + '_' + str(time.time()))
        #self.lr = args.learning_rate
        #self.epoch = args.max_iters // args.batch_size

        #if not os.path.exists(self.model_save_path):
        #    os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.deep_model.eval()
        
        
    def validation(self):
        dir = '/home/mariapap/Videos/dataset_v1/dataset/test//'
        ids = os.listdir(os.path.join(dir, 'im1'))

        torch.cuda.empty_cache()

        preds_all = []
        labels_all = []
        
        for _, id in enumerate(tqdm(ids)):
            pre_change_imgs = np.array(imageio.imread(os.path.join(dir, 'im1', id)), np.float32)
            post_change_imgs = np.array(imageio.imread(os.path.join(dir, 'im2', id)), np.float32)
            
            

            pre_change_imgs = imutils.normalize_img(pre_change_imgs)  # imagenet normalization
            pre_change_imgs = np.transpose(pre_change_imgs, (2, 0, 1))

            post_change_imgs = imutils.normalize_img(post_change_imgs)  # imagenet normalization
            post_change_imgs = np.transpose(post_change_imgs, (2, 0, 1))

            pre_change_imgs = torch.from_numpy(pre_change_imgs).unsqueeze(0)
            post_change_imgs = torch.from_numpy(post_change_imgs).unsqueeze(0)
            
            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()


            output_1, output_semantic_t1, output_semantic_t2 = self.deep_model(pre_change_imgs, post_change_imgs)
            print('aaaaaaaaaaa', output_1)


            change_mask = torch.argmax(output_1, axis=1)

            preds_A = torch.argmax(output_semantic_t1, dim=1)
            preds_B = torch.argmax(output_semantic_t2, dim=1)
            preds_A = (preds_A*change_mask.squeeze().long()).cpu().numpy()
            preds_B = (preds_B*change_mask.squeeze().long()).cpu().numpy()

            
            change_mask = change_mask.cpu().numpy()
            
            change_mask = np.array(change_mask[0]*255, dtype=np.uint8)
            change_mask = Image.fromarray(change_mask)
            change_mask.save('./results/{}'.format(id))

        return 'ok'         

def main():
    parser = argparse.ArgumentParser(description="Training on SECOND dataset")
    parser.add_argument('--cfg', type=str, default='./changedetection/configs/vssm1/vssm_small_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str) #, default='/notebooks/MambaCD/changedetection/saved_models/SECOND/MambaSCD_Small_1714041444.9677947/16000_model.pth')

    parser.add_argument('--dataset', type=str, default='SECOND')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='/data/ggeoinfo/datasets/xBD/train')
    parser.add_argument('--train_data_list_path', type=str, default='/data/ggeoinfo/datasets/xBD/xBD_list/train_all.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/notebooks/hi_respect/test')
    parser.add_argument('--test_data_list_path', type=str, default='/notebooks/hi_respect/list/test_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MambaSCD')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')

#    parser.add_argument('--resume', type=str, default='/notebooks/backup/changedetection/saved_models/hi_respect/best_Mamba/4500_model__.pth') #prwth petuxesa
    parser.add_argument('--resume', type=str, default='./pretrained_weights/2800_model.pth') #2h petuxesa

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()

    trainer = Trainer(args)
#    trainer.training()
    trainer.validation()

if __name__ == "__main__":
    main()
