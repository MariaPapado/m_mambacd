import sys
sys.path.append('/notebooks')

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
from MambaCD.changedetection.datasets.make_data_loader import SemanticChangeDetectionDatset, make_data_loader
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.STMambaSCD import STMambaSCD
import MambaCD.changedetection.utils_func.lovasz_loss as L
from torch.optim.lr_scheduler import StepLR
from MambaCD.changedetection.utils_func.mcd_utils import accuracy, SCDD_eval_all, AverageMeter

from tqdm import tqdm
import MambaCD.changedetection.datasets.imutils as imutils
import imageio

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

        #self.optim = optim.AdamW(self.deep_model.parameters(),
        #                         lr=args.learning_rate,
        #                         weight_decay=args.weight_decay)



        #self.scheduler = StepLR(self.optim, step_size=10000, gamma=0.5)
        self.deep_model.eval()
        
        
    def validation(self):
        dir = '/notebooks/hi_respect/test/'
        ids = os.listdir(os.path.join(dir, 'T2'))
        #ids = ids[:10]
        #print(ids)       

        torch.cuda.empty_cache()
        acc_meter = AverageMeter()

        preds_all = []
        labels_all = []
        
        for _, id in enumerate(tqdm(ids)):
            pre_change_imgs = np.array(imageio.imread(os.path.join(dir, 'T1', id)), np.float32)
            post_change_imgs = np.array(imageio.imread(os.path.join(dir, 'T2', id)), np.float32)
            label_cd = np.array(imageio.imread(os.path.join(dir, 'GT_CD', id)), np.float32)
            label_clf_t1 = np.array(imageio.imread(os.path.join(dir, 'GT_T1', id)), np.float32)
            label_clf_t2 = np.array(imageio.imread(os.path.join(dir, 'GT_T2', id)), np.float32)
            
            label_cd = label_cd / 255
            

            pre_change_imgs = imutils.normalize_img(pre_change_imgs)  # imagenet normalization
            pre_change_imgs = np.transpose(pre_change_imgs, (2, 0, 1))

            post_change_imgs = imutils.normalize_img(post_change_imgs)  # imagenet normalization
            post_change_imgs = np.transpose(post_change_imgs, (2, 0, 1))

            pre_change_imgs = torch.from_numpy(pre_change_imgs).unsqueeze(0)
            post_change_imgs = torch.from_numpy(post_change_imgs).unsqueeze(0)
            label_cd, label_clf_t1, label_clf_t2 = torch.from_numpy(label_cd).unsqueeze(0), torch.from_numpy(label_clf_t1).unsqueeze(0), torch.from_numpy(label_clf_t2).unsqueeze(0)
            
            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            label_cd = label_cd .cuda().long()
            label_clf_t1 = label_clf_t1.cuda().long()
            label_clf_t2 = label_clf_t2.cuda().long()

            
        
#        print('---------starting evaluation-----------')
#        dataset = SemanticChangeDetectionDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
#        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)

        # vbar = tqdm(val_data_loader, ncols=50)
#        print('ok')
#        for itera, data in enumerate(tqdm(val_data_loader)):
#            pre_change_imgs, post_change_imgs, label_cd, label_clf_t1, label_clf_t2, _ = data

#            pre_change_imgs = pre_change_imgs.cuda()
#            post_change_imgs = post_change_imgs.cuda()
#            label_cd = label_cd.cuda().long()
#            label_clf_t1 = label_clf_t1.cuda().long()
#            label_clf_t2 = label_clf_t2.cuda().long()


            # input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
            output_1, output_semantic_t1, output_semantic_t2 = self.deep_model(pre_change_imgs, post_change_imgs)
            #print('outputs', output_1.shape, output_semantic_t1.shape, output_semantic_t2.shape)

            label_cd = label_cd.cpu().numpy()
            labels_A = label_clf_t1.cpu().numpy()
            labels_B = label_clf_t2.cpu().numpy()

            change_mask = torch.argmax(output_1, axis=1)

            preds_A = torch.argmax(output_semantic_t1, dim=1)
            preds_B = torch.argmax(output_semantic_t2, dim=1)
            preds_A = (preds_A*change_mask.squeeze().long()).cpu().numpy()
            preds_B = (preds_B*change_mask.squeeze().long()).cpu().numpy()

            
            change_mask = change_mask.cpu().numpy()
            
            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                preds_all.append(pred_A)
                preds_all.append(pred_B)
                labels_all.append(label_A)
                labels_all.append(label_B)
                acc = (acc_A + acc_B)*0.5
                acc_meter.update(acc)
#        print('aaaa', preds_all[0].shape, labels_all[0].shape)


            change_mask = np.array(change_mask[0]*255, dtype=np.uint8)
            change_mask = Image.fromarray(change_mask)
            change_mask.save('./CHECKS/{}'.format(id))

        kappa_n0, Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, 7)
        print(f'Kappa coefficient rate is {kappa_n0}, F1 is {Fscd}, OA is {acc_meter.avg}, '
              f'mIoU is {IoU_mean}, SeK is {Sek}')
        
        return kappa_n0, Fscd, IoU_mean, Sek, acc_meter.avg


def main():
    parser = argparse.ArgumentParser(description="Training on SECOND dataset")
    parser.add_argument('--cfg', type=str, default='/notebooks/MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml')
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

    parser.add_argument('--resume', type=str, default='/notebooks/MambaCD/changedetection/saved_models/hi_respect/MambaSCD_Small_1717684721.817448_____/2800_model.pth')  ##petuxesa

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    args = parser.parse_args()
#    with open(args.train_data_list_path, "r") as f:
#        # data_name_list = f.read()
#        data_name_list = [data_name.strip() for data_name in f]
#    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
#    trainer.training()
    trainer.validation()

if __name__ == "__main__":
    main()
