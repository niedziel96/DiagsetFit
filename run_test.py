import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor
from ray import train, tune
from ray.tune.search.optuna import OptunaSearch
from timm.data import create_transform
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json 
from yacs.config import CfgNode as CN
import matplotlib.pyplot as plt

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

def pt_loader(path):
    sample = torch.load(path)
    sample = sample[0].unfold(1, 16, 16).transpose(1,2)
    return sample 

def parse_option():
    parser = argparse.ArgumentParser('SVIT test script', add_help=False)
   
    parser.add_argument('--model_path', type=str, help='path to model which will be tested')
    parser.add_argument('--model_type', type=str, help='path to model which will be tested')
    parser.add_argument('--result_directory', type=str, help='results save directory')
    parser.add_argument('--test_dataset_path', type=str, help='path to test data')
    parser.add_argument('--experimental', default='n', type=str, help='path to test data')
    parser.add_argument('--cfg', type=str, default=None, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args, unparsed = parser.parse_known_args()

    return args
	
def main(args, config = None): 
    # logger = create_logger(output_dir=args.result_directory, dist_rank=0, name="LOGS_TEST")
    # add transform  
    if args.experimental == 'n':
        b_size = 8
        n_workers = 8 
        p_mem = True 
        t = []
        size = int((256 / 224) * 224) # default img size 
        t.append(transforms.Resize(size, interpolation=_pil_interp('bicubic')),)
        t.append(transforms.CenterCrop(224))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        transform = transforms.Compose(t)
        dataset = datasets.ImageFolder(args.test_dataset_path, transform=transform)
    
    else:
        b_size = 1 
        n_workers = 0
        p_mem = False
        dataset = datasets.DatasetFolder(args.test_dataset_path, 
                                         loader = pt_loader,
                                         extensions=['.pt'])
        
    
    sampler_test = torch.utils.data.SequentialSampler(dataset)

    data_loader_test = torch.utils.data.DataLoader(
    dataset, sampler=sampler_test,
    batch_size=b_size,
    shuffle=False,
    num_workers=n_workers,
    pin_memory=p_mem,
    drop_last=False
    )
	
    if config is None:
    # actual test
        config = CN()
        config.FUSED_LAYERNORM = False
        config.MODEL = CN()
        config.MODEL.VIT = CN()
        config.MODEL.VIT.NUM_CLASSES = 8
        config.MODEL.TYPE = args.model_type



    model = build_model(config)
    print(args.model_path)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    model.eval()

    batch_time = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()

    # default values: 
    pred_all = []
    targ_all = []
    y_pred = []
    y_true = []
    test_loss = 0
    # confusion_matrix = torch.zeros(8, 8)
    correct = 0
    test_running_corrects = 0.0
    test_total = 0.0
    counter = 0
    print('Starting test...')
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader_test):
            counter += 1
            if counter % 20 == 0: 
                print(f'processed {counter} out of {len(data_loader_test)}')
            print('Start epoch..')
            if args.experimental == 'y':
                data = data[0].cuda(non_blocking=True)
                target = target[0].cuda(non_blocking=True)
            else: 
                data = data.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            output = model(data)
            
            if args.experimental == 'y':
                logits = output[0]
                _, pred = torch.topk(logits.cuda(), 1, 0,True, True)
                if pred[0] == target:
                    acc1 = 100.0
                    acc5 = 100.0
                else: 
                    acc1 = 0.0
                    acc5 = 0.0
                acc1_meter.update(acc1, 1)
                acc5_meter.update(acc5, 1)

            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                acc1_meter.update(acc1.item(), target.size(0))
                acc5_meter.update(acc5.item(), target.size(0))

            # logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f} for epoch {idx}')
            if args.experimental == 'y':
                tmp = pred[0].data.cpu().numpy()
                output_tmp = tmp#(torch.max(torch.exp(pred[0]), 1)[1]).data.cpu().numpy()
                val = output_tmp.tolist()
                y_pred.extend([val]) # Save Prediction

                labels_tmp = target.data.cpu().numpy()
                val2 = labels_tmp.tolist()
                y_true.extend([val2]) # Save Truth
            
            else: 
                output_tmp = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                y_pred.extend(output_tmp) # Save Prediction

                labels_tmp = target.data.cpu().numpy()
                y_true.extend(labels_tmp) # Save Truth

            batch_time.update(time.time() - end)
            end = time.time()

            if args.experimental == 'y':
                # pre = logits.argmax(1, keepdim=True) 
                pred_all.append(pred[0])
                targ_all.append(target)
                correct += pred.eq(target.view_as(pred)).sum().item()
                out = pred[0] 
                ts = 1 
            else: 
                pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
                pred_all.append(pred)
                targ_all.append(target)
                correct += pred.eq(target.view_as(pred)).sum().item()
                out_1 = output
                ts = target.size(0) 

                _, out = torch.max(out_1, 1)
            

            test_total += ts 
            test_running_corrects += (out == target).sum().item()

            # for t, p in zip(target.view(-1), output.view(-1)):
            #     confusion_matrix[t.long(), p.long()] += 1
	
    test_loss /= len(data_loader_test.dataset)
    # logger.info(f' * Final AVG: Acc@1 {acc1_meter.avg:.3f} AVG Acc@5 {acc5_meter.avg:.3f}.')

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(data_loader_test.dataset), 100. * correct / len(data_loader_test.dataset)))

    # save results: 
    cmap = sn.cm.rocket_r
    print(y_true)
    print(y_pred)

    # get classes
    idx2class = {v: k for k, v in data_loader_test.dataset.class_to_idx.items()}
    print(idx2class)
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in ['A','DC','F','LC','MC','PC','PT','TA']],
                         columns = [i for i in  ['A','DC','F','LC','MC','PC','PT','TA']]) # temp
    plt.figure(figsize = (12,7))
    ax = sn.heatmap(df_cm*100, annot=True, cmap=cmap, vmin=0, vmax=100, fmt='.2f')
    ax.figure.axes[0].set_ylabel('Ground truth', size=11)
    ax.figure.axes[0].set_xlabel('Predictions', size=11)

    plt.savefig(f'{args.result_directory}/class_heatmap.png')

    acc = accuracy_score(y_true, y_pred)
    ratios = get_cl_ratio(cf_matrix)
    AvAcc = sum(ratios.values())/8

    detailed_results = {}
    detailed_results['acc'] = acc
    detailed_results['AvAcc'] = AvAcc

    with open(f'{args.result_directory}/results.txt', 'w') as convert_file: 
        convert_file.write(json.dumps(detailed_results))
    
def get_cl_ratio(cm):
    cl_ratio = {}
    for i in range(len(cm)):
        cl_ratio[i] = cm[i][i]/sum(cm[i])

    return cl_ratio

			
if __name__ == '__main__':
    args = parse_option()

    root_mp = args.model_path
    root_out = args.result_directory 
    root_data_path = args.test_dataset_path

    if args.cfg:
        config = get_config(args)
        c_state = 'value'
    else:
        c_state = 'placeholder'

    if 'ensemble' in args.model_type:
        org_ch1 = config.MODEL.ENSEMBLE.CH_1
        org_ch2 = config.MODEL.ENSEMBLE.CH_2
        org_ch3 = config.MODEL.ENSEMBLE.CH_3


    if args.experimental == 'n':
        for f in ['fold_1','fold_2', 'fold_3', 'fold_4', 'fold_5']:
            for r in ['40', '100', '200', '400']:

                print(f'calculating for {f} and {r}')

                r_ = f'{r}x'
                res_dir = os.path.join(root_out, f, r_)
                # make suer result dir exists: 
                if not os.path.isdir(res_dir):
                    os.makedirs(res_dir)

                if 'ensemble' in args.model_type: 
                    config.defrost()
                    print(' ENSEMBLE MODE')
                    fld = f'{f}_x{r}'
                    config.MODEL.ENSEMBLE.CH_1 = os.path.join(org_ch1, fld, 'best_model.pth')
                    config.MODEL.ENSEMBLE.CH_2 = os.path.join(org_ch2, fld, 'best_model.pth')
                    config.MODEL.ENSEMBLE.CH_3 = os.path.join(org_ch3, fld, 'best_model.pth')
                    config.freeze()
                    print(config.MODEL.ENSEMBLE.CH_1)
                
                # edit args oon fly 
                nme = f'{f}_x{r}' 
                mne_f = nme #[:-1]
                actual_model = os.path.join(root_mp, mne_f, 'best_model.pth')
                print(root_data_path)
                print(root_data_path[0])
                new_path = os.path.join(root_data_path, f, 'all_class', r_, 'test')
                print(new_path)
                args.result_directory = res_dir
                args.test_dataset_path = new_path
                args.model_path = actual_model
                if 'placeholder' in c_state:
                    main(args)
                else: 
                    main(args, config)
                print('Finished!')
    else: 

        if not os.path.isdir(root_out):
            os.makedirs(root_out)

        actual_model = os.path.join(root_mp, 'best_model.pth')
        new_path = os.path.join(root_data_path[0],'test')
        print(new_path)
        args.test_dataset_path = new_path
        args.model_path = actual_model
        if 'placeholder' in c_state:
            main(args)
        else: 
            main(args, config)
        print('Finished!')