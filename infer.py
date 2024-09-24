import sys
import json
import os
import os.path as osp
from datetime import datetime
import argparse
import numpy as np
import torch
import time
from tqdm import trange
import nibabel as nib
from monai.inferers import sliding_window_inference
from utils.model_factory import get_model
from utils.loss_factory import get_loss
from utils.metric_factory import fast_multiclass_dice
import monai.transforms as t
import monai.data as d
from monai.data import decollate_batch
import logging
import pandas as pd

def str2bool(v):
    # as seen here: https://stackoverflow.com/a/43357954/3208255
    if isinstance(v, bool):
       return v
    if v.lower() in ('true','yes'):
        return True
    elif v.lower() in ('false','no'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--csv_path_ts', type=str, default='/data/PANORAMA/cvillaseca/NETWORKS_SEGMENTACIO/BASELINE_SEGMENTACIO/data/infer.csv', help='CSV file path for test data')
parser.add_argument('--model_name', type=str, default='tiny_swin_unetr', help='Model architecture name')
parser.add_argument('--n_classes', type=int, default=5, help='Number of categories to segment')
parser.add_argument('--patch_size', type=str, default='96/96/96', help='Patch size for sliding window inference')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the DataLoader')
parser.add_argument('--spacing', type=str, default='1.5/1.5/1.5', help='Voxel size')
parser.add_argument('--loss1', type=str, default='ce', choices=('ce', 'dice'), help='First loss function')
parser.add_argument('--loss2', type=str, default='dice', choices=('ce', 'dice'), help='Second loss function')
parser.add_argument('--alpha1', type=float, default=1.0, help='Multiplier for the first loss function')
parser.add_argument('--alpha2', type=float, default=1.0, help='Multiplier for the second loss function')
parser.add_argument('--load_weights', type=str, default=None, help='use pretrained weights if available')
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='use pretrained weights if available')
parser.add_argument('--save_path', type=str, default='test', help='Path to save test results')

def correct_label(l):
    # https://grand-challenge.org/forums/forum/panorama-pancreatic-cancer-diagnosis-radiologists-meet-ai-711/topic/label-problem-2275/
    l[l == 1] = 0
    l[l==2] = 0 
    l[l==3] = 0 
    l[l==4] = 1
    l[l==5] = 0 
    l[l==6] = 0 
    return l

def get_transforms_ct(spacing):
    wl, ww = -40, 400  # Ventana abdominal típica para imágenes CT
    clamp1 = t.ThresholdIntensityd(keys=('scan',), above=False, threshold=wl + (ww / 2), cval=wl + (ww / 2))
    clamp2 = t.ThresholdIntensityd(keys=('scan',), above=True, threshold=wl - (ww / 2), cval=wl - (ww / 2))
    norm = t.NormalizeIntensityd(keys=('scan',), nonzero=True)
    intensities = t.Compose([clamp1, clamp2, norm])

    space = t.Spacingd(keys=('scan', 'label'), pixdim=spacing, mode=('bilinear', 'nearest'))



    ts_transforms = t.Compose([
        t.LoadImaged(keys=('scan', 'label'), ensure_channel_first=True, image_only=True),
        t.Lambda(lambda d: {'scan': d['scan'], 'label': correct_label(d['label'])}),
        intensities,
        space,
        t.AsDiscreteD(keys=('label'), to_onehot=2),
    ])

    return ts_transforms

from torch.utils.data import DataLoader

def collate_fn(batch):
    scans = [item['scan'] for item in batch]
    labels = [item['label'] for item in batch]
    scans = torch.stack(scans)
    labels = torch.stack(labels)
    return scans, labels

def get_loader_as_tuple(csv_path_ts, spacing=(1.5,1.5,1.5), num_workers=8):
    df_ts = pd.read_csv(csv_path_ts, index_col=None)
    ts_files = df_ts.to_dict('records')

    ts_transforms = get_transforms_ct(spacing)
    batch_size = 1
    gpu = torch.cuda.is_available()

    ts_ds = d.Dataset(data=ts_files, transform=ts_transforms)
    ts_loader = DataLoader(
        ts_ds, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False, 
        pin_memory=gpu,
        collate_fn=collate_fn
    )
    
    label_list = ('Backg.', 'Spleen', 'Kidneys', 'Liver', 'Pancreas')
    ts_loader.dataset.label_list = label_list

    return ts_loader


def test(model, loader, loss_fn, log_file, slwin_bs=1):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    patch_size = model.patch_size
    all_dscs, losses = [], []

    with trange(len(loader)) as t:
        n_elems, running_dsc = 0, 0
        for test_data in loader:
            # Unpack the tuple directly
            test_images, test_labels = test_data  # Now test_images and test_labels are the scans and labels
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)  # Make sure both are moved to the correct device
            
            print('input images:', test_images.shape) 
            print('input labels:', test_labels.shape) 
            
            # Perform inference
            test_outputs = sliding_window_inference(test_images, patch_size, slwin_bs, model)
            print('test outputs:', test_outputs.shape)  
            print('test_outputs_voxel', test_outputs[0, :, 50, 50, 50]) 

            # Obtain the prediction
            pancreas_pred = (test_outputs.argmax(dim=1) == 4).float()
            pancreas_label = test_labels.float()  

            # Calculate the loss
            loss = loss_fn(pancreas_pred.unsqueeze(1), pancreas_label.unsqueeze(1)).item()

            # Calculate the Dice Score Coefficient (DSC) for the pancreas class
            dsc_score = fast_multiclass_dice(pancreas_label.cpu().numpy(), pancreas_pred.cpu().numpy(), n_classes=2)[1]

            # Log and display the metrics
            log_msg = f'Lote {n_elems + 1}/{len(loader)}, Loss: {loss:.4f}, Pancreas DSC: {dsc_score * 100:.2f}%'
            print(log_msg)
            logging.info(log_msg)

            t.set_postfix(DSC="{:.2f}".format(dsc_score * 100))
            t.update()

            all_dscs.append(dsc_score)
            losses.append(loss)
            n_elems += 1
            running_dsc += dsc_score
            run_dsc = running_dsc / n_elems

    return np.mean(all_dscs), np.mean(losses)


if __name__ == '__main__':
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    save_path = osp.join('test_results', datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    os.makedirs(save_path, exist_ok=True)
    log_file = osp.join(save_path, 'test_log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s %(message)s')

    patch_size = tuple(map(int, args.patch_size.split('/')))
    spacing = tuple(map(float, args.spacing.split('/')))

    model = get_model(args.model_name, n_classes=args.n_classes, pretrained=args.pretrained, patch_size=patch_size)
    model = model.to(device)

    ts_loader = get_loader_as_tuple(args.csv_path_ts, spacing)

    loss_fn = get_loss(args.loss1, args.loss2, args.alpha1, args.alpha2)

    print(f'* Instantiating loss function: {args.alpha1} * {args.loss1} + {args.alpha2} * {args.loss2}')
    print('* Starting test\n', '-' * 10)

    start = time.time()
    avg_dsc, avg_loss = test(model, ts_loader, loss_fn, log_file, slwin_bs=1)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'Test time: {int(hours):02}h {int(minutes):02}min {seconds:.2f}secs')
    print(f'Average DSC: {avg_dsc * 100:.2f}%, Average Loss: {avg_loss:.4f}')

    print('Finished.')
