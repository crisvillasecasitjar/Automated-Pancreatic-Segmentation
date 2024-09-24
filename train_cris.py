import sys, json, os, time, argparse
import os.path as osp
from datetime import datetime
import operator
from tqdm import trange
import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from utils.model_factory import get_model
from utils.loss_factory import get_loss
from utils.reproducibility import set_seeds
from utils.metric_factory import fast_multiclass_dice, fast_bin_auc, binary_ECE
from utils.data_load_cris import get_loaders


import logging
import csv


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

# argument parsing csv_path_tr, csv_path_test
parser = argparse.ArgumentParser()
parser.add_argument('--csv_path_tr', type=str, default='/data/PANORAMA/cvillaseca/NETWORKS_SEGMENTACIO/BASELINE_SEGMENTACIO/data/tr_AMOS.csv', help='csv for training data')
parser.add_argument('--cache', type=float, default=0., help='percentage of precomputed and cached data for loading')
parser.add_argument('--tr_percentage', type=float, default=0.1, help='amount of training data to use - for debugging')
parser.add_argument('--ovft_check', type=int, default=4, help='# of training samples used for monitoring overfitting, -1=all of it')
parser.add_argument('--num_workers', type=int, default=8, help='number of parallel (multiprocessing) workers to launch - to be changed')
parser.add_argument('--compile', type=str2bool, nargs='?', const=True, default=False, help='use torch.compile (experimental)')

parser.add_argument('--model_name', type=str, default='tiny_swin_unetr', help='architecture')
parser.add_argument('--n_classes', type=int, default=5, help=' categories to segment')
parser.add_argument('--load_weights', type=str, default=None, help='use pretrained weights if available')
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='use pretrained weights if available')

parser.add_argument('--batch_size', type=int, default=4, help=' batch size')
parser.add_argument('--acc_grad', type=int, default=1, help='gradient accumulation')
parser.add_argument('--patch_size', type=str, default='96/96/96', help='volumetric patch size')
parser.add_argument('--n_samples', type=int, default=8, help='nr of patches extracted per loaded volume before moving to the next one')
parser.add_argument('--neg_samples', type=int, default=1, help='out of n patches, 1/(1+neg_samples) are foreground-centered')
parser.add_argument('--spacing', type=str, default='1.5/1.5/1.5', help='voxel size')

parser.add_argument('--loss1', type=str, default='ce',   choices=('ce', 'dice'), help='1st loss')
parser.add_argument('--loss2', type=str, default='dice', choices=('ce', 'dice'), help='2nd loss')
parser.add_argument('--alpha1', type=float, default=1., help='multiplier in alpha1*loss1+alpha2*loss2')
parser.add_argument('--alpha2', type=float, default=1., help='multiplier in alpha1*loss1+alpha2*loss2')

parser.add_argument('--optimizer', type=str, default='nadam', choices=('sgd', 'adamw', 'nadam'), help='optimizer choice')
parser.add_argument('--lr', type=float, default=1e-3, help='max learning rate')
parser.add_argument('--n_epochs', type=int, default=2, help='training epochs')  # Reduced epochs
parser.add_argument('--vl_interval', type=int, default=1, help='how often we check performance and maybe save')
parser.add_argument('--cyclical_lr', type=str2bool, nargs='?', const=True, default=False, help='re-start lr each vl_interval epochs')
parser.add_argument('--metric', type=str, default='avgDSC', help='which metric to use for monitoring progress (multiclass DSC)')
parser.add_argument('--save_path', type=str, default='delete', help='path to save model (defaults to delete)')


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def validate(model, loader, loss_fn, slwin_bs=2):
    model.eval()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    patch_size = model.patch_size
    all_dscs, losses = [], []
    with trange(len(loader)) as t:
        n_elems, running_dsc = 0, 0
        for val_data in loader:
            val_images, val_labels = val_data['scan'].to(device), val_data['label']
            print('input images:', val_images.shape) #it should be (1 (batch_size),1 (black and white),96,96,96)
            print('input labels:', val_labels.shape) #it should be (1,7(classes), 96,96,96) --> this is in one hot encoding, which means that the second position of the tensor contains a 7-dimension vector with the probabilities of each class for that voxel. The second dimension is the number of classes, which is 7. This dimension contains the one-hot encoded vectors for each voxel.
            val_outputs = sliding_window_inference(val_images.to(device), patch_size, slwin_bs, model, overlap=0.1).cpu()
            print('val outputs:', val_outputs.shape) #Since we have the output as one-hot encoded, it should be (1,7,96,96,96), because in the second position (7) it is representing the model's confidence in that voxel belonging to one of the 7 classes. 
            print('val_outputs_voxel', val_outputs[0, :, 50, 50, 50]) #this should give a vector of 7 values, each consisting on the probability of the voxel in position (50,50,50) pertaining to each of the classes and the result of their sum is 1

            loss = loss_fn(val_outputs, val_labels)

            #argmax(dim=1) converts this one-hot encoding to class labels by taking the index of the maximum value along the class dimension.
            val_pred = val_outputs.argmax(dim=1).cpu().numpy()  # Takes the maximum value in the second position of the predicted output (the one that is a vector with the probabilitie sof the voxel pertaining to each class). Basically, we are converting from one-hot to class labels. Move tensor to CPU before converting to numpy. It will respond with a 
            print('val_pred:', val_pred.shape)
            print('val_pred_voxel', val_pred[:, 50, 50, 50])

            val_seg = val_labels.argmax(dim=1).cpu().numpy()  # Move tensor to CPU before converting to numpy
            print('val_seg:', val_seg.shape) #it is the ground truth, but since we did the argmax, instead of having it as one hot encoded, we have it as class labels, in order to be able to compare it with the model's predictions.
            print('val_seg_voxel', val_seg[:, 50, 50, 50])
            dsc_scores = fast_multiclass_dice(val_seg, val_pred, n_classes=5)
            print(dsc_scores)

            all_dscs.append(dsc_scores)
            losses.append(loss.item())
            n_elems += 1
            running_dsc += np.mean(dsc_scores[1:]) # not the background
            run_dsc = running_dsc / n_elems
            t.set_postfix(DSC="{:.2f}".format(100 * run_dsc))
            t.update()
    all_dscs = np.mean(all_dscs, axis=0)
    return [*all_dscs, np.mean(np.array(losses))]

def train_one_epoch(model, tr_loader, bs, acc_grad, loss_fn, optimizer, scheduler):
    model.train()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    n_opt_iters = 0
    with trange(len(tr_loader)) as t:
        step, n_elems, running_loss = 0, 0, 0
        for batch_data in tr_loader:  # load 1 scan from the training set
            n_samples = len(batch_data['label'])  # nr of px x py x pz patches (see args.n_samples)
            for m in range(0, n_samples, bs):  # we loop over batch_data picking up bs patches at a time
                step += bs
                inputs, labels = (batch_data['scan'][m:(m+bs)].to(device), batch_data['label'][m:(m+bs)].to(device))
                outputs = model(inputs)
                print('inputs: '+str(inputs.shape))
                print('outputs: '+str(outputs.shape))
                loss = loss_fn(outputs, labels)
                loss = loss / acc_grad
                loss.backward()
                if ((n_opt_iters + 1) % acc_grad == 0) or (n_opt_iters + 1 == len(tr_loader)):
                    # Update Optimizer
                    optimizer.step()
                    optimizer.zero_grad()
                n_opt_iters += 1
                lr = get_lr(optimizer)
                scheduler.step()
                optimizer.zero_grad()
                running_loss += loss.detach().item() * inputs.shape[0]
                n_elems += inputs.shape[0]  # total nr of items processed
                run_loss = running_loss / n_elems

            t.set_postfix(LOSS_lr="{:.4f}/{:.6f}".format(run_loss, lr))
            t.update()

def set_tr_info(tr_info, epoch=0, ovft_metrics=None, vl_metrics=None, best_epoch=False):
    # I customize this for each project.
    label_list = tr_info['label_list']
    if best_epoch:
        for l in label_list:
            tr_info['best_tr_dsc_'+l] = tr_info['tr_dscs_'+l][-1]
            tr_info['best_vl_dsc_' + l] = tr_info['vl_dscs_' + l][-1]
        tr_info['best_tr_avg_dsc'] = tr_info['tr_avg_dscs'][-1]
        tr_info['best_vl_avg_dsc'] = tr_info['vl_avg_dscs'][-1]
        tr_info['best_tr_loss'] = tr_info['tr_losses'][-1]
        tr_info['best_vl_loss'] = tr_info['vl_losses'][-1]
        tr_info['best_epoch'] = epoch
    else:
        for i in range(len(label_list)):
            l = label_list[i]
            tr_info['tr_dscs_' + l].append(ovft_metrics[i])
            tr_info['vl_dscs_' + l].append(vl_metrics[i])
        tr_info['tr_avg_dscs'].append(np.mean(ovft_metrics[1:-1]))  # exclude background and losses
        tr_info['vl_avg_dscs'].append(np.mean(vl_metrics[1:-1]))    # exclude background and losses
        tr_info['tr_losses'].append(ovft_metrics[-1])
        tr_info['vl_losses'].append(vl_metrics[-1])

    return tr_info

def init_tr_info(label_list):
    # I customize this function for each project.
    tr_info = dict()
    for l in label_list:
        tr_info['tr_dscs_'+l], tr_info['vl_dscs_'+l] = [], []
    tr_info['tr_avg_dscs'], tr_info['vl_avg_dscs'] = [], []
    tr_info['tr_losses'], tr_info['vl_losses'] = [], []
    tr_info['label_list'] = label_list
    return tr_info

def get_eval_string(tr_info, epoch, finished=False, vl_interval=1):
    # I customize this function for each project.
    # Pretty prints values of train/val metrics to a string and returns it
    # Used also by the end of training (finished=True)
    ep_idx = len(tr_info['tr_losses'])-1
    if finished:
        ep_idx = epoch
        epoch = (epoch+1) * vl_interval - 1
    s = 'Ep. {}: Train/Val '.format(str(epoch+1).zfill(3))
    for l in tr_info['label_list'][1:]: # skip background in print
        s += '{} DSC: {:.2f}/{:.2f} - '.format(l, tr_info['tr_dscs_'+l][ep_idx], tr_info['vl_dscs_'+l][ep_idx])
    s += 'Loss: {:.4f}||{:.4f}'.format(tr_info['tr_losses'][ep_idx], tr_info['vl_losses'][ep_idx])
    return s


def train_model(model, optimizer, acc_grad, loss_fn, bs, tr_loader, ovft_loader, vl_loader, scheduler, metric, n_epochs, vl_interval, save_path):
    best_metric, best_epoch = -1, 0
    label_list = tr_loader.dataset.label_list
    tr_info = init_tr_info(label_list)
    # Clear GPU memory cache before starting training loop
    torch.cuda.empty_cache()
    for epoch in range(n_epochs):
        print('Epoch {:d}/{:d}'.format(epoch + 1, n_epochs))
        torch.cuda.empty_cache()
        # train one epoch
        train_one_epoch(model, tr_loader, bs, acc_grad, loss_fn, optimizer, scheduler)

        if (epoch + 1) % vl_interval == 0:
            with torch.no_grad():
                ovft_metrics = validate(model, ovft_loader, loss_fn)
                vl_metrics = validate(model, vl_loader, loss_fn)
            tr_info = set_tr_info(tr_info, epoch, ovft_metrics, vl_metrics)
            s = get_eval_string(tr_info, epoch)
            print(s)
            with open(osp.join(save_path, 'train_log.txt'), 'a') as f: print(s, file=f)
            # check if performance was better than anyone before and checkpoint if so
            if metric =='avgDSC': curr_metric = tr_info['vl_avg_dscs'][-1]
            elif metric == 'pancreas_DSC': curr_metric = tr_info['vl_dscs_Pancreas'][-1]

            if curr_metric > best_metric:
                print('-------- Best {} attained. {:.2f} --> {:.2f} --------'.format(metric, best_metric, curr_metric))
                torch.save(model.state_dict(), osp.join(save_path, 'best_model.pth'))
                torch.save(optimizer.state_dict(), os.path.join(save_path, "best_optimizer_state.pth"))
                best_metric, best_epoch = curr_metric, epoch + 1
                tr_info = set_tr_info(tr_info, epoch+1, best_epoch=True)
            else:
                print('-------- Best {} so far {:.2f} at epoch {:d} --------'.format(metric, best_metric, best_epoch))
    #del model, tr_loader, vl_loader
    # maybe this works also? tr_loader.dataset._fill_cache
    #print('-------- Saving last-cycle checkpoint --------')
    #torch.save(model.state_dict(), os.path.join(save_path, "last_model.pth"))
    #torch.save(optimizer.state_dict(), os.path.join(save_path, "last_optimizer_state.pth"))
    
    torch.cuda.empty_cache()  # Clear GPU memory cache at the end of training
    
    return tr_info


if __name__ == '__main__':

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print('Device: '+str(device))
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # logging
    if args.save_path == 'date_time': save_path = osp.join('experiments', datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    else: save_path = osp.join('experiments', args.save_path)
    os.makedirs(save_path, exist_ok=True)
    config_file_path = osp.join(save_path, 'config.cfg')
    with open(config_file_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # gather parser parameters
    model_name = args.model_name
    optimizer_choice = args.optimizer
    lr, bs, ns, negs = args.lr, args.batch_size, args.n_samples, args.neg_samples
    n_epochs, vl_interval, metric = args.n_epochs, args.vl_interval, args.metric
    acc_grad, nw = args.acc_grad, args.num_workers

    patch_size = args.patch_size.split('/')
    patch_size = tuple(map(int, patch_size))
    spacing = args.spacing.split('/')
    spacing = tuple(map(float, spacing))

    print('* Instantiating a {} model'.format(model_name))
    model = get_model(args.model_name, n_classes=args.n_classes, pretrained=args.pretrained, patch_size=patch_size)
    if args.compile:
        model = torch.compile(model)

    #if args.load_weights != None:
        #state_dict = torch.load(args.load_weights)['state_dict']
        #model.load_state_dict(state_dict)
        #print('* weights loaded from {}'.format(args.load_weights))

    print('* Creating Dataloaders, batch size = {}, samples/vol = {}, workers = {}'.format(bs, ns, nw))
    tr_loader, ovft_loader, vl_loader = get_loaders(args.csv_path_tr, patch_size, spacing,
                                                    n_samples=args.n_samples, neg_samples=args.neg_samples,
                                                    n_classes=args.n_classes, cache=args.cache, num_workers=nw,
                                                    tr_percentage=args.tr_percentage, ovft_check=args.ovft_check)

    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_choice == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    elif optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    else:
        sys.exit('please choose between sgd, adam or nadam optimizers')

    if args.cyclical_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=vl_interval*len(tr_loader)*ns//bs, eta_min=0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs*len(tr_loader)*ns//bs, eta_min=0)

    #loss_fn = DiceCELoss() 
    loss_fn= get_loss(args.loss1, args.loss2, args.alpha1, args.alpha2)

    print('* Instantiating loss function {:.2f}*{} + {:.2f}*{}'.format(args.alpha1, args.loss1, args.alpha2, args.loss2))
    print('* Starting to train\n', '-' * 10)
    start = time.time()
    tr_info = train_model(model, optimizer, acc_grad, loss_fn, bs, tr_loader, ovft_loader, vl_loader, scheduler, metric, n_epochs, vl_interval, save_path)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))


    label_list = tr_info['label_list']
    with open(osp.join(save_path, 'log.txt'), 'a') as f:
        s = 'Best epoch = {}/{}: Train/Val '.format(tr_info['best_epoch'], n_epochs)
        for l in tr_info['label_list'][1:]:  # skip background in print
            s += '{} DSC: {:.2f}/{:.2f} - '.format(l, tr_info['best_tr_dsc_' + l], tr_info['best_vl_dsc_' + l])
        s += 'Loss: {:.4f}||{:.4f}'.format(tr_info['best_tr_loss'], tr_info['best_vl_loss'])
        print(s, file=f)
        for j in range(n_epochs//vl_interval):
            s = get_eval_string(tr_info, epoch=j, finished=True, vl_interval=vl_interval)
            print(s, file=f)
        print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)

    print('Done. Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))


    print('Finished.')

