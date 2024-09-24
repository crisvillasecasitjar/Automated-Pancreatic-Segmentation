import torch
import monai.transforms as t
import monai.data as d
from monai.data import decollate_batch
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import logging

def check(t):
    print(t.shape)
    print(t.unique())
    sys.exit('www')
    return t



def get_transforms_ct(patch_size, spacing, n_classes, n_samples, neg_samples):
    wl, ww = -40, 400  # https://www.stepwards.com/?page_id=21646#ABDOMINAL_WINDOW
    clamp1 = t.ThresholdIntensityd(keys=('scan',), above=False, threshold=wl + (ww / 2), cval=wl + (ww / 2))
    clamp2 = t.ThresholdIntensityd(keys=('scan',), above=True, threshold=wl - (ww / 2), cval=wl - (ww / 2))
    norm = t.NormalizeIntensityd(keys=('scan',), nonzero=True)
    intensities = t.Compose([clamp1, clamp2, norm])

    space = t.Spacingd(keys=('scan', 'label'), pixdim=spacing, mode=('bilinear', 'nearest'))

    p_app, pr_geom = 0.1, 0.1

    tr_transforms = t.Compose([
        t.LoadImaged(keys=('scan', 'label'), ensure_channel_first=True, image_only=True),
        intensities,
        space,
        t.RandCropByPosNegLabeld(keys=('scan', 'label'), label_key='label', spatial_size=patch_size,
                                 num_samples=n_samples, pos=1, neg=neg_samples),
        t.RandScaleIntensityd(keys=('scan', ), factors=0.05, prob=p_app),
        t.RandShiftIntensityd(keys=('scan', ), offsets=0.05, prob=p_app),
        t.RandFlipd(keys=('scan', 'label'), prob=pr_geom, spatial_axis=0),
        t.RandFlipd(keys=('scan', 'label'), prob=pr_geom, spatial_axis=1),
        t.RandFlipd(keys=('scan', 'label'), prob=pr_geom, spatial_axis=2),
        t.RandRotate90d(keys=('scan', 'label'), prob=pr_geom, max_k=3),
        t.AsDiscreted(keys=('label',), to_onehot=n_classes),
    ])

    vl_transforms = t.Compose([
        t.LoadImaged(keys=('scan', 'label'), ensure_channel_first=True, image_only=True),
        intensities,
        space,
        t.AsDiscreteD(keys=('label'), to_onehot=n_classes),
        #t.Lambda(lambda d: print_unique_classes(d, "LoadImaged")),
    ])

    return tr_transforms, vl_transforms

def get_loaders(csv_path_tr, patch_size, spacing=(1.5,1.5,1.5), n_samples=1, neg_samples=1, n_classes=5, cache=0., num_workers=0, tr_percentage=1., ovft_check=-1):
    df_tr = pd.read_csv(csv_path_tr, index_col=None)
    df_vl = pd.read_csv(csv_path_tr.replace('tr', 'vl'), index_col=None)
    if ovft_check == -1: df_ovft = df_tr
    else: df_ovft = df_tr.sample(n=ovft_check)

    #df_tr['scan'] = df_tr['scan'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8') #faig tot això perquè hi ha problemes amb el format dels strings i hi ha alguns que deuen tenir caràcters amagats i no me'ls llegeix bé. Bàsicament estic normalitzant els paths per evitar aquests problemes
    #df_tr['scan'] = df_tr['scan'].str.strip()
    #df_vl['scan'] = df_vl['scan'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    #df_vl['scan'] = df_vl['scan'].str.strip()
    #df_ovft['scan'] = df_ovft['scan'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    #df_ovft['scan'] = df_ovft['scan'].str.strip()

    tr_files, vl_files, ovt_files = df_tr.to_dict('records'), df_vl.to_dict('records'), df_ovft.to_dict('records')

    if tr_percentage < 1.:
        print(60*'-')
        n_tr_examples = len(tr_files)
        random_indexes = np.random.permutation(n_tr_examples)
        kept_indexes = int(n_tr_examples * tr_percentage)
        tr_files = [tr_files[i] for i in random_indexes[:kept_indexes]]

        n_vl_examples = len(vl_files)
        random_indexes = np.random.permutation(n_vl_examples)
        kept_indexes = int(n_vl_examples * tr_percentage)
        vl_files = [vl_files[i] for i in random_indexes[:kept_indexes]]
        print('Reducing training/validation data from {}/{} items to {}/{}'.format(n_tr_examples, len(tr_files),
                                                                                   n_vl_examples, len(vl_files)))
        print(60 * '-')

    tr_transforms, vl_transforms = get_transforms_ct(patch_size, spacing, n_classes, n_samples, neg_samples)
    batch_size = 1
    test_batch_size = 1
    gpu = torch.cuda.is_available()

    if cache>0.:
        tr_ds = d.CacheDataset(data=tr_files, transform=tr_transforms, cache_rate=cache, num_workers=8, )
        vl_ds = d.CacheDataset(data=vl_files, transform=vl_transforms, cache_rate=cache, num_workers=8, )
        if ovft_check > 0:
            ovft_ds = d.CacheDataset(data=ovt_files[:ovft_check], transform=vl_transforms, cache_rate=cache, )
        else:
            ovft_ds = d.CacheDataset(data=ovt_files, transform=vl_transforms, cache_rate=cache, )

        tr_loader = d.ThreadDataLoader(tr_ds, num_workers=0, batch_size=batch_size, shuffle=True)
        vl_loader = d.ThreadDataLoader(vl_ds, num_workers=0, batch_size=test_batch_size)
        ovft_loader = d.ThreadDataLoader(ovft_ds, num_workers=0, batch_size=test_batch_size)

    else:
        tr_ds = d.Dataset(data=tr_files, transform=tr_transforms)
        vl_ds = d.Dataset(data=vl_files, transform=vl_transforms)

        tr_loader = d.DataLoader(tr_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=gpu)
        vl_loader = d.DataLoader(vl_ds, batch_size=test_batch_size, num_workers=num_workers, pin_memory=gpu)
        if ovft_check > 0: ovft_ds = d.Dataset(data=ovt_files[:ovft_check], transform=vl_transforms)
        else: ovft_ds = d.Dataset(data=ovt_files, transform=vl_transforms)
        ovft_loader = d.DataLoader(ovft_ds, batch_size=test_batch_size, num_workers=num_workers, pin_memory=gpu)

    #label_list = ('Backg.', 'PDAC Lesion', 'Veins', 'Arteries', 'Pancreas', 'Pancreas Duct', 'Bile Duct')
    label_list = ('Backg.', 'Spleen', 'Kidneys', 'Liver', 'Pancreas')
    #label_list = ('Backg.', 'Spleen', 'r.kidney', 'l.kidney', 'gall bladder', 'esophagus', 'liver', 'stomach', 'arota', 'postcava', 'pancreas', 'r. adrenal gland', 'l.adrenal gland', 'duodenum', 'bladder', )
    tr_loader.dataset.label_list = label_list

    return tr_loader, ovft_loader, vl_loader
