import nibabel as nib
import numpy as np
from totalsegmentator.python_api import totalsegmentator
import os
import torch
import pandas as pd
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ThresholdIntensityd, NormalizeIntensityd, Spacingd, Lambda, AsDiscreted
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import AsDiscreted
import numpy as np
from tqdm import trange
import torch

# Paths and device configuration
data_dir = "/data/PANORAMA/cvillaseca/NETWORKS_SEGMENTACIO/BASELINE_SEGMENTACIO/data"  
csv_file = "/data/PANORAMA/cvillaseca/NETWORKS_SEGMENTACIO/BASELINE_SEGMENTACIO/data/infer.csv"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device= torch.device('cpu')

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file)

# Define test transforms
wl, ww = -40, 400  # For abdominal window
clamp1 = ThresholdIntensityd(keys=('image',), above=False, threshold=wl + (ww / 2), cval=wl + (ww / 2))
clamp2 = ThresholdIntensityd(keys=('image',), above=True, threshold=wl - (ww / 2), cval=wl - (ww / 2))
norm = NormalizeIntensityd(keys=('image',), nonzero=True)
space = Spacingd(keys=('image', 'label'), pixdim=(1.5, 1.5, 1.5), mode=('bilinear', 'nearest'))

def correct_label(l):
    # https://grand-challenge.org/forums/forum/panorama-pancreatic-cancer-diagnosis-radiologists-meet-ai-711/topic/label-problem-2275/
    l[l == 1] = 0
    l[l==2] = 0 
    l[l==3] = 0 
    l[l==4] = 1
    l[l==5] = 0 
    l[l==6] = 0 
    l[l==7] = 0 
    l[l==8] = 0 
    return l

test_org_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    clamp1,
    clamp2,
    norm,
    space,
    Lambda(lambda d: {'image': d['image'], 'label': correct_label(d['label'])}),
    AsDiscreted(keys=('label'), to_onehot=2)

])

# Create a MONAI dataset from the CSV file
class CustomDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

    def __getitem__(self, index):
        data_dict = {
            'image': self.df.iloc[index]['scan'],
            'label': self.df.iloc[index]['label']
        }
        if self.transforms:
            data_dict = self.transforms(data_dict)
        return data_dict

    def __len__(self):
        return len(self.df)

# Initialize the dataset and data loader
test_dataset = CustomDataset(df=df, transforms=test_org_transforms)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def fast_multiclass_dice(actual, predicted, n_classes):
    actual = np.squeeze(np.array(actual))
    predicted = np.squeeze(np.array(predicted))
    print('actual shape:', actual.shape)
    print('actual dtype', actual.dtype)
    print('predicted shape:', predicted.shape)
    print('predicted dtype', predicted.dtype)

    # Initialize an array to store the dice score for each class
    dices = np.zeros(n_classes) 
    for cls in range(n_classes):
        actual_cls = (actual == cls)
        predicted_cls = (predicted == cls)
        actual_cls = np.array(actual_cls).astype(bool)
        predicted_cls = np.array(predicted_cls).astype(bool)
        print('actual_cls shape:', actual_cls.shape)
        print('actual_cls dtype', actual_cls.dtype)
        print('predicted_cls shape:', predicted_cls.shape)
        print('predicted_cls dtype', predicted_cls.dtype)
        
        intersections = np.logical_and(actual_cls, predicted_cls).sum(axis=(0, 1, 2))
        im_sums = actual_cls.sum(axis=(0, 1, 2)) + predicted_cls.sum(axis=(0, 1, 2))
        dices[cls] = 2. * intersections / np.maximum(im_sums, 1e-6)
    return dices

# Define post-processing transforms to get predictions as discrete values
post_transform = Compose([
    AsDiscreted(keys="pred", argmax=True),  # Get class indices from the 5-class predictions
    Lambda(lambda d: {'pred': map_classes_to_binary(d['pred']), 'label': d['label']}),  # Map class 4 to foreground, rest to background
])

def map_classes_to_binary(pred):
    pred = pred.long()

    print(f"Raw prediction shape: {pred.shape}, unique values: {torch.unique(pred)}")

    # Initialize all as background (0)
    binary_pred = torch.zeros_like(pred)

    # Map class 4 to foreground (1)
    binary_pred[pred == 4] = 1

    print(f"Binary prediction shape: {binary_pred.shape}, unique values: {torch.unique(binary_pred)}")

    return binary_pred

log_dir = "/data/PANORAMA/cvillaseca/NETWORKS_SEGMENTACIO/BASELINE_SEGMENTACIO/test_results/Total_segmentator"
log_file = os.path.join(log_dir, "dsc_log_TS.txt")
# Ensure the directory exists
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

all_dsc= []
for i in test_loader:
    test_images, test_labels = i['image'].to(device), i['label']
    nifti_img = nib.Nifti1Image(np.squeeze(i['image'].cpu().detach().numpy()), affine=np.eye(4))
    roi_subset = ['liver', 'kidney_left', 'kidney_right', 'pancreas', 'spleen']
    prediction_ts = np.transpose(np.array(totalsegmentator(input=nifti_img, output='/data/PANORAMA/cvillaseca/NETWORKS_SEGMENTACIO/BASELINE_SEGMENTACIO/test_results/Total_segmentator', roi_subset=roi_subset, quiet=True,skip_saving=False).get_fdata()), (2,0,1)).astype(np.uint8)
    output_nifti = nib.Nifti1Image(prediction_ts, affine=np.eye(4))
    output_path='/data/PANORAMA/cvillaseca/NETWORKS_SEGMENTACIO/BASELINE_SEGMENTACIO/test_results/Total_segmentator'
    nib.save(output_nifti, f'{output_path}/segmentation_result_{i["id"]}.nii.gz')#prediction_ts[prediction_ts != 4] = 0  
 