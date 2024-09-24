import os
import torch
import pandas as pd
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    ThresholdIntensityd, NormalizeIntensityd, Spacingd, Lambda, AsDiscreted
)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import AsDiscreted
from utils.loss_factory import get_loss
import numpy as np
from tqdm import trange
import torch
from scipy.stats import rankdata


# Paths and device configuration
data_dir = "/data/PANORAMA/cvillaseca/NETWORKS_SEGMENTACIO/BASELINE_SEGMENTACIO/data"  
model_path = "/data/PANORAMA/cvillaseca/NETWORKS_SEGMENTACIO/BASELINE_SEGMENTACIO/experiments/AMOS_pesos/best_model.pth"  
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

# Load the model
model = SwinUNETR(img_size=(96, 96, 96), in_channels=1, out_channels=5, feature_size=12, use_checkpoint=False)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'

loss1='ce'
loss2='dice'
alpha1=1
alpha2=1

# Initialize loss and metrics
loss_fn= get_loss(loss1, loss2, alpha1, alpha2)

#def fast_multiclass_dice(actual, predicted, n_classes):
    #actual = np.squeeze(np.array(actual))
    #predicted = np.squeeze(np.array(predicted))
    #print('actual shape:', actual.shape)
    #print('actual dtype', actual.dtype)
    #print('predicted shape:', predicted.shape)
    #print('predicted dtype', predicted.dtype)

    # Initialize an array to store the dice score for each class
    #dices = np.zeros(n_classes) 
    #for cls in range(n_classes):
        #actual_cls = (actual == cls)
        #predicted_cls = (predicted == cls)
        #actual_cls = np.array(actual_cls).astype(bool)
        #predicted_cls = np.array(predicted_cls).astype(bool)
        #print('actual_cls shape:', actual_cls.shape)
        #print('actual_cls dtype', actual_cls.dtype)
        #print('predicted_cls shape:', predicted_cls.shape)
        #print('predicted_cls dtype', predicted_cls.dtype)
        
        #intersections = np.logical_and(actual_cls, predicted_cls).sum(axis=(0, 1, 2))
        #im_sums = actual_cls.sum(axis=(0, 1, 2)) + predicted_cls.sum(axis=(0, 1, 2))
        #dices[cls] = 2. * intersections / np.maximum(im_sums, 1e-6)
    #return dices

def fast_bin_auc(actual, predicted, partial=False):
    actual, predicted = actual.flatten(), predicted.flatten()
    if partial:
        n_nonzeros = np.count_nonzero(actual)
        n_zeros = len(actual) - n_nonzeros
        k = min(n_zeros, n_nonzeros)
        predicted = np.concatenate([
            np.sort(predicted[actual == 0])[::-1][:k],
            np.sort(predicted[actual == 1])[::-1][:k]
        ])
        actual = np.concatenate([np.zeros(k), np.ones(k)])

    r = rankdata(predicted)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    if n_pos == 0 or n_neg == 0: return 0
    return (np.sum(r[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


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

# Define the directory and log file path
log_dir = "/data/PANORAMA/cvillaseca/NETWORKS_SEGMENTACIO/BASELINE_SEGMENTACIO/test_results/AUC_logs"
log_file = os.path.join(log_dir, "AUC_log.txt")

# Ensure the directory exists
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Inference loop
all_preds = []
all_labels = []
all_dsc = []

with torch.no_grad():
    with trange(len(test_loader)) as t:
        n_elems, running_dsc = 0, 0
        for val_data in test_loader:
            val_images, val_labels = val_data['image'].to(device), val_data['label']
            #torch.cuda.empty_cache()
            print('input images:', val_images.shape) #it should be (1 (batch_size),1 (black and white),x,y,z)
            print('input labels:', val_labels.shape) #it should be (1,2(classes), x,y,z) --> this is in one hot encoding, which means that the second position of the tensor contains a 2-dimension vector with the probabilities of each class for that voxel. The second dimension is the number of classes, which is 2. This dimension contains the one-hot encoded vectors for each voxel.
            val_outputs = sliding_window_inference(val_images.to(device), (96,96,96), 1, model).cpu()
            print('val outputs:', val_outputs.shape) #Since we have the output as one-hot encoded, it should be (1,5,x,y,z), because in the second position (5) it is representing the model's confidence in that voxel belonging to one of the 5 classes. 
            print('val_outputs_voxel', val_outputs[0, :, 50, 50, 50]) #this should give a vector of 5 values, each consisting on the probability of the voxel in position (50,50,50) pertaining to each of the classes and the result of their sum is 1

            #argmax(dim=1) converts this one-hot encoding to class labels by taking the index of the maximum value along the class dimension.
            val_pred = val_outputs.argmax(dim=1).cpu().numpy()  # Takes the maximum value in the second position of the predicted output (the one that is a vector with the probabilitie sof the voxel pertaining to each class). Basically, we are converting from one-hot to class labels. Move tensor to CPU before converting to numpy. It will respond with a 
            # Map class 4 (pancreas) to 1 and all other classes to 0 (background)
            val_pred[val_pred != 4] = 0  # Set all non-class-4 predictions to background (0)
            val_pred[val_pred == 4] = 1  # Set class 4 predictions to 1 (foreground)
            print('val_pred:', val_pred.shape)
            print('val_pred_voxel', val_pred[:, 50, 50, 50])

            val_seg = val_labels.argmax(dim=1).cpu().numpy()  # Move tensor to CPU before converting to numpy
            print('val_seg:', val_seg.shape) #it is the ground truth, but since we did the argmax, instead of having it as one hot encoded, we have it as class labels, in order to be able to compare it with the model's predictions.
            print('val_seg_voxel', val_seg[:, 50, 50, 50])
            #dsc_scores = fast_multiclass_dice(val_seg, val_pred, n_classes=2)
            dsc_scores=fast_bin_auc(val_seg,val_pred)
            # Append DSC scores to the log list
            all_dsc.append(dsc_scores)
            print(dsc_scores)

            all_dsc.append(dsc_scores)
            n_elems += 1
            #running_dsc += np.mean(dsc_scores[1:]) # not the background
            #run_dsc = running_dsc / n_elems
            #t.set_postfix(DSC="{:.2f}".format(100 * run_dsc))
            #t.update()

            with open(log_file, 'a') as f:
                f.write(f"Batch {n_elems}, {dsc_scores.tolist()}\n")

mean_dsc= np.mean(all_dsc)
print(f"Mean DSC: {mean_dsc}")

with open(log_file, 'a') as f:
    f.write(f"Mean DSC: {mean_dsc}\n")

