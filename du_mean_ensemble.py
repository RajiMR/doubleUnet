import argparse
import os
import sys
import time
import warnings
import pandas as pd
import gc
import numpy as np
import logging
import nibabel as nib

import torch
from torch.utils.data import Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from monai.transforms.utils import allow_missing_keys_mode

from monai.apps import DecathlonDataset
from monai.engines import EnsembleEvaluator
from monai.handlers import MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.data import Dataset, decollate_batch, CSVDataset, CacheDataset, partition_dataset, ThreadDataLoader, DataLoader
from monai.inferers import SlidingWindowInferer, sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Activationsd,
    AsDiscrete,
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    ResizeWithPadOrCropd,
    Spacingd,
    ToDeviced,
    EnsureTyped,
    MeanEnsembled,
    SaveImaged,
)
from monai.utils import set_determinism
from sklearn. model_selection import train_test_split
from sklearn.model_selection import KFold

import dUnet
from dUnet import doubleUnet, model1, model2

import json

def main(args):
    
    device='cuda:1'
    outpath = args.outpath
    modelpath = args.modelpath
    modelname = args.modelname

    #Create output directories
    os.makedirs(os.path.join(outpath,modelname, 'pred'), exist_ok=True)
    
    #Read parameters
    parameter = os.path.join(modelpath, modelname, 'input.json')
    with open(parameter, 'r') as f:
        paths = json.load(f)   
            
        INchannels1 = int(paths["input_channels_model1"])
        INchannels2 = int(paths["input_channels_model2"])
        
        OUTchannels = int(paths["output_channels"])
                
        modal1 = paths["modality_model1"]
        input_sequence1 = [f'{item}' for item in modal1]
        
        modal2 = paths["modality_model2"]
        input_sequence2 = [f'{item}' for item in modal2]
        
        input_label = paths["mask"]
        
    #Read data
    datafile = os.path.join(outpath,'dataset.csv')
    
    dataset1 = CSVDataset(
            src=[datafile], 
            col_groups = {"image1": input_sequence1, "label": [input_label]},
            )
    test_transforms1 = Compose(
        [
            LoadImaged(keys=["image1", "label"]),
            EnsureChannelFirstd(keys=["image1", "label"]),
            #Orientationd(keys=["image1", "label"], axcodes="RAS"),
            #Spacingd(
            #    keys=["image1", "label"],
            #    pixdim=(1.0, 1.0, 1.0),
            #    mode=("bilinear", "nearest"),
            #),
            NormalizeIntensityd(keys="image1", nonzero=True, channel_wise=True),
            #DivisiblePadd(keys=["image1", "label"], k=8, mode=('constant'), method= ("symmetric")),
            EnsureTyped(keys=["image1", "label"]),
        ]
    )
    test_ds1 = CacheDataset(data=dataset1.data, transform=test_transforms1, num_workers=4, cache_rate=1.0)
    test_loader1 = DataLoader(test_ds1, batch_size=1, shuffle=False, num_workers=1)
        
    dataset2 = CSVDataset(
            src=[datafile], 
            col_groups = {"image2": input_sequence2, "label": [input_label]},
            )
    test_transforms2 = Compose(
        [
            LoadImaged(keys=["image2", "label"]),
            EnsureChannelFirstd(keys=["image2", "label"]),
            #Orientationd(keys=["image2", "label"], axcodes="RAS"),
            #Spacingd(
            #    keys=["image2", "label"],
            #    pixdim=(1.0, 1.0, 1.0),
            #    mode=("bilinear", "nearest"),
            #),
            NormalizeIntensityd(keys="image2", nonzero=True, channel_wise=True),
            #DivisiblePadd(keys=["image2", "label"], k=8, mode=('constant'), method= ("symmetric")),
            EnsureTyped(keys=["image2", "label"]),
        ]
    )
    test_ds2 = CacheDataset(data=dataset2.data, transform=test_transforms2, num_workers=4, cache_rate=1.0)
    test_loader2 = DataLoader(test_ds2, batch_size=1, shuffle=False, num_workers=1)
 
    num_models = 5
    models = []

    for fold in range(num_models):
        device = torch.device("cuda:0") 
        model_path = os.path.join(modelpath, modelname, 'model_weights', "best_metric_model_fold_" + str(fold) + ".pth")

        """
        Loads a model trained with DistributedDataParallel (DDP) onto a single GPU.

        Args:
            model_path (str): Path to the saved DDP model checkpoint.
            model_class (class): The class of your model (e.g., MyModel).
            device (str): The device to load the model onto (e.g., 'cuda:0').

        Returns:
            torch.nn.Module: The loaded model on the specified device.
        """

        # Load the state dictionary from the checkpoint
        state_dict = torch.load(model_path, map_location=device)

        # Remove the 'module.' prefix from the keys in the state dictionary
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # Create an instance of your model class
        model = doubleUnet(model1(channels = 32, input_channels = INchannels1), model2(channels = 32, input_channels = INchannels2),channels = 32, output_channels = OUTchannels).to(device)

        # Load the state dictionary into the model
        model.load_state_dict(new_state_dict)

        # Move the model to the specified device
        model.to(device)
        
        models.append(model)
        
    pred_values = []
    pid_values = []
    pred_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=int(OUTchannels))]) 
    post_label = Compose([AsDiscrete(to_onehot=int(OUTchannels))])
        
    with torch.no_grad():
        for test_data1, test_data2 in zip(test_loader1, test_loader2):
            test_inputs1, test_labels1 = test_data1["image1"].to(device), test_data1["label"].to(device)
            test_inputs2, test_labels2 = test_data2["image2"].to(device), test_data2["label"].to(device)

            original_affine = test_labels1.meta["original_affine"][0].numpy()
            _, _, h, w, d = test_labels1.shape
            target_shape = (h, w, d)

            img_name = test_inputs1.meta["filename_or_obj"][0].split("/")[-1]
            pid1 = img_name.split(".")[0]
            pid = pid1.split("_")[2]
            
            predictions = []
            stacked_images = []

            for model in models:
                model.eval()  # Set the model to evaluation mode
                test_outputs = model(test_inputs1, test_inputs2)
                predictions.append(test_outputs)

            stacked_images = torch.stack(predictions, dim=0)

            ensemble_predictions1=torch.mean(stacked_images, dim=0)            
            test_pred = [post_pred(i) for i in decollate_batch(ensemble_predictions1)]
            pred_metric(y_pred=test_pred, y=test_labels1)

            ensemble_predictions=torch.mean(stacked_images, dim=0)[0].detach().cpu()
            
            ensemble_predictions.applied_operations = (test_labels1)[0].applied_operations
            seg_dict = {"label": ensemble_predictions}
            with allow_missing_keys_mode(test_transforms1):
                inverted_seg = test_transforms1.inverse(seg_dict)
                out_seg = inverted_seg["label"]
                out_seg = out_seg.unsqueeze(0)
            
            out_seg = torch.softmax(out_seg, 1).cpu().numpy()
            out_seg = np.argmax(out_seg, axis=1).astype(np.uint8)[0]   

            outFile = "ensemble_" + pid + ".nii.gz"
            nib.save(nib.Nifti1Image(out_seg.astype(np.uint8), original_affine), os.path.join(outpath, modelname, 'pred', outFile))
            
            pred = pred_metric.aggregate().item()
            print(f"{pred:.4f}")
            pred_values.append(f"{pred:.4f}")
            pid_values.append(pid)
            pred_metric.reset()

        #Write the loss values to a CSV file
        diceFile = os.path.join(outpath, 'results', 'ensemble_dice_' + modelname + '.csv')
        fieldnames = ['pid','dice'] 
        df = pd.DataFrame(zip(pid_values, pred_values), columns=fieldnames) 
        df.to_csv(diceFile, index=False)
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
    parser.add_argument("--modelpath")
    parser.add_argument("--outpath")
    parser.add_argument("--modelname")
    args = parser.parse_args()

    if args.seed is not None:
        set_determinism(seed=args.seed)
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    main(args=args)
