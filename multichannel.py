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
import json

import torch
from torch.utils.data import Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from monai.transforms.utils import allow_missing_keys_mode

from monai.apps import DecathlonDataset
from monai.data import Dataset, decollate_batch, CSVDataset, CacheDataset, partition_dataset, ThreadDataLoader, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
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
)
from monai.utils import set_determinism
from monai.transforms.utils import allow_missing_keys_mode
from sklearn. model_selection import train_test_split
from sklearn.model_selection import KFold

import dUnet
from dUnet import doubleUnet, model1, model2


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MASTER_ADDR"] = "localhost"

def main_worker(args):
    # disable logging for processes except 0 on every node
    if int(os.environ["LOCAL_RANK"]) != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f
    
    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    
    start_time = time.time()
    total_start = time.time()
    
    #Create output directories
    outpath = args.folderpath
    os.makedirs(outpath, exist_ok=True)
    folderNames = ['index', 'model_weights', 'log', 'pred']
    for folder in folderNames:
        full_path = os.path.join(outpath,folder)
        os.makedirs(os.path.join(outpath,folder), exist_ok=True)

    #Read parameters
    parameter = os.path.join(outpath,'input.json')
    with open(parameter, 'r') as f:
        paths = json.load(f)
        
        INchannels1 = int(paths["input_channels_model1"])
        INchannels2 = int(paths["input_channels_model2"])
        
        OUTchannels = int(paths["output_channels"])
        
        max_epochs = int(paths["num_of_epochs"])
        Num_of_patches = int(paths["num_of_patches"])
        roi = paths["roi_size"]
        batch = int(paths["batch_size"])
        
        modal1 = paths["modality_model1"]
        input_sequence1 = [f'{item}' for item in modal1]
        
        modal2 = paths["modality_model2"]
        input_sequence2 = [f'{item}' for item in modal2]
        
        input_label = paths["mask"]      


    #Read data
    datafile = os.path.join(outpath,'dataset.csv')
    dataset1 = CSVDataset(
            src=[datafile], 
            col_groups = {"image1": input_sequence1, "label1": [input_label]},
            )
    
    dataset2 = CSVDataset(
            src=[datafile], 
            col_groups = {"image2": input_sequence2, "label2": [input_label]},
            )
    
    #Split Dataset as hold and test
    splits = KFold(n_splits = 5, shuffle = True, random_state = 21)
    for fold, (hold_idx, test_idx) in enumerate(splits.split(np.arange(len(dataset1.data)))):   
        print(str(fold))

        te = pd.DataFrame(test_idx)
        out_testIdx = os.path.join(outpath,'index', 'test_idx_fold_' + str(fold) + '.csv')
        te.to_csv(out_testIdx, index=False)   

        train_idx, val_idx = train_test_split(hold_idx, test_size = 0.1)

        tr = pd.DataFrame(train_idx)
        out_trainIdx = os.path.join(outpath,'index', 'train_idx_fold_' + str(fold) + '.csv')
        tr.to_csv(out_trainIdx, index=False)

        val = pd.DataFrame(val_idx)
        out_valIdx = os.path.join(outpath,'index', 'val_idx_fold_' + str(fold) + '.csv')
        val.to_csv(out_valIdx, index=False)

        #create a training data 
        train_set1 = Subset(dataset1.data, train_idx)
        val_set1 = Subset(dataset1.data, val_idx)
        
        train_set2 = Subset(dataset2.data, train_idx)
        val_set2 = Subset(dataset2.data, val_idx)

        train_transforms1 = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image1", "label1"]),
                EnsureChannelFirstd(keys=["image1", "label1"]),
                EnsureTyped(keys=["image1", "label1"]),
                NormalizeIntensityd(keys="image1", nonzero=True, channel_wise=True),
                RandCropByPosNegLabeld(
                    keys=["image1", "label1"],
                    label_key="label1",
                    spatial_size= roi,
                    pos=1,
                    neg=1,
                    num_samples= Num_of_patches,
                    image_key="image1",
                    image_threshold=0,
                    allow_smaller=True
                    ),
                ResizeWithPadOrCropd(keys=["image1", "label1"], spatial_size=roi, mode='constant'),
            ]
        )
    
        # create a training data loader
        train_data1 = Dataset(data=train_set1, transform=train_transforms1)
        #train_data1 = CacheDataset(data=train_set1, transform=train_transforms1, num_workers=4, cache_rate=1.0)
    
        train_ds1 = partition_dataset(
                    data=train_data1,
                    num_partitions=dist.get_world_size(),
                    shuffle=True,
                    seed=0,
                    drop_last=False,
                    even_divisible=True,
                )[dist.get_rank()]
    
        # ThreadDataLoader can be faster if no IO operations when caching all the data in memory
        train_loader1 = ThreadDataLoader(train_ds1, num_workers=0, batch_size=batch, shuffle=True)
    
        # validation transforms and dataset
        val_transforms1 = Compose(
            [
                LoadImaged(keys=["image1", "label1"]),
                EnsureChannelFirstd(keys=["image1", "label1"]),
                EnsureTyped(keys=["image1", "label1"]),
                NormalizeIntensityd(keys="image1", nonzero=True, channel_wise=True),
                DivisiblePadd(keys=["image1", "label1"], k=64, mode=('constant'), method= ("symmetric")),
            ]
        )
    
        # create a training data loader
        val_data1 = Dataset(data=val_set1, transform=val_transforms1)
        #val_data1 = CacheDataset(data=val_set1, transform=val_transforms1, num_workers=4, cache_rate=1.0)
    
        val_ds1 = partition_dataset(
                    data=val_data1,
                    num_partitions=dist.get_world_size(),
                    shuffle=False,
                    seed=0,
                    drop_last=False,
                    even_divisible=False,
                )[dist.get_rank()]
    
        # ThreadDataLoader can be faster if no IO operations when caching all the data in memory
        val_loader1 = ThreadDataLoader(val_ds1, num_workers=0, batch_size=batch, shuffle=False)
        
        train_transforms2 = Compose(
            [
                # load 4 Nifti images and stack them together
                LoadImaged(keys=["image2", "label2"]),
                EnsureChannelFirstd(keys=["image2", "label2"]),
                EnsureTyped(keys=["image2", "label2"]),
                NormalizeIntensityd(keys="image2", nonzero=True, channel_wise=True),
                RandCropByPosNegLabeld(
                    keys=["image2", "label2"],
                    label_key="label2",
                    spatial_size= roi,
                    pos=1,
                    neg=1,
                    num_samples= Num_of_patches,
                    image_key="image2",
                    image_threshold=0,
                    allow_smaller=True
                    ),
                ResizeWithPadOrCropd(keys=["image2", "label2"], spatial_size=roi, mode='constant'),
            ]
        )
    
        # create a training data loader
        train_data2 = Dataset(data=train_set2, transform=train_transforms2)
        #train_data2 = CacheDataset(data=train_set2, transform=train_transforms2, num_workers=4, cache_rate=1.0)
    
        train_ds2 = partition_dataset(
                    data=train_data2,
                    num_partitions=dist.get_world_size(),
                    shuffle=True,
                    seed=0,
                    drop_last=False,
                    even_divisible=True,
                )[dist.get_rank()]
    
        # ThreadDataLoader can be faster if no IO operations when caching all the data in memory
        train_loader2 = ThreadDataLoader(train_ds2, num_workers=0, batch_size=batch, shuffle=True)
        
        # validation transforms and dataset
        val_transforms2 = Compose(
            [
                LoadImaged(keys=["image2", "label2"]),
                EnsureChannelFirstd(keys=["image2", "label2"]),
                EnsureTyped(keys=["image2", "label2"]),
                NormalizeIntensityd(keys="image2", nonzero=True, channel_wise=True),
                DivisiblePadd(keys=["image2", "label2"], k=64, mode=('constant'), method= ("symmetric")),
            ]
        )
    
        # create a training data loader
        val_data2 = Dataset(data=val_set2, transform=val_transforms2)
        #val_data2 = CacheDataset(data=val_set2, transform=val_transforms2, num_workers=4, cache_rate=1.0)
    
        val_ds2 = partition_dataset(
                    data=val_data2,
                    num_partitions=dist.get_world_size(),
                    shuffle=False,
                    seed=0,
                    drop_last=False,
                    even_divisible=False,
                )[dist.get_rank()]
    
        # ThreadDataLoader can be faster if no IO operations when caching all the data in memory
        val_loader2 = ThreadDataLoader(val_ds2, num_workers=0, batch_size=batch, shuffle=False)
    
        # create network, loss function and optimizer
        model = doubleUnet(model1(channels = 32, input_channels = INchannels1), model2(channels = 32, input_channels = INchannels2),channels = 32, output_channels = OUTchannels).to(device)
        model = DistributedDataParallel(model, device_ids=[device])
  
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True) 
    
        # Create optimizer based on the argument
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), args.lr)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=0.00004)
    
    
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        post_label = AsDiscrete(to_onehot=5)
        post_pred = AsDiscrete(argmax=True, to_onehot=5)
        
        val_interval = 2
        best_metric = -1
        best_metric_epoch = -1
        
        epoch_num = []
        epoch_loss_values = []
        epoch_times = []
        
        metric_values = []

    
        for epoch in range(max_epochs):
            epoch_start = time.time()
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data1, batch_data2 in zip(train_loader1, train_loader2):
                step += 1
                inputs1, labels1 = (
                                batch_data1["image1"].to(device),
                                batch_data1["label1"].to(device),
                            )
                
                inputs2, labels2 = (
                                batch_data2["image2"].to(device),
                                batch_data2["label2"].to(device),
                            )
                
                optimizer.zero_grad()
                outputs = model(inputs1, inputs2) 
                loss = loss_function(outputs, labels1)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"{step}/{len(train_ds1) // train_loader1.batch_size}, " f"train_loss: {loss.item():.4f}")
            
            epoch_loss /= step
            epoch_num.append(epoch + 1)
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    
            ##Validation
            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for val_data1, val_data2 in zip(val_loader1, val_loader2):
                        val_inputs1, val_labels1 = (
                                val_data1["image1"].to(device),
                                val_data1["label1"].to(device),
                            )
                        val_inputs2, val_labels2 = (
                                val_data2["image2"].to(device),
                                val_data2["label2"].to(device),
                            )

                        val_outputs = model(val_inputs1, val_inputs2)
                        
                        val_labels_list = decollate_batch(val_labels1)
                        val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                        
                        val_outputs_list = decollate_batch(val_outputs)
                        val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                        
                        dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                        
                    metric = dice_metric.aggregate().item()
                    metric_values.append(metric)

                    #Write the val metrics to a CSV file
                    val_metrics = os.path.join(outpath,'log', 'val_metrics_' + str(fold) +'.csv')
                    fieldnames = ['val_dice']
                    df_val = pd.DataFrame(zip(metric_values), columns=fieldnames)
                    df_val.to_csv(val_metrics, index=False)

                    # reset the status for next validation round
                    dice_metric.reset()
    
                    if metric > best_metric:
                        best_metric = metric
    
                        best_metric_epoch = epoch + 1
                        if dist.get_rank() == 0:
                            torch.save(model.state_dict(), os.path.join(outpath, "model_weights", "best_metric_model_fold_" + str(fold) +  ".pth"))
                            print("saved new best metric model")
                        print(
                            f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                            f"\nbest mean dice: {best_metric:.4f} "
                            f"at epoch: {best_metric_epoch}"
                            )
    
            print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
            epoch_times.append(f"{(time.time() - epoch_start):.4f}")

            #Write the loss values to a CSV file
            loss_values = os.path.join(outpath,'log', 'loss_' + str(fold) +'.csv')
            fieldnames = ['epoch','train_loss', 'epoch_times']
            df = pd.DataFrame(zip(epoch_num, epoch_loss_values, epoch_times), columns=fieldnames) 
            df.to_csv(loss_values, index=False)
    
        ##Start predicting 
        test_set1 = Subset(dataset1.data, test_idx)
        test_transforms1 = Compose(
            [
                LoadImaged(keys=["image1", "label1"]),
                EnsureChannelFirstd(keys=["image1", "label1"]),
                EnsureTyped(keys=["image1", "label1"]),
                NormalizeIntensityd(keys="image1", nonzero=True, channel_wise=True),
                DivisiblePadd(keys=["image1", "label1"], k=64, mode=('constant'), method= ("symmetric")),
            ]
        )
        #test_ds1 = CacheDataset(data=test_set1, transform=test_transforms1, num_workers=4, cache_rate=1.0)
        test_ds1 = Dataset(data=test_set1, transform=test_transforms1)
        test_loader1 = DataLoader(test_ds1, batch_size=1, shuffle=False, num_workers=1)
        
        
        test_set2 = Subset(dataset2.data, test_idx)
        test_transforms2 = Compose(
            [
                LoadImaged(keys=["image2", "label2"]),
                EnsureChannelFirstd(keys=["image2", "label2"]),
                EnsureTyped(keys=["image2", "label2"]),
                NormalizeIntensityd(keys="image2", nonzero=True, channel_wise=True),
                DivisiblePadd(keys=["image2", "label2"], k=64, mode=('constant'), method= ("symmetric")),
            ]
        )
        #test_ds2 = CacheDataset(data=test_set2, transform=test_transforms2, num_workers=4, cache_rate=1.0)
        test_ds2 = Dataset(data=test_set2, transform=test_transforms2)
        test_loader2 = DataLoader(test_ds2, batch_size=1, shuffle=False, num_workers=1)
        
        map_location = (f"cuda:{os.environ['LOCAL_RANK']}")
        model.load_state_dict(torch.load(os.path.join(outpath, 'model_weights', "best_metric_model_fold_" + str(fold) + ".pth"), map_location=map_location, weights_only=True))
    
        def split_channels_to_binary(image_tensor):
            binary_images = []
            for channel in range(image_tensor.size(0)):
                binary_image = (image_tensor[channel] > 0).float()
                binary_images.append(binary_image)
            return binary_images

        pred_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        post_label = AsDiscrete(to_onehot=5)
        post_pred = AsDiscrete(argmax=True, to_onehot=5)
       
        pred = []
        pid_values = []

        with torch.no_grad():
            for test_data1, test_data2 in zip(test_loader1, test_loader2):
                
                test_inputs1, test_labels1 = test_data1["image1"].to(device), test_data1["label1"].to(device)
                test_inputs2, test_labels2 = test_data2["image2"].to(device), test_data2["label2"].to(device)
                
                test_outputs = model(test_inputs1, test_inputs2)
              
                original_affine = test_labels1.meta["original_affine"][0].numpy()
                img1_name = test_inputs1.meta["filename_or_obj"][0].split("/")[-1]
                pid = img1_name.split(".")[0]
                os.makedirs(os.path.join(outpath,'pred', pid), exist_ok=True)

                #Invert and save output to nii file     
                seg = (test_outputs)[0].detach().cpu()
                seg.applied_operations = test_data1["label1"][0].applied_operations
                seg_dict = {"label1": seg}
                
                with allow_missing_keys_mode(test_transforms1):
                    inverted_seg = test_transforms1.inverse(seg_dict)
                    out_seg = inverted_seg["label1"]
                    out_seg1 = out_seg.unsqueeze(0)

                out_seg1 = torch.softmax(out_seg1, 1).numpy()
                out_seg1 = np.argmax(out_seg1, axis=1).astype(np.uint8)[0]         

                outFile = "pred_" + pid + ".nii.gz"
                nib.save(nib.Nifti1Image(out_seg1.astype(np.uint8), original_affine), os.path.join(outpath,'pred', pid, outFile))
                
                binary_images = split_channels_to_binary(out_seg)

                label_NETC = binary_images[1]
                #label_NETC = torch.softmax(label_NETC, 1).numpy()
                #label_NETC = np.argmax(label_NETC, axis=1).astype(np.uint8)
                nib.save(nib.Nifti1Image(label_NETC.astype(np.uint8), original_affine), os.path.join(outpath,'pred', pid, 'NETC.nii.gz'))

                label_SNFH = binary_images[2]
                #label_SNFH = torch.softmax(label_SNFH, 0).cpu().numpy()
                #label_SNFH = np.argmax(label_SNFH, axis=0).astype(np.uint8)
                nib.save(nib.Nifti1Image(label_SNFH.astype(np.uint8), original_affine), os.path.join(outpath,'pred', pid, 'SNFH.nii.gz'))

                label_ET = binary_images[3]
                #label_ET = torch.softmax(label_ET, 0).cpu().numpy()
                #label_ET = np.argmax(label_ET, axis=0).astype(np.uint8)
                nib.save(nib.Nifti1Image(label_ET.astype(np.uint8), original_affine), os.path.join(outpath,'pred', pid, 'ET.nii.gz'))

                label_RC = binary_images[4]
                #label_TC = torch.softmax(label_TC, 1).cpu().numpy()
                #label_RC = np.argmax(label_RC, axis=0).astype(np.uint8)
                nib.save(nib.Nifti1Image(label_RC.astype(np.uint8), original_affine), os.path.join(outpath,'pred', pid, 'RC.nii.gz'))

                #write out dice metric
                test_labels_list = decollate_batch(test_labels1)
                test_labels_convert = [post_label(test_label_tensor) for test_label_tensor in test_labels_list]
                        
                test_outputs_list = decollate_batch(test_outputs)
                test_output_convert = [post_pred(test_pred_tensor) for test_pred_tensor in test_outputs_list]
                        
                pred_metric(y_pred=test_output_convert, y=test_labels_convert)
                        
                pred_dice = pred_metric.aggregate().item()                    
                pred.append(f"{pred_dice:.4f}")
            
                pid_values.append(pid)
            
            pred_metric.reset()

            #Write the val metrics to a CSV file
            pred_metrics = os.path.join(outpath,'log', 'pred_metrics_' + str(fold) +'.csv')
            fieldnames = ['pid', 'pred_dice']
            df_val = pd.DataFrame(zip(pid_values, pred), columns=fieldnames)
            df_val.to_csv(pred_metrics, index=False)

        print("--- %s seconds ---" % (time.time() - start_time))
    
        print('clearing')
        gc.collect()
        torch.cuda.empty_cache()

    dist.destroy_process_group()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
    parser.add_argument("--nproc_per_node")
    parser.add_argument("--folderpath")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer to use (default: Adam)')
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

    main_worker(args=args)
