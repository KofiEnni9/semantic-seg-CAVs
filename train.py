import os
import warnings
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from calc_iou import calculate_iou
from infer.infer import inferring_img
from losses.builder import build_criteria
from network._deeplab import convert_to_separable_conv
import network.modeling
from dataprocessin.d_builder_ds import build_dataset, build_dataloader
from utils import set_bn_momentum


warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Configure logging
logging.basicConfig(
    filename='training.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info("Starting the script.")

# Configuration for dataset paths
train_cfg = {
    'type': 'prepSegmentationDataset',
    'img_dir': 'data/new_TrainM/imgs',
    'ann_dir': 'data/new_TrainM/annos',
    'is_train': True 
}

val_cfg = {
    'type': 'prepSegmentationDataset',
    'img_dir': 'data/new_TestM/mixed/imgs',
    'ann_dir': 'data/new_TestM/mixed/annos',
    'is_train': False
}

# Define the datasets using build_dataset
train_dataset = build_dataset(train_cfg)
val_dataset = build_dataset(val_cfg)

# Define the DataLoader with build_dataloader
logging.info("Building dataloaders...")
train_loader = build_dataloader(dataset=train_dataset, samples_per_gpu=6, workers_per_gpu=0, dist=False)
val_loader = build_dataloader(dataset=val_dataset, samples_per_gpu=6, workers_per_gpu=0, dist=False)
logging.info("Dataloaders built successfully.")


def deeplab_train(train_loader, val_loader, num_classes, device='cpu'):

    # Initialize model
    logging.info("Starting model...")
    model = network.modeling.__dict__["deeplabv3plus_resnet50"](num_classes=num_classes, output_stride=16)
    
    convert_to_separable_conv(model.classifier)
    set_bn_momentum(model.backbone, momentum=0.01)
    num_epochs = 15
    
    # Setup training components
    # ce_criterion = nn.CrossEntropyLoss(ignore_index=0 )
    ce_criterion = build_criteria([
        {
            'type': 'CrossEntropyLoss',
            'loss_weight': 0.6,
            'ignore_index': 0
        },
        {
            'type': 'LovaszLoss',
            'mode': 'multiclass',
            'loss_weight': 0.4,
            'ignore_index': 0
        }
    ])
    
    optimizer = optim.AdamW([
        {'params': model.parameters(), 'lr': 1e-4},
    ], weight_decay=0.01)

    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch in train_bar:
            images = batch['img'].to(device)
            masks = batch['gt_semantic_seg'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            if isinstance(outputs, tuple):
                aux_loss = ce_criterion(outputs[0], masks)
                main_loss = ce_criterion(outputs[1], masks)
                loss = main_loss + 0.4 * aux_loss
            else:
                loss = ce_criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        logging.info(f'Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0
        correct = total = 0
        ious = []
        
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for batch in val_bar:
                images = batch['img'].to(device)
                masks = batch['gt_semantic_seg'].to(device)
                
                outputs = model(images)
                
                if isinstance(outputs, tuple):
                    _, main_output = outputs
                else:
                    main_output = outputs
                
                loss = ce_criterion(main_output, masks)
                val_loss += loss.item()
                
                _, predicted = torch.max(main_output.data, 1)

                # Calculate correct predictions and total number of elements
                correct += (predicted == masks).sum().item()
                total += masks.numel()

                # Calculate IoU for this batch
                batch_ious = calculate_iou(predicted, masks, num_classes)
                ious.append(batch_ious)

                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # if epoch >= warm_up:
        #     scheduler.step()

        # Inference on a few validation images and save results
        for j in range(min(6, predicted.shape[0])):
            inferring_img(predicted[j], masks[j], epoch * len(val_loader) + j)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total * 100

        # Calculate mean IoU correctly
        ious = np.array(ious)
        mean_ious = np.nanmean(ious, axis=0)
        miou = np.nanmean(mean_ious)

        logging.info(f'Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%, mIoU: {miou:.4f}')
        logging.info(f'Epoch {epoch+1} - IoU per class: {", ".join([f"{iou:.4f}" for iou in mean_ious])}\n')


        print(f'Epoch {epoch+1}/{num_epochs}:')
        # print(f'Current Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        print(f'Validation mIoU: {miou:.4f}')
        print(f'IoU per class: {", ".join([f"{iou:.4f}" for iou in mean_ious])}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_os16.pth')

            print('Saved new best model')
            logging.info(f'Epoch {epoch+1} - Saved new best model.\n')



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_classes = 7  # Adjust based on dataset
    
    deeplab_train(train_loader, val_loader, num_classes)

    