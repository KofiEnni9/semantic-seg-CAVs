from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
import os
import torch
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import cv2  # Import OpenCV for image processing
import numpy as np
import os
from .d_builder_ds import DATASETS
import albumentations as A


@DATASETS.register_module()
class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    concat the group flag for image aspect ratio.

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = datasets[0].PALETTE


@DATASETS.register_module()
class RepeatDataset(object):
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item from original dataset."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """The length is multiplied by ``times``"""
        return self.times * self._ori_len
    
@DATASETS.register_module()
class prepSegmentationDataset(Dataset):
    def __init__(self, img_dir, ann_dir, is_train, transform=None):
        """
        Args:
            img_dir (str): Directory with images.
            ann_dir (str): Directory with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.is_train = is_train
        self.image_list = self.get_image_list()

        # Define augmentations for training
        if is_train:
            self.aug_transform = A.Compose([
                A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(p=0.2),
                A.OneOf([
                    A.MotionBlur(p=0.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                ], p=0.2),
            ])
        else:
            self.aug_transform = A.Compose([
                A.Resize(height=512, width=512)
            ])

        self.target_colors = {

                # what it does is search for the colors below and if a color doesnt match any of these ones below 
                # it defaults to 0 which is ignored in training

            0: (0, 0, 0),        # ignored: every color without matches below to be pased to this
            1: (208, 254, 157),  # Light green   # light greenish  # the use of two colors is because the images in some folders have different colors for say grass 
            2: (59, 93, 4),  # Dark green   # deep green
            3: (155, 155, 154),  # Gray
            4: (138, 87, 42),    # Brown
            5: (183, 21, 123),   # Pink
            6: (73, 143, 225)    # Blue

                # adjustments for MAVS sim
                # 0: (0, 0, 0),
                # 1: (235, 124, 47), # deep brown veg ground 
                # 2: (144, 208, 79), # light green tree
                # 3: (55, 86, 34), # deep green grass
                # 4: (134, 206, 234), # blue sky
                # 5: (254, 191, 0) # ligh brown tranversible ground
            }

    def get_image_list(self):
        img_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.bmp', '.png', '.jpeg'))])
        ann_files = sorted([f for f in os.listdir(self.ann_dir) if f.lower().endswith(('.bmp', '.png', '.jpeg'))])
        assert len(img_files) == len(ann_files), "Number of raw images and annotations do not match"
        return list(zip(img_files, ann_files))
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_rel_path, ann_rel_path = self.image_list[idx]
        img_path = os.path.join(self.img_dir, img_rel_path)
        ann_path = os.path.join(self.ann_dir, ann_rel_path)

        # Load image and annotation
        image = self.load_image(img_path)
        annotation = self.load_annotation(ann_path)


        if image is None or annotation is None:
            raise ValueError(f"Failed to load image or annotation at index {idx}")

        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        annotation = cv2.resize(annotation, (512, 512), interpolation=cv2.INTER_NEAREST)


        # Apply augmentations
        transformed = self.aug_transform(image=image, mask=annotation)
        image, annotation = transformed['image'], transformed['mask']

        return {
            'img': torch.from_numpy(image).permute(2, 0, 1).float() / 255.0,  # Normalize to [0,1]
            'gt_semantic_seg': torch.from_numpy(annotation).long()
        }


    def load_image(self, path):
        """Load and validate image"""
        image = cv2.imread(path)
        if image is None:
            print(f"Failed to load image: {path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return image

    def load_annotation(self, path):
        """Load and process annotation mask"""
        # Read the annotation image
        annotation = cv2.imread(path, cv2.IMREAD_COLOR)
        if annotation is None:
            print(f"Failed to load annotation: {path}")
            return None

        # Convert BGR to RGB
        annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2RGB)

        # Initialize the mask
        mask = np.zeros((annotation.shape[0], annotation.shape[1]), dtype=np.int64)
        
        # Reshape annotation for efficient color comparison
        annotation_reshaped = annotation.reshape(-1, 3)
        mask_reshaped = np.zeros(annotation_reshaped.shape[0], dtype=np.int64)

        # Check for each target color
        tolerance = 5  # Define tolerance for color matching
        for class_idx, color in self.target_colors.items():
            if isinstance(color, tuple):  # Single RGB color
                within_tolerance = np.all(np.abs(annotation_reshaped - color) <= tolerance, axis=1)
            elif isinstance(color, list): 
                color1, color2 = color
                within_tolerance1 = np.all(np.abs(annotation_reshaped - color1) <= tolerance, axis=1)
                within_tolerance2 = np.all(np.abs(annotation_reshaped - color2) <= tolerance, axis=1)

                within_tolerance = np.logical_or(within_tolerance1, within_tolerance2)

            # Assign class index where color matches
            mask_reshaped[within_tolerance] = class_idx

        # Reshape the mask back to the original image dimensions
        mask = mask_reshaped.reshape(annotation.shape[:2])

        return mask
