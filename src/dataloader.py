import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class DIV2KDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        """
        Args:
            lr_dir: Directory with low-resolution images (64x64)
            hr_dir: Directory with high-resolution images (256x256)
            transform: Optional transform to be applied on images
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform
        
        # Get list of image files
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        
        # Ensure same number of images
        assert len(self.lr_images) == len(self.hr_images), "Mismatch in number of LR and HR images"
    
    def __len__(self):
        return len(self.lr_images)
    
    def __getitem__(self, idx):
        # Load LR and HR images
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        
        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')
        
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        
        return lr_image, hr_image

class DIV2KDataModule:
    def __init__(self, lr_dir='Dataset', hr_dir='Dataset'):
        """
        Args:
            lr_dir: Directory with low-resolution images
            hr_dir: Directory with high-resolution images
            batch_size: Number of samples per batch
            num_workers: Number of subprocesses to use for data loading
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # self.batch_size = batch_size
        # self.num_workers = num_workers

    def setup(self):
        self.train_dataset = DIV2KDataset(
            lr_dir=os.path.join(self.lr_dir, 'train'),
            hr_dir=os.path.join(self.hr_dir, 'train_labels'),
            transform=self.transform
        )
        
        self.val_dataset = DIV2KDataset(
            lr_dir=os.path.join(self.lr_dir, 'validation'),
            hr_dir=os.path.join(self.hr_dir, 'validation_labels'),
            transform=self.transform
        )
        
        self.test_dataset = DIV2KDataset(
            lr_dir=os.path.join(self.lr_dir, 'test'),
            hr_dir=os.path.join(self.hr_dir, 'test_labels'),
            transform=self.transform
        )

    def train_dataloader(self, batch_size=16, num_workers=4):
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    def val_dataloader(self, batch_size=16, num_workers=4):
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    def test_dataloader(self, batch_size=16, num_workers=4):
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
