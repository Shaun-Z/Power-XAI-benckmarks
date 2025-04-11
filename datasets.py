import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class MixedImgDataset(Dataset):
    def __init__(self, root_dir='./data/MixedImg', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        # Collect all image paths and their labels
        for label in self.classes:
            class_path = os.path.join(root_dir, label)
            if os.path.isdir(class_path):
                image_files = glob.glob(os.path.join(class_path, '*.jpg')) + glob.glob(os.path.join(class_path, '*.png'))
                for img_path in image_files:
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[label])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label