import os
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

DATASET_PATH = r'/home/blt/archive/SiM3D'

def class_labels():
    return [
        "plastic_stool",
        "rubbish_bin",
        "wicker_vase",
        "bathroom_forniture",
        "container",
        "plastic_vase",
        "sink_cabinet",
        "wooden_stool",
    ]

class SquarePad():
    def __call__(self, image):
        max_wh = max(image.shape)
        pad_shape = [image.shape[-1], image.shape[0]]
        p_left, p_top = [(max_wh - s) // 2 for s in pad_shape]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(pad_shape, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return transforms.functional.pad(Image.fromarray(image), padding, padding_mode = 'edge')
    
class RemoveInf():
    def __call__(self, image):
        image[image == np.inf] = 0.0
        return image
    
class RemoveMax():
    def __call__(self, image):
        image[image == image.max()] = 0.0
        return image
    
class BaseAnomalyDetectionDataset(Dataset):
    def __init__(self, class_name, img_size, dataset_path):
        self.IMAGENET_MEAN = [0.445]
        self.IMAGENET_STD = [0.269]

        self.cls = class_name
        self.size = img_size
        self.dataset_path = dataset_path # /home/blt/archive/SiM3D

        self.cls_path = os.path.join(self.dataset_path, self.cls) # /home/blt/archive/SiM3D/plastic_stool
        
        self.rgb_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((self.size, self.size), interpolation = transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean = self.IMAGENET_MEAN, std = self.IMAGENET_STD)
            ])
        
        self.xyz_transform = transforms.Compose([
            RemoveInf(),
            SquarePad(),
            transforms.Resize((self.size, self.size), interpolation = transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            ])
        
        self.xyz_synth_transform = transforms.Compose([
            RemoveMax(),
            SquarePad(),
            transforms.Resize((self.size, self.size), interpolation = transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            ])

class TrainDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(class_name = class_name, img_size = img_size, dataset_path = dataset_path)

        self.img_paths, self.xyz_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        xyz_paths = glob.glob(os.path.join(self.cls_path, self.cls + '_real', 'depth') + "/*.npy") # /home/blt/archive/SiM3D/plastic_stool/plastic_stool_real/depth/*.npy
        rgb_paths = [path.replace('.npy', '_2.png').replace('depth', 'rgb') for path in xyz_paths] # /home/blt/archive/SiM3D/plastic_stool/plastic_stool_real/rgb/*_2.png
        rgb_paths.sort(), xyz_paths.sort()
        return rgb_paths, xyz_paths, [0] * len(rgb_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path, xyz_path, label = self.img_paths[idx], self.xyz_paths[idx], self.labels[idx]

        rgb = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        rgb = rgb / rgb.max()
        xyz = np.load(xyz_path)

        rgb = self.rgb_transform(rgb)
        rgb = rgb.repeat(3, 1, 1)

        xyz = self.xyz_transform(xyz)
        xyz = xyz / xyz.max()

        path = rgb_path.replace(self.dataset_path, '') # /plastic_stool/plastic_stool_real/rgb/*_2.png

        return (rgb, xyz), label, path

class TrainSynthDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(class_name = class_name, img_size = img_size, dataset_path = dataset_path)

        self.img_paths, self.xyz_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        xyz_paths = glob.glob(os.path.join(self.cls_path, self.cls + '_synth', 'DEPTH') + "/*.exr")
        rgb_paths = [path.replace('.exr', '.png').replace('DEPTH', 'RGB').replace('_depth', '_2') for path in xyz_paths]
        rgb_paths.sort(), xyz_paths.sort()
        return rgb_paths, xyz_paths, [0] * len(rgb_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path, xyz_path, label = self.img_paths[idx], self.xyz_paths[idx], self.labels[idx]

        rgb = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        rgb = rgb / rgb.max()
        xyz = cv2.imread(xyz_path, cv2.IMREAD_UNCHANGED)

        rgb = self.rgb_transform(rgb)
        rgb = rgb.repeat(3, 1, 1)

        xyz = self.xyz_synth_transform(xyz[...,0])
        xyz = xyz / xyz.max()
        xyz[xyz < 0.0] = 0.0

        path = rgb_path.replace(self.dataset_path, '')

        return (rgb, xyz), label, path

class TestDataset(BaseAnomalyDetectionDataset):
    def __init__(self, class_name, img_size, dataset_path):
        super().__init__(class_name = class_name, img_size = img_size, dataset_path = dataset_path)

        self.img_paths, self.xyz_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        sub_folders = os.listdir(self.cls_path)
        sub_folders = [f for f in sub_folders if 'real' not in f and 'synth' not in f]
        sub_folders.sort()

        rgb_paths_f, xyz_paths_f, labels_f = [], [], [] 

        for sub_folder in sub_folders:
            xyz_paths = glob.glob(os.path.join(self.cls_path, sub_folder, 'depth') + "/*.npy")
            rgb_paths = [path.replace('.npy', '_2.png').replace('depth', 'rgb') for path in xyz_paths]
            rgb_paths.sort(), xyz_paths.sort()

            if 'bad' in sub_folder:
                labels = [1] * len(rgb_paths)
            elif 'bad' not in sub_folder:
                labels = [0] * len(rgb_paths)

            rgb_paths_f.extend(rgb_paths), xyz_paths_f.extend(xyz_paths), labels_f.extend(labels)

        return rgb_paths_f, xyz_paths_f, labels_f

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        rgb_path, xyz_path, label = self.img_paths[idx], self.xyz_paths[idx], self.labels[idx]

        rgb = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
        rgb = rgb / rgb.max()
        xyz = np.load(xyz_path)

        rgb = self.rgb_transform(rgb)
        rgb = rgb.repeat(3, 1, 1)
        
        xyz = self.xyz_transform(xyz)
        xyz = xyz / xyz.max()

        path = rgb_path.replace(self.dataset_path, '') # /plastic_stool/plastic_stool_01_bad/rgb/....

        return (rgb, xyz), label, path

def get_data_loader(split, class_name, dataset_path, img_size, batch_size, shuffle = False):
    if split == 'train':
        dataset = TrainDataset(class_name = class_name, img_size = img_size, dataset_path = dataset_path)
    elif split == 'synth':
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        dataset = TrainSynthDataset(class_name = class_name, img_size = img_size, dataset_path = dataset_path)
    elif split == 'test':
        dataset = TestDataset(class_name = class_name, img_size = img_size, dataset_path = dataset_path)

    data_loader = DataLoader(
        dataset = dataset, batch_size = batch_size, shuffle = shuffle, 
        num_workers = 1, drop_last = False, pin_memory = True
        )
    
    return data_loader