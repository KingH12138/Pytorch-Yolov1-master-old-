import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from utils.bbox_utils import bbox_format_transform, bbox_normalization, convert_yolov1_label, reshape_bbox
from utils.data_utils import txt_to_bboxinfo
from utils.train_utils import getconfigparser
import os


class VOCDataset(Dataset):
    def __init__(self, image_dir, csv_path, batch_size, resize=(448, 448)):
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.resize = resize
        self.classes = (
            getconfigparser('{}\config\config.cfg'.format(os.getcwd())).get('dataset-config', 'classes')).split(
            ',')  # configparser无法识别相对路径
        self.batch_size = batch_size
        self.num_classes = len(self.classes)
        self.transformer = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
        ])
        self.df = pd.read_csv(self.csv_path, encoding='utf-8')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        # 处理输入图片
        img_file = Image.open(self.df['path'][item])
        h, w = img_file.height, img_file.width
        x_train = self.transformer(img_file)
        # 处理bbox
        bboxes_init = txt_to_bboxinfo(self.df['object_txt_path'][item])
        scale = (self.resize[0] / h, self.resize[1] / w)
        bboxes_reshape = [reshape_bbox(bbox, scale) for bbox in bboxes_init]
        bboxes_trans = [bbox_format_transform(bbox, mode=1) for bbox in bboxes_reshape]
        bboxes_normal = [bbox_normalization(bbox,self.resize) for bbox in bboxes_trans]
        for i in range(len(bboxes_normal)):
            bboxes_normal[i][0] = self.classes.index(bboxes_normal[i][0])
        y_train = convert_yolov1_label(bboxes_normal)
        y_train = torch.Tensor(y_train)
        return x_train, y_train


def VOCDataloader(image_dir, csv_path, train_precent, batch_size, resize=(448, 448), shuffle=True):
    dataset = VOCDataset(image_dir, csv_path, batch_size, resize)
    num_sample = len(dataset)
    train_num = round(num_sample * train_precent)
    test_num = num_sample - train_num
    train_ds, test_ds = random_split(dataset, [train_num, test_num])
    train_dl = DataLoader(train_ds, batch_size, shuffle)
    test_dl = DataLoader(test_ds, batch_size, shuffle)

    return train_dl, test_dl
