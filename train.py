from utils.train_utils import log_generator
from datetime import datetime
from tqdm import tqdm
from utils.train_utils import getconfigparser, stringsplot
from model.loss import Yolov1_loss
from data.dataset import VOCDataloader
from model.model import YOLOv1_resnet
from torch.optim import Adam
import torch
import os

workplace = os.getcwd()
# 读取参数
parser = getconfigparser(workplace + '/config/config.cfg')

NUM_BBOX = parser.getint("dataset-config", "num_bbox")
CLASSES = parser.get("dataset-config", 'classes').split(',')
GRID_NUM = (parser.getint("dataset-config", "grid_x"), parser.getint("dataset-config", "grid_y"))

csv_path = parser.get('dataset-config', 'csv_path')
txt_dir = parser.get('dataset-config', 'txt_dir')
image_dir = parser.get('dataset-config', 'image_dir')
xml_dir = parser.get('dataset-config', 'xml_dir')
num_bbox = parser.get('dataset-config', 'num_bbox')
resizeh = parser.getint('dataset-config', 'resizeh')
resizew = parser.getint('dataset-config', 'resizew')

epoches = parser.getint('train-config', 'epoches')
learning_rate = parser.getfloat('train-config', 'learning_rate')
log_dir = parser.get('train-config', 'log_dir')
batch_size = parser.getint('train-config', 'batch_size')

# getvoccsv(xml_dir,'./VOC2007',txt_dir,image_dir)

# 载入数据集
train_dl, _ = VOCDataloader(image_dir, csv_path, 0.8, batch_size, resize=(resizeh, resizew))

# 定义训练组件
model = YOLOv1_resnet()
state_dict = torch.load('./weights_use/resnet34-333f7ec4.pth')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
loss_fn = Yolov1_loss()
optimizer = Adam(model.parameters(), learning_rate)

# 训练网络
stringsplot()
trainlosses = []
train_operation = ""
st = datetime.now()

for epoch in range(epoches):
    trainloss = 0.
    print("Epoch:{}/{}".format(epoch + 1, epoches))
    train_operation=train_operation+"Epoch:{}/{}\n".format(epoch + 1, epoches)
    for item in tqdm(iter(train_dl)):
        X_train, y_train = item
        X_train = X_train.to(device)
        outputs = model(X_train)
        y_train = y_train.permute(0, 3, 1, 2).to(device)
        # print(outputs.shape,y_train.shape) # torch.Size([32, 30, 7, 7]) torch.Size([32, 7, 7, 30])
        loss = loss_fn(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        trainloss += loss.data
        optimizer.step()
    print("train loss:{}.".format(trainloss))
    train_operation = train_operation + "train loss:{}.\n".format(trainloss)
    trainlosses.append(trainloss.data)
et = datetime.now()
duration = et - st
log_generator("yolov1_voc2007", trainlosses, optimizer, model, epoches, learning_rate, batch_size, train_operation,
              log_dir, duration=duration)
