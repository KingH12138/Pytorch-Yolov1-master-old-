from model.model import YOLOv1_resnet
from utils.train_utils import getconfigparser
from utils.data_utils import DrawBBox
from utils.bbox_utils import convert_yolov1label_to_bbox
from torchvision import transforms
from PIL import Image
import torch

COLOR = [(255, 0, 0), (255, 125, 0), (255, 255, 0), (255, 0, 125), (255, 0, 250),
         (255, 125, 125), (255, 125, 250), (125, 125, 0), (0, 255, 125), (255, 0, 0),
         (0, 0, 255), (125, 0, 255), (0, 125, 255), (0, 255, 255), (125, 125, 255),
         (0, 255, 0), (125, 255, 125), (255, 255, 255), (100, 100, 100), (0, 0, 0), ]
# 读取配置参数
parser = getconfigparser('./config/config.cfg')
weight_path = parser.get('predict-config', 'weight_path')
test_path = parser.get('predict-config', 'predict_path')
resizeh = parser.getint('dataset-config', 'resizeh')
resizew = parser.getint('dataset-config', 'resizew')
CLASSES = parser.get("dataset-config", 'classes').split(',')

# 测试图读取
resize = (resizeh,resizew)
transformer = transforms.Compose([
    transforms.Resize(resize),
    transforms.ToTensor(),
])
img_file = Image.open(test_path)
h, w = img_file.height, img_file.width
img_file.resize((resizeh,resizew))
origin_size = (img_file.height, img_file.width)
x_test = transformer(img_file).reshape((1, -1, resizeh, resizew))
# 前向传播
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLOv1_resnet().to(device)
model.load_state_dict(torch.load(weight_path))
x_test = x_test.to(device)
prediction = model(x_test).cpu().detach() # (1,30,7,7)
prediction = prediction.reshape((30,7,7))
prediction = prediction.permute((1,2,0))  # 转换为(7,7,30)
# 转成正常标签形式
prediction = convert_yolov1label_to_bbox(prediction).numpy()


prediction[:, 0] *= w
prediction[:, 1] *= h
prediction[:, 2] *= w
prediction[:, 3] *= h

prediction = prediction.tolist()
out = []
for bbox in prediction:
    if bbox[4] >= 0.1:
        out.append([round(max(bbox[5:]), 2)] + bbox[0:4])

DrawBBox(img_file, out, fig_save_path='./imgs/pred1.jpg')






