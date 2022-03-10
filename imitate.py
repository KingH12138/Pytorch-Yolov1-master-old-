import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from utils.data_utils import txt_to_bboxinfo, DrawBBox
from utils.bbox_utils import bbox_format_transform, bbox_normalization, convert_yolov1_label, reshape_bbox,convert_yolov1label_to_bbox
from utils.train_utils import getconfigparser

parser = getconfigparser('./config/config.cfg')
CLASSES = parser.get("dataset-config", 'classes').split(',')
df = pd.read_csv('./object.csv')

resize = (448, 448)  # 尺度变换后的大小
item = 12  # 样本序号
transformer = transforms.Compose([
    transforms.Resize(resize),
    transforms.ToTensor(),
])

# 图片的处理(reshape+totensor)
img_file = Image.open(df['path'][item])  # 图片PIL对象
h, w = img_file.height, img_file.width
x_train = transformer(img_file)
# bbox处理(reshape,format_trans,normalization)
bboxes_init = txt_to_bboxinfo(df['object_txt_path'][item])  # 位置信息 :format(xa,ya,xb,yb)
scale = (resize[0] / h, resize[1] / w)
bboxes_reshape = [reshape_bbox(bbox, scale) for bbox in bboxes_init]
# DrawBBox(img_file.resize(resize),bboxes_reshape,mode=1)  #  用drawbbox函数检验一下reshape的bbox和图片，发现没有问题
bboxes_trans = [bbox_format_transform(bbox, mode=1) for bbox in bboxes_reshape]
# DrawBBox(img_file.resize(resize),grid_num=(7,7),bboxes=bboxes_trans, mode=2,fig_save_path='./test.jpg')  #  用drawbbox函数检验一下格式转换的bbox和图片，发现没有问题
bboxes_normal = [bbox_normalization(bbox, resize) for bbox in bboxes_trans]  # 归一化是为了减少计算难度并方便我们后续将其转为(px,py,dw,dh)的格式
# # 目前的bbox是(dx,dy,dw,dh)——dh,dw指的是bbox的长宽相对于图片的相对长度,dx,dy同理
for i in range(len(bboxes_normal)):
    bboxes_normal[i][0] = CLASSES.index(bboxes_normal[i][0])  # 标签数值化

y_train = convert_yolov1_label(bboxes_normal)  # (dx,dy,dw,dh) -> (px,py,dw,dh)    # 缺点所在：多个物体中心点落在同一个网格只能检测一个问题
y_train = torch.Tensor(y_train)


y_train = y_train.permute((2, 0, 1))

# 将预测值进行转换-开始解码

y_train = y_train.permute((1, 2, 0))

bboxes_p = convert_yolov1label_to_bbox(y_train) # (px,py,dw,dh) -> (dx,dy,dw,dh)


bboxes_p[:,0]=bboxes_p[:,0]*resize[1]
bboxes_p[:,1]=bboxes_p[:,1]*resize[0]
bboxes_p[:,2]=bboxes_p[:,2]*resize[1]
bboxes_p[:,3]=bboxes_p[:,3]*resize[0]

bboxes_p_out=[]
for bbox in bboxes_p:
    if bbox[4] == 1:
        classlist=list(bbox[5:].numpy())
        idlist=[]
        for i in range(len(classlist)):
            if classlist[i]==1:
                idlist.append(i)
        bboxes_p_out.append(idlist[:1]+list(bbox[0:4].numpy()))
DrawBBox(img_file.resize(resize), bboxes_p_out,mode=2)
