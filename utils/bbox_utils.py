import numpy as np
import torch
from utils.train_utils import getconfigparser
import matplotlib.pyplot as plt

def bbox_format_transform(bbox, mode=1):
    """
    bbox kind:(cls,x,x,x,x)
    :param mode:
    if mode=1,convert (xa,ya,xb,yb) to (x,y,w,h)
    else if mode=2 convert (x,y,w,h) to (xa,ya,xb,yb)
    :return:another kind of bbox
    """
    if mode == 1:
        return convert_bboxm_to_bboxcen(bbox)
    if mode == 2:
        return convert_bboxcen_to_bboxm(bbox)
    else:
        TypeError("Please type in right mode")


def convert_bboxm_to_bboxcen(bbox):
    """
    convert (xa,ya,xb,yb) to (x,y,w,h)
    """
    cls = bbox[0]
    bbox_copy = bbox[1:]
    x = (bbox_copy[0] + bbox_copy[2]) / 2
    y = (bbox_copy[1] + bbox_copy[3]) / 2
    w = (bbox_copy[2] - bbox_copy[0])
    h = (bbox_copy[3] - bbox_copy[1])

    return [cls, x, y, w, h]


def convert_bboxcen_to_bboxm(bbox):
    """
    convert (x,y,w,h) to (xa,ya,xb,yb)
    """
    cls = bbox[0]
    bbox_copy = bbox[1:]
    w = bbox_copy[-2]
    h = bbox_copy[-1]
    xa = bbox_copy[0] - w / 2
    xb = bbox_copy[0] + w / 2
    ya = bbox_copy[1] - h / 2
    yb = bbox_copy[1] + h / 2

    return [cls, xa, ya, xb, yb]


def reshape_bbox(bbox,reshape_scale,mode=1):
        """
        :param bbox:input bbox  like ([cls,x,x,x,x])
        :param reshape_scale:a tuple with 2 elements.(height_scale,width_scale)

        height_scale=\frac{img-height_a}{img-height_b}=\frac{bbox-y_a}{bbox-y_b}

        width_scale=\frac{img-width_a}{img-height_b}=\frac{bbox-x_a}{bbox-x_b}

        (a:after,b:before)

        :param mode:
        if mode=1 bbox is like:(xa,ya,xb,yb)
        elif mode=2 bbox is like:(x,y,w,h)
        else:
            raise type error.
        :return:reshaped bbox
        """
        if mode==1:
            return reshape_bbox_xymodel(bbox,reshape_scale)
        if mode==2:
            return reshape_bbox_whmodel(bbox,reshape_scale)
        else:
            TypeError("Please type in right mode")


def reshape_bbox_xymodel(bbox,reshape_scale):
        cls,xa,ya,xb,yb=bbox[0],bbox[1],bbox[2],bbox[3],bbox[4]
        xa*=reshape_scale[1]
        xb*=reshape_scale[1]
        ya*=reshape_scale[0]
        yb*=reshape_scale[0]
        return [cls,xa,ya,xb,yb]


def reshape_bbox_whmodel(bbox,reshape_scale):
        cls=bbox[0]
        x,y,w,h=bbox[1:5]
        x*=reshape_scale[1]
        y*=reshape_scale[0]
        w*=reshape_scale[1]
        h*=reshape_scale[0]
        return [cls,x,y,w,h]


def get_iou(pred, label):
    x1, a1 = pred[0], label[0]
    y1, b1 = pred[1], label[1]
    x2, a2 = pred[2], label[2]
    y2, b2 = pred[3], label[3]
    ax = max(x1, a1)  # 相交区域左上角横坐标
    ay = max(y1, b1)  # 相交区域左上角纵坐标
    bx = min(x2, a2)  # 相交区域右下角横坐标
    by = min(y2, b2)  # 相交区域右下角纵坐标

    area_N = (x2 - x1) * (y2 - y1)
    area_M = (a2 - a1) * (b2 - b1)

    w = bx - ax
    h = by - ay
    # 假设相交，那么按道理算出来的相交区域
    # 的w和h如果只要有一个是小于0的，那么
    # 就不成立(反证法)
    if w <= 0 or h <= 0:
        return 0
    area_X = w * h
    return area_X / (area_N + area_M - area_X)


def bbox_normalization(bbox, size):
    """
    Normalize bbox's data between 0-1.
    
    bbox:must be the sequence like (cls,x,y,w,b)
    
    img_origin_size:a tuple like (h,w)
    """
    bbox_copy = bbox.copy()
    dh = 1. / size[0]
    dw = 1. / size[1]
    bbox_copy[1] = dw*bbox[1]
    bbox_copy[2] = dh*bbox[2]
    bbox_copy[3] = dw*bbox[3]
    bbox_copy[4] = dh*bbox[4]

    return bbox_copy


def convert_yolov1_label(bboxes):
    """
    将多个bbox((-1,5))转换为(30,7,7)的网络输出形式
    输入：(-1,5)的bboxes的list或者numpy. format:(bbox number,bbox)
    输出：(7,7,30)的yolov1格式标签  format:(height,width,information_number(30))
    tips:长度为30的信息框=[相对横坐标，相对纵坐标，归一化宽，归一化高，置信度，20个类别概率预测]
    """
    label = np.zeros((7, 7, 30))    # 7,7,30+1_(这个网格中是否已经有中心点了)
    grid_size = (1. / 7, 1. / 7)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        grid_x = int(bbox[1] / grid_size[1])  # 中心点所坐落的网格横下标——注意是下标哦！
        grid_y = int(bbox[2] / grid_size[0])  # 中心点所坐落的网格纵下标
        # (将bbox中心点坐标与网格左上角坐标之差)再除以网格大小——>bbox的中心点相对于grid的位置
        # 脱去括号就是:中心点坐标/网格大小 - 左上角坐标/网格大小 = 中心点坐标/网格大小 - 网格
        # 在网格坐标系中的坐标也就是grid_x,grid_y(tips:/是向下取整)
        grid_px = bbox[1] / grid_size[1] - grid_x
        grid_py = bbox[2] / grid_size[0] - grid_y

        label[grid_x, grid_y, 0:5] = np.array([grid_px, grid_py, bbox[3], bbox[4], 1])  # 将标签的一个grid中的两个检测框都填充为目标的位置和置信度
        label[grid_x, grid_y, 5:10] = np.array([grid_px, grid_py, bbox[3], bbox[4], 1])

        label[grid_x, grid_y, 10 + bbox[0]] = 1  # 分类填写

    return label


def convert_yolov1label_to_bbox(outputs):
    """
    将yolov1网络的输出转换为(7,7,30)后输入该函数，得到bboxes(98,25)。
    输入：(7,7,30)的tensor
    输出：(98,25)的tensor-(x,y,x,y)格式
    tips:98=2*(7*7)，也就是说98个信息框中，i,i+1是同一个方格的两个检测框(0<=i<49)
    """
    if outputs.shape[0] != 7 or outputs.shape[1] != 7:
        raise ValueError("Wrong input's shape:({},{}).".format(outputs.shape[0], outputs.shape[1]))
    bboxes = torch.zeros((98, 25))
    for i in range(7):
        for j in range(7):
            # 将每一个网格的预测框的(px,py,w,h)转换为(xa,ya,xb,yb)
            # 第一个检测框
            # print(outputs[i, j, 0:4])
            bboxes[2 * (i * 7 + j), 0:4] = torch.Tensor([
                (outputs[i, j, 0] + j) / 7,
                (outputs[i, j, 1] + i) / 7,
                outputs[i, j, 2],
                outputs[i, j, 3]
            ])
            bboxes[2 * (i * 7 + j), 4] = outputs[i, j, 4]  # 置信度转换
            bboxes[2 * (i * 7 + j), 5:] = outputs[i, j, 10:]  # 分类转换
            # 第二个检测框
            bboxes[2 * (i * 7 + j) + 1, 0:4] = torch.Tensor([
                (outputs[i, j, 5] + j) / 7,
                (outputs[i, j, 6] + i) / 7,
                outputs[i, j, 7],
                outputs[i, j, 8]
            ])
            bboxes[2 * (i * 7 + j) + 1, 4] = outputs[i, j, 9]
            bboxes[2 * (i * 7 + j) + 1, 5:] = outputs[i, j, 10:]
    return bboxes


def NMS(bboxes, class_confidence_threshold=0.1,iou_threshold=0.3):
    """
    输入decode操作之后的(98,25)的bbox矩阵，这个bbox矩阵的每一条sample的前四个是(xa,ya,xb,yb)位置信息，第五个是置信度，后20个位类别预测概率。
    输入：(98,25)的tensor
    输出：(98,6)的tensor   (分类信息，位置信息(4个)，置信度)
    """
    sample_num=bboxes.size()[0]
    class_prob=bboxes[:,5:].clone() # (98,20)
    confidence=bboxes[:,4].clone().unsqueeze(1).expand_as(class_prob)    #(98,)
    # 使用confidence乘以位置信息就可以得到物体的存在与否、物体的位置信息
    bbox_weight=confidence*class_prob
    # 过滤操作
    bbox_weight[bbox_weight<=class_confidence_threshold]=0
    # 从大到小排列每一个类别的概率并返回其下标
    for kind in range(20):
        rank=torch.sort(bbox_weight[:,kind],descending=True).indices
        for i in range(98): # 开始遍历bbox中weight最大的，并用这个最大的bbox遍历其余未被筛选掉的框
            if bbox_weight[rank[i],kind]!=0:
                for j in range(i+1,98):
                    if bbox_weight[rank[j],kind]!=0:
                        iou=get_iou(bboxes[rank[i],0:4],bboxes[rank[j],0:4])
                        if iou>iou_threshold:   # 过滤操作
                            bbox_weight[rank[j],kind]=0
    bboxes=bboxes[torch.max(bbox_weight,dim=1).values>0]    # 筛选出weight大于0的
    # 存放最后的信息
    outputs=torch.ones((sample_num,6))
    outputs[:,1:5]=bboxes[:,0:4]    # 存放位置信息
    outputs[:,0]=torch.argmax(bboxes[:,5:],dim=1).int()  # 存放bbox对应的类别信息
    outputs[:,5]=torch.max(bbox_weight,dim=1).values    # 存放置信度权值
    return outputs
