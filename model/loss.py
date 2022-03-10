import torch
from torch import nn
from utils.bbox_utils import get_iou, bbox_format_transform


class Yolov1_loss(nn.Module):
    """
    这个类是根据yolov1训练机制编写的，包含了四个主要loss：
    1.使用MSE对分类进行loss计算;
    2.使用bbox回归对有物体的边界框进行损失计算;
    3.对无物体的边界框进行损失计算——只算置信度;
    4.使用MSE对bbox置信度进行损失计算；
    
    inputs:pred 和 labels具有相同的shape:(N,30,7,7)
    """

    def __init__(self, object_loss_weight=5., noobject_loss_weights=0.1):
        super(Yolov1_loss, self).__init__()
        self.object_loss_weight = object_loss_weight
        self.noobject_loss_weights = noobject_loss_weights

    def forward(self, pred, labels):
        """
        :param pred: (batchsize,30,7,7)的网络输出数据
        :param labels: (batchsize,30,7,7)的样本标签数据
        :return: 当前批次样本的平均损失值(loss-sum/batch_sie)
        """
        # 限定以后后面会使用到的各种变量:训练批次大小、bbox个数、类别个数、预测的bbox、标签bbox等
        batch_size = pred.shape[0]
        bbox_num = 2
        classes_num = 20
        grid_size = (7, 7)  # h,w
        confidence_loss = 0.
        classification_loss = 0.
        bbox_loss = 0.
        noobj_bbox_loss = 0.

        for n in range(batch_size):
            for i in range(7):
                for j in range(7):
                    # 如果有标签，计算四种损失值
                    if labels[n, 4, i, j] == 1:
                        # 首先要得到iou
                        pred_bbox1 = bbox_format_transform(pred[n, 0:4, i, j], mode=2)
                        pred_bbox2 = bbox_format_transform(pred[n, 5:9, i, j], mode=2)
                        label_bbox = bbox_format_transform(labels[n, 0:4, i, j], mode=2)
                        iou1 = get_iou(pred_bbox1, label_bbox)
                        iou2 = get_iou(pred_bbox2, label_bbox)
                        # print(iou1,iou2)
                        # 选取最接近真实值的bbox
                        if iou1 >= iou2:
                            confidence_loss += (pred[n, 4, i, j] - labels[n, 4, i, j]) ** 2
                            noobj_bbox_loss += self.noobject_loss_weights * (pred[n, 9, i, j] - iou1) ** 2
                            bbox_loss += self.object_loss_weight * (torch.sum(
                                (pred[n, 0, i, j] - labels[n, 0, i, j]) ** 2 + (
                                            pred[n, 1, i, j] - labels[n, 1, i, j]) ** 2)
                                                                    + torch.sum(
                                        (pred[n, 2, i, j].sqrt() - labels[n, 2, i, j].sqrt()) ** 2 + (
                                                    pred[n, 3, i, j].sqrt() - labels[n, 3, i, j].sqrt()) ** 2))
                        else:
                            confidence_loss += (pred[n, 9, i, j] - labels[n, 9, i, j]) ** 2
                            noobj_bbox_loss += self.noobject_loss_weights * (pred[n, 4, i, j] - iou2) ** 2
                            bbox_loss += self.object_loss_weight * (torch.sum(
                                (pred[n, 5, i, j] - labels[n, 5, i, j]) ** 2 + (
                                            pred[n, 6, i, j] - labels[n, 6, i, j]) ** 2)
                                                                    + torch.sum(
                                        (pred[n, 7, i, j].sqrt() - labels[n, 7, i, j].sqrt()) ** 2 + (
                                                    pred[n, 8, i, j].sqrt() - labels[n, 8, i, j].sqrt()) ** 2))
                        # 不管选哪个，都要把类别损失给算了——使用MSE
                        classification_loss += torch.sum((pred[n, 10:, i, j] - labels[n, 10:, i, j]) ** 2)
                    # 如果没有标签，仅仅计算confidence_loss与0之前的差距，即没有物体状态下的noobject_loss
                    else:
                        noobj_bbox_loss += self.noobject_loss_weights * (
                                    (pred[n, 4, i, j]) ** 2 + (pred[n, 9, i, j]) ** 2)
        # bbox loss 曾出现了nan的情况,是因为sqrt负数为nan
        yolo_loss = bbox_loss + noobj_bbox_loss + confidence_loss + classification_loss
        yolo_loss /= batch_size

        return yolo_loss

# class Yolov1_loss(nn.Module):

#     """
#     这个类是根据yolov1训练机制编写的，包含了四个主要loss：
#     1.使用MSE对分类进行loss计算;
#     2.使用bbox回归对有物体的边界框进行损失计算;
#     3.对无物体的边界框进行损失计算——只算置信度;
#     4.使用MSE对bbox置信度进行损失计算；

#     inputs:pred 和 labels具有相同的shape:(N,30,7,7)
#     """
#     def __init__(self):
#         super(Yolov1_loss,self).__init__()

#     def forward(self, pred, labels):
#         """
#         :param pred: (batchsize,30,7,7)的网络输出数据
#         :param labels: (batchsize,30,7,7)的样本标签数据
#         :return: 当前批次样本的平均损失值(loss-sum/batch_sie)
#         """
#         # 限定以后后面会使用到的各种变量:训练批次大小、bbox个数、类别个数、预测的bbox、标签bbox等
#         batch_size=pred.shape[0]
#         confidence_loss=0.
#         classification_loss=0.
#         bbox_loss=0.
#         noobj_bbox_loss=0.
#         num_gridx, num_gridy = labels.size()[-2:] 

#         for n in range(batch_size):
#             for i in range(7):
#                 for j in range(7):
#                     # 如果有标签，计算四种损失值
#                     if labels[n,4,i,j]==1:
#                         # 首先要得到iou
#                         pred_bbox1 = ((pred[n,0,i,j]+j)/num_gridx - pred[n,2,i,j]/2,(pred[n,1,i,j]+j)/num_gridy - pred[n,3,i,j]/2,
#                                             (pred[n,0,i,j]+j)/num_gridx + pred[n,2,i,j]/2,(pred[n,1,i,j]+i)/num_gridy + pred[n,3,i,j]/2)
#                         pred_bbox2 = ((pred[n,5,i,j]+j)/num_gridx - pred[n,7,i,j]/2,(pred[n,6,i,j]+j)/num_gridy - pred[n,8,i,j]/2,
#                                             (pred[n,5,i,j]+j)/num_gridx + pred[n,7,i,j]/2,(pred[n,6,i,j]+i)/num_gridy + pred[n,8,i,j]/2)
#                         label_bbox = ((labels[n,0,i,j]+j)/num_gridx - labels[n,2,i,j]/2,(labels[n,1,i,j]+i)/num_gridy - labels[n,3,i,j]/2,
#                                         (labels[n,0,i,j]+j)/num_gridx + labels[n,2,i,j]/2,(labels[n,1,i,j]+i)/num_gridy + labels[n,3,i,j]/2)
#                         iou1=get_iou(pred_bbox1,label_bbox)
#                         iou2=get_iou(pred_bbox2,label_bbox)
#                         # 选取最接近真实值的bbox
#                         if iou1>=iou2:
#                             confidence_loss += (pred[n,4,i,j]-labels[n,4,i,j])**2
#                             noobj_bbox_loss += (pred[n,9,i,j]-iou1)**2
#                             # print(   (pred[n,0,i,j]-labels[n,0,i,j])**2 + (pred[n,1,i,j]-labels[n,1,i,j])**2 )
#                             bbox_loss += (  torch.sum(  (pred[n,0,i,j]-labels[n,0,i,j])**2 + (pred[n,1,i,j]-labels[n,1,i,j])**2 ) 
#                             + torch.sum( (pred[n,2,i,j].sqrt() - labels[n,2,i,j].sqrt())**2 + (pred[n,3,i,j].sqrt() - labels[n,3,i,j].sqrt())**2 )  )
#                         else:
#                             confidence_loss += (pred[n,9,i,j]-labels[n,9,i,j])**2
#                             noobj_bbox_loss += (pred[n,4,i,j]-iou2)**2
#                             print(  (pred[n,5,i,j]-labels[n,5,i,j])**2 + (pred[n,6,i,j]-labels[n,6,i,j])**2 )
#                             bbox_loss += (  torch.sum(  (pred[n,5,i,j]-labels[n,5,i,j])**2 + (pred[n,6,i,j]-labels[n,6,i,j])**2 ) 
#                             + torch.sum( (pred[n,7,i,j].sqrt() - labels[n,7,i,j].sqrt())**2 + (pred[n,8,i,j].sqrt() - labels[n,8,i,j].sqrt())**2 )  )
#                         # 不管选哪个，都要把类别损失给算了——使用MSE
#                         classification_loss += torch.sum((pred[n,10:,i,j]-labels[n,10:,i,j])**2)
#                     # 如果没有标签，仅仅计算confidence_loss与0之前的差距，即没有物体状态下的noobject_loss
#                     else:
#                         noobj_bbox_loss += ( (pred[n,4,i,j])**2 + (pred[n,9,i,j])**2 )
#         # bbox loss 曾出现了nan的情况,是因为sqrt负数为nan
#         yolo_loss=bbox_loss+noobj_bbox_loss+confidence_loss+classification_loss
#         yolo_loss/=batch_size

#         return yolo_loss


# ---------------------------------------------------------------------------------------------------------
# 踩坑日记1：
# bbox_format_transform函数涉及到了浮
# 点数除法运算在backward过程中，由于p-
# ytorch的自动求导不支持浮点数除法运算
# 求导而报错.
# 报错信息：
# Traceback (most recent call last):
#   File "f:/PycharmProjects/yolov1/main.py", line 42, in <module>
#     trainmodel(train_dl,model,epoches,learning_rate,batch_size,optimizer,loss_fn,device)
#   File "f:\PycharmProjects\yolov1\train.py", line 23, in trainmodel
#     loss.backward()
#   File "C:\Users\22704\Anaconda3\envs\Pytorch\lib\site-packages\torch\tensor.py", line 221, in backward
#     torch.autograd.backward(self, gradient, retain_graph, create_graph)
#   File "C:\Users\22704\Anaconda3\envs\Pytorch\lib\site-packages\torch\autograd\__init__.py", line 130, in backward
#     Variable._execution_engine.run_backward(
# RuntimeError: derivative for floor_divide is not implemented
# 踩坑日记2：
# bbox loss 曾出现了nan的情况,是因为tensor中元素<0时，sqrt会变为nan
# 求导而报错.
# ---------------------------------------------------------------------------------------------------------

# # Debug
# loss_fn_test=Yolov1_loss_test()
# loss_fn=Yolov1_loss()
# loss_fn_1=Loss_yolov1()

# test_pred=torch.randn((32,30,7,7))
# test_label=torch.randn((32,30,7,7))

# loss1=loss_fn_test(test_pred,test_label)
# loss2=loss_fn(test_pred,test_label)
# loss3=loss_fn_1(test_pred,test_label)
# print(loss1,loss2,loss3)
