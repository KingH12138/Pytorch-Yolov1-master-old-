import os
from xml.dom.minidom import parse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def readvocxml(xml_path, image_dir):
    """

    The function can read single xml file and transform information of xml file into a list containing:
    the filename of the xml indicates(str),
    the filepath of image that xml indicates(a str.you need to give the dir which this image located in.Aka,the second parameter.)
    the depth,height,width of the Image(three int data.channel first),
    the annotated objects' infomation.(
        a 2D int list:
        [
            row1:[label_1,xmin_1,ymin_1,xmax_1,ymax_1]
            row2:[label_2,xmin_2,ymin_2,xmax_2,ymax_2]
            ....
            row_i[label_i,xmin_i,ymin_i,xmax_i,ymax_i]
        ]
    )

    Args:

    xml_path:singal xml file's path.

    image_dir:the image's location dir that xml file indicates.


    """
    tree = parse(xml_path)
    rootnode = tree.documentElement
    sizenode = rootnode.getElementsByTagName('size')[0]
    width = int(sizenode.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(sizenode.getElementsByTagName('height')[0].childNodes[0].data)
    depth = int(sizenode.getElementsByTagName('depth')[0].childNodes[0].data)

    name_node = rootnode.getElementsByTagName('filename')[0]
    filename = name_node.childNodes[0].data

    path = image_dir + '/' + filename

    objects = rootnode.getElementsByTagName('object')
    objects_info = []
    for object in objects:
        label = object.getElementsByTagName('name')[0].childNodes[0].data
        xmin = int(object.getElementsByTagName('xmin')[0].childNodes[0].data)
        ymin = int(object.getElementsByTagName('ymin')[0].childNodes[0].data)
        xmax = int(object.getElementsByTagName('xmax')[0].childNodes[0].data)
        ymax = int(object.getElementsByTagName('ymax')[0].childNodes[0].data)
        info = []
        info.append(label)
        info.append(xmin)
        info.append(ymin)
        info.append(xmax)
        info.append(ymax)
        objects_info.append(info)

    return [filename, path, depth, height, width, objects_info]


def DrawBBox(img, bboxes, grid_num=None, bbox_color='r', bbox_linewidth=5, content_color='red', fig_save_path=None,
             font_size=16, mode=1):
    """
    ---bbox---

    (left,upper)         (right,upper)
        ----------------------
        |                    |
        |       bbox         |
        |                    |
        ----------------------
        (left,lower)         (right,lower)
    Arg:
        img: a PIL object.

        bbox:a 2D int list
            [
                row1:[label_1,xmin_1,ymin_1,xmax_1,ymax_1]
                row2:[label_2,xmin_2,ymin_2,xmax_2,ymax_2]
                ....
                row_i[label_i,xmin_i,ymin_i,xmax_i,ymax_i]
            ]

        [xmin,ymin,xmax,ymax] is equal to [left,upper,right,lower].

        bbox_color:a str.It's bbox's bounding color.
        eg:
            'r' :red
            'b':blue
            'w':white
            'y':yellow
            'g':green

        content_color:content font's color.

        bbox_info:bounding box's some infomation that you want to display.

        fig_save_path:image with bbox displayed's saved path.

        font_size:content's font size.

    """
    h, w = img.height, img.width
    fig = plt.figure(figsize=(6, 6))
    axis = fig.gca()  # get figure's axis
    if len(bboxes[0]) == 5:
        for bbox in bboxes:
            if mode == 1:
                bboxer = plt.Rectangle(bbox[1:3], bbox[3] - bbox[1], bbox[4] - bbox[2], linewidth=bbox_linewidth,
                                       edgecolor=bbox_color, facecolor='none')
            if mode == 2:
                bboxer = plt.Rectangle((int(bbox[1] - bbox[3] / 2), int(bbox[2] - bbox[4] / 2)), bbox[3], bbox[4],
                                       linewidth=bbox_linewidth, edgecolor=bbox_color, facecolor='none')
            axis.add_patch(bboxer)
    elif len(bboxes[0]) == 4:
        for bbox in bboxes:
            if mode == 1:
                bboxer = plt.Rectangle(bbox[0:2], bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=bbox_linewidth,
                                       edgecolor=bbox_color, facecolor='none')
            if mode == 2:
                bboxer = plt.Rectangle((int(bbox[0] - bbox[2] / 2), int(bbox[1] - bbox[3] / 2)), bbox[2], bbox[3],
                                       linewidth=bbox_linewidth, edgecolor=bbox_color, facecolor='none')
            axis.add_patch(bboxer)
    plt.imshow(img)
    if grid_num:
        grid_width = (int(h / grid_num[0]), int(w / grid_num[1]))
        x = 0
        y = 0
        for i in range(grid_num[0] + 1):
            bboxer1 = plt.Rectangle((x, y + i * grid_width[0]), w, 5, linewidth=2, edgecolor='black', facecolor='black')
            axis.add_patch(bboxer1)
        x = 0
        y = 0
        for j in range(grid_num[1] + 1):
            bboxer1 = plt.Rectangle((x + j * grid_width[1], y), 5, h, linewidth=2, edgecolor='black', facecolor='black')
            axis.add_patch(bboxer1)
    if fig_save_path:
        plt.savefig(fig_save_path, bbox_inches='tight', pad_inches=0.0)


def get_bbox_txt(name, bbox_info, txt_save_dir):
    if os.path.exists(txt_save_dir) == 0:
        os.makedirs(txt_save_dir)
    f = open('{}/{}.txt'.format(txt_save_dir, name), encoding='utf-8', mode='w')
    for object_info in bbox_info:
        for info in object_info:
            f.write(str(info))
            f.write(' ')
        f.write('\n')
    txt_path = '{}/{}.txt'.format(txt_save_dir, name)
    f.close()
    return txt_path


def getvoccsv(xml_dir, csv_save_dir, txt_save_dir, image_dir):
    """
    得到csv和txt文件，并返回数据集样本数
    """
    col = ['filename', 'path', 'depth', 'height', 'width', 'object_txt_path']
    array = []

    for xml_name in os.listdir(xml_dir):
        xml_path = xml_dir + '/' + xml_name
        [filename, path, depth, height, width, objectinfo] = readvocxml(xml_path, image_dir=image_dir)
        object_txt_path = get_bbox_txt(filename[:-4], objectinfo, txt_save_dir)
        arr = [filename, path, depth, height, width, object_txt_path]
        array.append(arr)
    array = np.array(array)
    df = pd.DataFrame(array, columns=col)
    df.to_csv(csv_save_dir + '/' + 'object.csv', encoding='utf-8')


def txt_to_bboxinfo(txt_path):
    bbox_info = []
    f = open(txt_path, mode='r', encoding='utf-8')
    content = f.read()
    for info in content.split('\n'):
        info = info.split(' ')
        if len(info) == 1:
            continue
        label = info[0]
        xmin = int(info[1])
        ymin = int(info[2])
        xmax = int(info[3])
        ymax = int(info[4])
        bbox_info.append([label, xmin, ymin, xmax, ymax])
    return bbox_info
