from tkinter.ttk import Progressbar
from utils.data_utils import getvoccsv
from utils.train_utils import log_generator, log_plot
from data.dataset import VOCDataloader
from model.model import YOLOv1
from model.loss import Yolov1_loss
from torch.optim import Adam
from tkinter import *
import torch
from datetime import datetime
import pandas as pd


# ------------------------------------------------------------------------
# 主要功能界面
def csv_txt_gener(csv_dir, image_dir, xml_dir, txt_dir):
    getvoccsv(xml_dir, csv_dir, txt_dir, image_dir)


def open_train_window():
    root_train = Toplevel()
    root_train.geometry('1000x500')
    root_train.title("训练")

    divider = Label(root_train, bg='black')
    divider.place(x=500, y=0, width=4, height=500)
    train_tip = Label(root_train, text="训练界面", bg='black', fg='white', font=('宋体', 15))
    train_tip.place(x=0, y=0, width=100, height=40)
    dataset_tip = Label(root_train, text="处理数据集:", justify='left', font=('宋体', 12))
    dataset_tip.place(x=0, y=50, width=150, height=40)

    data_inp1 = Entry(root_train)
    data_inp2 = Entry(root_train)
    data_inp3 = Entry(root_train)
    data_inp4 = Entry(root_train)
    data_lb1 = Label(root_train, fg='black', text='csv path:')
    data_lb2 = Label(root_train, fg='black', text='image dir:')
    data_lb3 = Label(root_train, fg='black', text='xml dir:')
    data_lb4 = Label(root_train, fg='black', text='txt dir:')
    txt_gener = Button(root_train, text="开始处理",
                       command=lambda: csv_txt_gener(data_inp1.get(), data_inp2.get(), data_inp3.get(),
                                                     data_inp4.get()))
    ready_tips = Label(root_train, text="已经生成相关文件，跳过生成操作", fg="red")
    data_lb1.place(x=0, y=100, width=100, height=20)
    data_lb2.place(x=0, y=125, width=100, height=20)
    data_lb3.place(x=0, y=150, width=100, height=20)
    data_lb4.place(x=0, y=175, width=100, height=20)

    data_inp1.place(x=100, y=100, width=200, height=20)
    data_inp2.place(x=100, y=125, width=200, height=20)
    data_inp3.place(x=100, y=150, width=200, height=20)
    data_inp4.place(x=100, y=175, width=200, height=20)

    txt_gener.place(x=350, y=125, width=100, height=40)

    divder2 = Label(root_train, bg='black')
    divder2.place(relx=0, rely=0.5, height=5, relwidth=0.5)

    def train(epoches, batch_size, lr, log_dir, weight_dir, train_data_precent):
        train_dataloader, _ = VOCDataloader(data_inp2.get(), data_inp1.get(), train_data_precent, batch_size)

        trainlosses = []
        train_operation = ""

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loss_fn = Yolov1_loss()
        model_object = YOLOv1()
        optimizer = Adam(model_object.parameters(), lr)
        model_object = model_object.to(device)
        sample_num = len(pd.read_csv(data_inp1.get(), encoding='utf-8'))

        epochlabel = Label(root_train, '', justify=LEFT)
        progbar = Progressbar(root_train, maximum=sample_num)
        progbar.place(x=150, y=450, width=200, height=20)
        progbar.value = 0
        epochlabel.place(x=150, y=430, width=200, height=20)

        st = datetime.now()
        for epoch in range(epoches):
            train_operation = train_operation + "Epoch:{}/{}\n".format(epoch + 1, epoches)
            trainloss = 0.
            train_operation = train_operation + "Train period...\n"
            i = 0
            for item in iter(train_dataloader):
                epochlabel.config(text="Epoch:{}/{} batch:{}/{}".format(epoch + 1, epoches, i + 1, sample_num))
                progbar.value = (i + 1)
                root_train.update()
                X_train, y_train = item
                X_train = X_train.to(device)
                y_train = y_train.to(device)
                i += 1
                outputs = model_object(X_train)
                loss = loss_fn(outputs, y_train)
                optimizer.zero_grad()
                loss.backward()
                trainloss += loss.data
                trainlosses.append(loss.data)
                optimizer.step()
        et = datetime.now()
        duration = et - st
        log_generator("yolov1_voc2007", optimizer, model_object, epoches, lr, batch_size,
                      train_operation, duration=duration, log_save_dir=log_dir,
                      weight_save_dir=weight_dir)
        log_plot(epoches, trainlosses, save_fig_path='./imgs/img.jpg')
        global img
        img = PhotoImage('./imgs/img.jpg')

    data_inp5 = Entry(root_train)
    data_inp6 = Entry(root_train)
    data_inp7 = Entry(root_train)
    data_inp8 = Entry(root_train)
    data_inp9 = Entry(root_train)
    data_inp10 = Entry(root_train)

    data_lb5 = Label(root_train, fg='black', text='epoches:')
    data_lb6 = Label(root_train, fg='black', text='batch size:')
    data_lb7 = Label(root_train, fg='black', text='learning rate:')
    data_lb8 = Label(root_train, fg='black', text='log dir:')
    data_lb9 = Label(root_train, fg='black', text='weight dir:')
    data_lb10 = Label(root_train, fg='black', text='train data percent:')
    data_lb5.place(x=0, y=275, width=100, height=20)
    data_lb6.place(x=0, y=300, width=100, height=20)
    data_lb7.place(x=0, y=325, width=100, height=20)
    data_lb8.place(x=0, y=350, width=100, height=20)
    data_lb9.place(x=0, y=375, width=100, height=20)
    data_lb10.place(x=0, y=400, width=100, height=20)

    data_inp5.place(x=110, y=275, width=210, height=20)
    data_inp6.place(x=110, y=300, width=210, height=20)
    data_inp7.place(x=110, y=325, width=210, height=20)
    data_inp8.place(x=110, y=350, width=210, height=20)
    data_inp9.place(x=110, y=375, width=210, height=20)
    data_inp10.place(x=110, y=400, width=210, height=20)

    train_but = Button(
        root_train, text="开始训练",
        command=lambda: train(int(data_inp5.get()), int(data_inp6.get()), float(data_inp7.get()),
                              data_inp8.get(), data_inp9.get(), float(data_inp10.get())
                              )
    )
    train_but.place(x=350, y=325, width=100, height=40)

    summary_tip = Label(root_train, text="结果概要", bg='black', fg='white', font=('宋体', 15))
    summary_tip.place(x=504, y=0, width=100, height=40)

    root_train.mainloop()


# ------------------------------------------------------------------------
# yolov1简介
def open_vision_window():
    root_vision = Toplevel()
    root_vision.title("关于YOLOV1")
    root_vision.geometry('500x500')
    root_vision.mainloop()


# ------------------------------------------------------------------------
# 开发人员简介
def open_about_window():
    root_about = Toplevel()
    root_about.title("关于我们")
    root_about.geometry('500x500')
    root_about.mainloop()


# ------------------------------------------------------------------------
# 预测界面
def open_predict_window():
    root_predict = Toplevel()
    root_predict.title("预测界面")
    root_predict.geometry('500x500')
    root_predict.mainloop()


# ------------------------------------------------------------------------
# 主菜单

root = Tk()
root.title("YOLOV1")
root.geometry("300x350")

top = Label(root, text="YOLOV1", font=60, fg='black')
top.place(x=110, y=20, width=100, height=50)

enter = Button(root, text="训练界面", command=open_train_window)
exiter = Button(root, text="预测界面", command=open_predict_window)
vision = Button(root, text="关于YOLOV1", command=open_vision_window)
about = Button(root, text="关于我们", command=open_about_window)
predict = Button(root, text="退出系统", command=root.destroy)

enter.place(x=100, y=80, width=120, height=25)
exiter.place(x=100, y=120, width=120, height=25)
vision.place(x=100, y=160, width=120, height=25)
about.place(x=100, y=200, width=120, height=25)
predict.place(x=100, y=240, width=120, height=25)

verson = Label(root, text="verson 1.0", font=5)
verson.place(x=100, y=280, width=100, height=25)

root.mainloop()
