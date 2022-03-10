import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from configparser import ConfigParser

def getconfigparser(config_path):
        parser = ConfigParser()
        parser.read(config_path)
        return parser
def log_generator(train_theme_name,
                    tags,
                    optimizer,
                    model,
                    epochs,
                    lr,
                    batchsize,
                    training_opeartion,
                    log_save_dir,duration):
        nowtime = datetime.now()
        year = str(nowtime.year)
        month = str(nowtime.month)
        day = str(nowtime.day)
        hour = str(nowtime.hour)
        minute = str(nowtime.minute)
        second = str(nowtime.second)
        nowtime_strings = year + '/' + month + '/' + day + '/' + hour + ':' + minute + ':' + second
        workplace_path = os.getcwd()
        content = """
    Theme:{}\n
    batch size:{}\n
    learning rate:{}\n
    epochs:{}\n
    Date:{}\n
    workplace:{}\n
    Time used:{}\n
    Optimizer:\n{}\n
    Model:\n{}\n,
    Train:\n{}\n
        """.format(
            train_theme_name,
            batchsize,
            lr,
            epochs,
            nowtime_strings,
            workplace_path,
            duration,
            str(optimizer),
            str(model),
            training_opeartion)
        exp_name = 'exp-{}_{}_{}_{}_{}_{}'.format(
            train_theme_name,
            year, month, day,
            hour, minute, second)
        exp_path=log_save_dir+'/'+exp_name
        if os.path.exists(exp_path)==0:
            os.makedirs(exp_path)
        log_name = '{}_{}_{}_{}_{}_{}.log'.format(
            train_theme_name,
            year, month, day,
            hour, minute, second)
        file = open(exp_path + '/' + log_name, 'w', encoding='utf-8')
        file.write(content)
        file.close()
        torch.save(model.state_dict(), exp_path + '/' + '{}_{}_{}_{}_{}_{}.pth'.format(
            train_theme_name,
            year, month, day, hour,
            minute, second
        ))
        log_plot(epochs,tags,save_fig_path=exp_path + '/' + '{}_{}_{}_{}_{}_{}.jpg'.format(
            train_theme_name,
            year, month, day, hour,
            minute, second
        ))
        print("Training log has been saved into path:{}.".format(exp_path))


def log_plot(epoches,tags,save_fig_path=None):
    plt.figure()
    tags=np.array([x.cpu().numpy() for x in tags])
    plt.plot(np.arange(1,epoches+1),tags,'r')
    plt.xlabel("epoch")
    plt.ylabel("train Loss")
    plt.grid()
    if save_fig_path:
        plt.savefig(save_fig_path)
        
    
def stringsplot():
    content=r"""
======================================================================================
                
   \\      //           /=======\               ||                      /=======\ 
    \\    //           /         \              ||                     /         \  
     \\  //           /           \             ||                    /           \    
      \\//           |             |            ||                   |             | 
       ||            |             |            ||                   |             | 
       ||            |             |            ||                   |             | 
       ||            |             |            ||                   |             | 
       ||            \             /            ||                   \             /                     
       ||             \           /             ||                    \           /                                         
       ||              \=========/              ===============        \=========/ 

                                                            (Create by  BDS_jhb)            
======================================================================================    
    """
    print(content)
    