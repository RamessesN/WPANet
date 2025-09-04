import torch

def _init():  # 初始化
    global parameter

    if torch.cuda.is_available():
        device_str = "cuda:0"
    elif hasattr(torch, "mps") and torch.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"

    parameter = {
        # net
        'channels': 30,
        'windowSize': 10,
        'out_features': [15, 20, 6, 8, 7],
        'depth': [[2,2,2],[2,2,2],2],
        # train
        'device': device_str,
        'lr': 0.00001,
        'epoch_nums': 10,
        'batch_size': 36,
        'num_workers': 2,
        'random_seed': 6,
        'visualization': False,
        'model_savepath': ['../model/Houston2013_model.pth',
                           '../model/Houston2018_model.pth',
                           '../model/Trento_model.pth',
                           '../model/Berlin_model.pth',
                           '../model/Augsburg_model.pth'],
        'log_path': ['../log/Houston2013_log.txt',
                     '../log/Houston2018_log.txt',
                     '../log/Trento_log.txt',
                     '../log/Berlin_log.txt',
                     '../log/Augsburg_log.txt'],
        'report_path': ['../report/Houston2013_report.txt',
                        '../report/Houston2018_report.txt',
                        '../report/Trento_report.txt',
                        '../report/Berlin_report.txt',
                        '../report/Augsburg_report.txt'],
        'image_path': ['../pic/Houston2013.png',
                       '../pic/Houston2018.png',
                       '../pic/Trento.png',
                       '../pic/Berlin.png',
                       '../pic/Augsburg.png']
    }

def set_value(key, value):
    parameter[key] = value

def get_value(key):
    try:
        return parameter[key]
    except:
        print('读取'+key+'失败\r\n')

def get_taskInfo():
    return '-----------------------taskInfo----------------------- \n lr:\t{} \n epoch_nums:\t{} \n batch_size:\t{} \n window_size:\t{} \n depth:\t{} \n------------------------------------------------------'.format(parameter['lr'], parameter['epoch_nums'], parameter['batch_size'], parameter['windowSize'], parameter['depth'])


