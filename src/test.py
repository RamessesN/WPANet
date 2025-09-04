from dataset import *
from report import *
from visualization import *

import parameter

parameter._init()

def myTest(datasetType):
    device_str = parameter.get_value('device')

    if device_str == "cuda:0" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif device_str == "mps" and torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[Test] Using device: {device}")

    channels = parameter.get_value('channels')
    windowSize = parameter.get_value('windowSize')
    batch_size = parameter.get_value('batch_size')
    num_workers = parameter.get_value('num_workers')
    random_seed = parameter.get_value('random_seed')
    visualization = parameter.get_value('visualization')
    model_savepath = parameter.get_value('model_savepath')
    report_path = parameter.get_value('report_path')
    image_path = parameter.get_value('image_path')

    # net = torch.load(model_savepath[datasetType])
    net = torch.load(model_savepath[datasetType], map_location=device, weights_only=False).to(device)
    net.eval()

    train_loader, test_loader, trntst_loader, all_loader = getMyData(datasetType, channels, windowSize, batch_size, num_workers)
    set_random_seed(random_seed)
    getMyReport(datasetType, net, test_loader, report_path[datasetType], device)
    if visualization:
        getMyVisualization(datasetType, net, trntst_loader, image_path[datasetType], device)