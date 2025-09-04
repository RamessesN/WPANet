from train import *
from test import *

import parameter

parameter._init()

def myTask(lr, epoch_nums, datasetType):
    parameter.set_value('epoch_nums', epoch_nums)
    parameter.set_value('lr', lr)
    parameter.set_value('device', device)
    parameter.set_value('visualization', visualization)
    myTrain(datasetType, net)
    myTest(datasetType)

device = 'cuda0'
net = 'WPANet'
visualization = False


if __name__ == '__main__':
    myTask(0.0001, 100, 0)
    # myTest(0)



