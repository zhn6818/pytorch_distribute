
import time
import torch
import torch.nn as nn

# import torch.backends.cudnn as cudnn
# import torch.optim
# import torch.utils.data
# from utils.model import resnet18
# from utils.dataset import get_train_dataset, get_test_dataset, get_fogData
# from utils.util import reduce_mean
# from utils.newModel import FCNnew
from utils.FCN8s import FCN
import cv2

# from utils.validation import validate
# import torch.optim as optim
import os
import copy
import torch.distributed as dist
import torchvision.transforms as transforms

from utils.newModel import FCNnew

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '1023'

if __name__ == "__main__":
    init_method = 'env://'

    # 1. 分布式初始化，对于每一个进程都需要进行初始化，所以定义在 main_worker中
    dist.init_process_group(backend='nccl', init_method=init_method,world_size=1,rank=0)
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
    
    model = FCNnew(num_classes=2)
    if os.path.exists("./checkpoint/test_newModel_80_0.0005473231829366409.pth"):
        model.load_state_dict(torch.load("./checkpoint/test_newModel_80_0.0005473231829366409.pth").module.state_dict())
        
    model.cuda()
    model.eval()
    path = '/data5/zhn/fog/model_train_data_5/image/image_1'
    # '/data5/zhn/fog/model_train_data_5/image/image_1/'
    img_list = os.listdir(path)
    
    i = 1
    for file in img_list:
        
        imgfile = os.path.join(path, file)
        img = cv2.imread(imgfile)
        
        imgcp = copy.deepcopy(img)
        img = cv2.resize(img, (1280, 640))
        filename = "./result/img/" + str(i) + '.png' 
        cv2.imwrite(filename, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = img_transform(img)
        
        img = img_tensor.unsqueeze(0).cuda()
        # torch.cuda.synchronize()
        
        logits = model(img)
        
        logitsimg = logits[0][1] * 255
        
        filenameresult = "./result/img/" + str(i) + '_result.png' 
        cv2.imwrite(filenameresult, logitsimg.cpu().detach().numpy())
        i = i + 1
        print('hello')
    
    
    
    