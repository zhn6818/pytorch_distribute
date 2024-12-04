from torchvision import transforms, datasets
import torch
import sys
sys.path.append('/data1/zhn/server/pytorch-distributed-training/utils')
# from custom import Custom

def get_train_dataset():

    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    trainset = datasets.CIFAR100(root='/data1/zhn/server/pytorch-distributed-training/data',
                                train=True,
                                download=True,
                                transform=transform_train)
    return trainset


def get_test_dataset():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
    ])
    testset = datasets.CIFAR100(root='/data1/zhn/server/pytorch-distributed-training/data',
                                train=False,
                                download=True,
                                transform=transform_test)
    return testset

# def get_fogData():
    
#     trainFog = Custom('/data5/zhn/fog/model_train_data_5/train_0713.txt')
    
#     return trainFog
class FogData:
    """
    A custom class to load and manage fog data.
    """
    def __init__(self, file_path):
        """
        Initialize the data loader with the file path.
        
        Args:
            file_path (str): Path to the data file.
        """
        self.file_path = file_path
        self.data = []
        self.load_data()

    def load_data(self):
        """
        Load data from the file into memory.
        """
        try:
            with open(self.file_path, 'r') as file:
                self.data = [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
        except Exception as e:
            print(f"Error loading data: {e}")

    def get_data(self):
        """
        Return the loaded data.
        
        Returns:
            list: List of data lines.
        """
        return self.data

    def filter_data(self, condition):
        """
        Filter the data based on a condition.
        
        Args:
            condition (callable): A function that takes a data line and returns a boolean.
        
        Returns:
            list: Filtered data.
        """
        return [line for line in self.data if condition(line)]



if __name__ == "__main__":
    # get_train_dataset()
    # tt = get_test_dataset()
    # 初始化数据类
    fog_data = FogData('/data5/zhn/fog/model_train_data_5/train_0713.txt')

    # 获取所有数据
    traindata = fog_data.get_data()
    # traindata = get_fogData()
    train_loader = torch.utils.data.DataLoader(traindata, batch_size=1)
    # it = train_loader.__next__()
    img, label = next(iter(train_loader))
    print("test")
