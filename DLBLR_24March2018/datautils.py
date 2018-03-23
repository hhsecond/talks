import torch
from torch.utils import data


class FizBuzDataSet(data.Dataset):
    """
    dataset generates the data for fizbuz
    """

    def __init__(self, start=0, end=1000):
        super().__init__()
        x = []
        y = []
        for i in range(start, end):
            x.append(self.encoder(i))
            if i % 15 == 0:
                y.append([1, 0, 0, 0])
            elif i % 5 == 0:
                y.append([0, 1, 0, 0])
            elif i % 3 == 0:
                y.append([0, 0, 1, 0])
            else:
                y.append([0, 0, 0, 1])
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    @staticmethod
    def encoder(num):
        ret = [int(i) for i in '{0:b}'.format(num)]
        return [0] * (10 - len(ret)) + ret

    @staticmethod
    def decoder(array):
        ret = 0
        for i in array:
            ret = ret * 2 + int(i)
        return ret

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def check_fizbuz(i):
    if i % 15 == 0:
        return 'fizbuz'
    elif i % 5 == 0:
        return 'buz'
    elif i % 3 == 0:
        return 'fiz'
    else:
        return 'number'

def print_out(epoch, x, pred, loss):
    epoch_print = f'Epoch: {epoch:3d}'
    input_print = f'Input: {x:4d} -> {check_fizbuz(x):8s}'
    pred_print = f'Prediction: {pred:8s}'
    loss_print = f'loss: {loss:5.3f}'
    print(f'{epoch_print} | {input_print} | {pred_print} | {loss_print}')