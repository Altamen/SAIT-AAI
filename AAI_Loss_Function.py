import torch
import torch.nn as nn

from ExpAAI_RMSE_and_CC import *


class Masked_MSE_Loss(nn.Module):
    def __init__(self):
        super(Masked_MSE_Loss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(
            self,
            pred,
            label,
            mask,
            device
        ):
        # pred, label 和 mask 的形状都是 (batch_size, duration, dimension)

        LOSS_all = self.criterion(pred, label)
        LOSS = torch.mean(LOSS_all * mask)
        LOSS = LOSS.to(device)

        return LOSS


class PCC_Loss(nn.Module):
    def __init__(self, is_avg=True):
        super(PCC_Loss, self).__init__()
        self.is_avg = is_avg
    

    def forward(
            self,
            tensor1,
            tensor2,
            original_lengths,
            device,
            unavail_indices=[]
        ):
        """
        tensor1 和 tensor2 的形状都是 (batch_size, duration, channels)
        original_lengths 的形状是 (batch_size)
        unavail_indices 是列表, 如果都是有效 channels, 就传入一个空列表
        """
        LOSS_sum = 0

        batch_size = tensor1.size(0)
        channel_num = tensor1.size(-1)

        indices = list(range(channel_num))
        avail_indices = [x for x in indices if x not in unavail_indices]

        for batch_index in range(batch_size):
            X = tensor1[batch_index, :, :]
            Y = tensor2[batch_index, :, :]

            original_length = original_lengths[batch_index]

            X = X[0:original_length, :]
            Y = Y[0:original_length, :]

            X = X[:, avail_indices]
            Y = Y[:, avail_indices]

            channel_avg_PCC = get_channel_avg_PCC(X, Y)
            LOSS_sum += 1 - channel_avg_PCC
        
        if self.is_avg:
            LOSS = LOSS_sum / batch_size
        else:
            LOSS = LOSS_sum
        
        LOSS = LOSS.to(device)
        
        return LOSS