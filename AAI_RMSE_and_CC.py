import torch

"""
返回值都是一维的 tensor
"""


def get_PCC(tensor1, tensor2):
    """
    tensor1 和 tensor2 的形状都是 (length, 1)
    """
    # 去中心化
    centered_tensor1 = tensor1 - torch.mean(tensor1)
    centered_tensor2 = tensor2 - torch.mean(tensor2)

    numerator = torch.sum(centered_tensor1 * centered_tensor2)
    denominator = torch.sqrt(
        torch.sum(centered_tensor1 ** 2) * \
            torch.sum(centered_tensor2 ** 2)
    )

    PCC = numerator / denominator

    return PCC


def get_RMSE(tensor1, tensor2):
    """
    tensor1 和 tensor2 的形状都是 (length, 1)
    """
    diff = tensor1 - tensor2
    squared_diff = diff.pow(2)

    MSE = squared_diff.mean()
    RMSE = torch.sqrt(MSE)

    return RMSE


def get_channel_avg_PCC(tensor1, tensor2):
    """
    tensor1 和 tensor2 的形状都是 (length, channels)

    返回一个一维 tensor
    """
    # 中心化
    centered_tensor1 = \
        tensor1 - torch.mean(tensor1, dim=0, keepdim=True)
    centered_tensor2 = \
        tensor2 - torch.mean(tensor2, dim=0, keepdim=True)
    
    # 计算相关系数
    numerator = torch.sum(
        centered_tensor1 * centered_tensor2,
        dim=0
    )
    denominator = torch.sqrt(
        torch.sum(centered_tensor1 ** 2, dim=0) * \
            torch.sum(centered_tensor2 ** 2, dim=0)
    )
    
    correlation_per_channel = numerator / denominator

    # 对所有通道的相关系数进行平均
    avg_correlation = torch.mean(correlation_per_channel)
    
    return avg_correlation


def get_channel_avg_RMSE(tensor1, tensor2):
    """
    tensor1 和 tensor2 的形状都是 (length, channels)
    """

    # 计算两个张量的差值
    diff = tensor1 - tensor2
    
    # 对差值进行平方
    squared_diff = diff.pow(2)
    
    # 计算每个通道上的平方差值的平均值
    MSE_per_channel = squared_diff.mean(dim=0)
    
    # 计算每个通道上的均方根误差（RMSE）
    RMSE_per_channel = torch.sqrt(MSE_per_channel)
    
    # 计算所有通道的均方根误差（RMSE）的平均值
    channel_avg_RMSE = RMSE_per_channel.mean()
    
    return channel_avg_RMSE


def get_batch_avg_RMSE_and_PCC(
        tensor1,
        tensor2,
        original_lengths,
        unavail_indices=[]
    ):
    """
    tensor1 和 tensor2 的形状都是 (batch_size, duration, channels)
    original_lengths 的形状是 (batch_size)
    unavail_indices 是列表, 如果都是有效 channels, 就传入一个空列表
    """
    
    RMSE_sum = 0
    PCC_sum = 0
    
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

        channel_avg_RMSE = get_channel_avg_RMSE(X, Y)
        channel_avg_PCC = get_channel_avg_PCC(X, Y)
        
        RMSE_sum += channel_avg_RMSE
        PCC_sum += channel_avg_PCC
        
    batch_avg_RMSE = RMSE_sum / batch_size
    batch_avg_PCC = PCC_sum / batch_size
    

    return batch_avg_RMSE, batch_avg_PCC


def get_batch_avg_RMSE_and_PCC_with_stats(
        tensor1,
        tensor2,
        original_lengths,
        batch_mean,
        batch_std,
        target_std,
        unavail_indices=[]
    ):
    """
    tensor1 和 tensor2 的形状都是 (batch_size, duration, channels)
    original_lengths 的形状是 (batch_size)
    batch_mean 和 batch_std 的形状都是 (batch_size, channels)
    target_std 是一个数
    unavail_indices 是列表, 如果都是有效 channels, 就传入一个空列表
        - 也就是说默认全部 channels 都有效

    本函数先通过 batch_mean, batch_std, target_std 信息, 将归一化以后的 tensor1, tensor2 还原
    然后再计算它们之间的 RMSE 与 CC
    """
    RMSE_sum = 0
    PCC_sum = 0
    
    batch_size = tensor1.size(0)
    channel_num = tensor1.size(-1)

    indices = list(range(channel_num))
    avail_indices = [x for x in indices if x not in unavail_indices]

    for batch_index in range(batch_size):
        X = tensor1[batch_index, :, :]
        Y = tensor2[batch_index, :, :]

        mean = batch_mean[batch_index]
        std = batch_std[batch_index]

        original_length = original_lengths[batch_index]

        X = X[0:original_length, :]
        Y = Y[0:original_length, :]

        X = denormalise(X, mean, std, target_std)
        Y = denormalise(Y, mean, std, target_std)

        X = X[:, avail_indices]
        Y = Y[:, avail_indices]

        channel_avg_RMSE = get_channel_avg_RMSE(X, Y)
        channel_avg_PCC = get_channel_avg_PCC(X, Y)
        
        RMSE_sum += channel_avg_RMSE
        PCC_sum += channel_avg_PCC
        
    batch_avg_RMSE = RMSE_sum / batch_size
    batch_avg_PCC = PCC_sum / batch_size
    
    return batch_avg_RMSE, batch_avg_PCC


def denormalise(normalised_array, mean, std, target_std):
    denormalised_array = normalised_array * (std / target_std) + mean
    return denormalised_array


def get_RMSE_across_all_channels(tensor_1, tensor_2):
    """
    tensor_1 和 tensor_2 的形状相同, 都是 (duration, channels)

    return: 
        - avg_RMSE (RMSE averaged over all channels)
        - RMSE_list (a list containing RMSE of all channels)

    其中, avg_RMSE 和 RMSE_list 都是【数】, 不是 tensor
    """
    channel_num = tensor_1.shape[-1]
    RMSE_list = []

    for current_channel in list(range(channel_num)):
        channel_tensor_1 = tensor_1(current_channel)
        channel_tensor_2 = tensor_2(current_channel)

        channel_RMSE = get_RMSE(channel_tensor_1, channel_tensor_2)
        RMSE_list.append(channel_RMSE)
    
    avg_RMSE = sum(RMSE_list) / channel_num

    return avg_RMSE, RMSE_list

def get_PCC_across_all_channels(tensor_1, tensor_2):
    """
    tensor_1 和 tensor_2 的形状相同, 都是 (duration, channels)

    return:
        - avg_PCC (PCC averaged over all channels)
        - PCC_list (a list containing PCC of all channels)

    其中, avg_PCC 和 PCC_list 都是【数】, 不是 tensor
    """
    channel_num = tensor_1.shape[-1]
    PCC_list = []

    for current_channel in list(range(channel_num)):
        channel_tensor_1 = tensor_1(current_channel)
        channel_tensor_2 = tensor_2(current_channel)

        channel_PCC = get_PCC(channel_tensor_1, channel_tensor_2)
        PCC_list.append(channel_PCC)
    
    avg_PCC = sum(PCC_list) / channel_num

    return avg_PCC, PCC_list