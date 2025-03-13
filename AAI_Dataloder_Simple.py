import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SpeakerSpecificDataset(Dataset):
    """
    用于特定说话人训练场景

    SF_name 控制所输入的发音特征
    input_EMA_channel_list, input_EMA_TV_list 控制是否加入外部可见发音特征作为输入的一部分

    if_output_with_TV 控制输出是否包括 3 个舌部 TVs (TTCL, TMCL, TRCL)

    more_stats 控制是否输出 EMA 的 mean 和 std, 便于还原数据

    new_std 确定归一化时新的标准差
    """
    def __init__(
            self,
            SpeakerData_Manager,
            index_list,
            SF_name,
            input_EMA_channel_list=[],
            input_EMA_TV_list=[],
            if_output_with_TV=False,
            more_stats=False,
            new_std=2
        ):

        self.SpeakerData_Manager = SpeakerData_Manager
        self.index_list = index_list
        self.SF_name = SF_name
        self.input_EMA_channel_list = input_EMA_channel_list
        self.input_EMA_TV_list = input_EMA_TV_list
        self.if_output_with_TV = if_output_with_TV
        self.more_stats = more_stats
        self.new_std = new_std

        if (input_EMA_channel_list == []) and (input_EMA_TV_list == []):
            self.additional_input = False
        else:
            self.additional_input = True

        self.output_sensors_list = ["tt", "tb", "td"]
        if self.if_output_with_TV:
            self.output_TVs_list = ["TTCL", "TMCL", "TRCL"]
        else:
            self.output_TVs_list = []

        """
        加载数据
        """
        # 加载输入
        SF_list = self.SpeakerData_Manager.load_data_list_from_dir_under_prep_data(
            dir_name=self.SF_name,
            index_list=self.index_list
        )

        if SF_name == "MFCC429":
            SF_name = "MFCC39"

        # 加载输出
        EMA_list = self.SpeakerData_Manager.load_EMA_sensors_from_dir_under_prep_data(
            dir_name="df_preped_EMA_"+SF_name,
            sensors_list=self.output_sensors_list,
            TVs_list=self.output_TVs_list,
            index_list=self.index_list
        )

        # 用于去除静音段
        PS_list = self.SpeakerData_Manager.load_data_list_from_dir_under_prep_data(
            dir_name="PhoneSeq_"+SF_name,
            index_list=self.index_list
        )

        """
        去除静音段, z-score 归一化
        """
        SF_list, _ = self.SpeakerData_Manager.last_process(
            data_list=SF_list,
            PS_list=PS_list,
            if_normalise=True,
            new_std=self.new_std,
        )
        EMA_list, _, EMA_stats_list = self.SpeakerData_Manager.last_process(
            data_list=EMA_list,
            PS_list=PS_list,
            if_normalise=True,
            new_std=self.new_std,
            more_stats=True
        )

        self.SF_list = SF_list
        self.EMA_list = EMA_list
        self.EMA_stats_list = EMA_stats_list

        if self.additional_input:
            additional_input_list = self.SpeakerData_Manager.load_EMA_sensors_from_dir_under_prep_data(
                dir_name="df_preped_EMA_"+SF_name,
                sensors_list=self.input_EMA_channel_list,
                TVs_list=self.output_TVs_list,
                index_list=self.index_list
            )
            additional_input_list, _ = self.SpeakerData_Manager.last_process(
                data_list=additional_input_list,
                PS_list=PS_list,
                if_normalise=True,
                new_std=self.new_std,
            )
            self.additional_input_list = additional_input_list

        print("数据集载入完成")
    
    def __len__(self):
        return len(self.EMA_list)

    def __getitem__(self, index):
        idx_SF, SF = self.SF_list[index]
        idx_EMA, EMA = self.EMA_list[index]
        idx_EMA_stats, mean, std = self.EMA_stats_list[index]
        
        if idx_SF != idx_EMA or idx_EMA_stats != idx_EMA:
            raise ValueError(f"idx 不一致!")
        
        if self.additional_input:
            if idx_additional != idx_EMA:
                raise ValueError(f"idx 不一致!")
            idx_additional, additional_input = self.additional_input_list[index]
            SF = np.hstack((SF, additional_input))

        SF = torch.from_numpy(SF)
        EMA = torch.from_numpy(EMA)
        
        if not self.more_stats:
            return SF, EMA
        else:
            mean = torch.from_numpy(mean)
            std = torch.from_numpy(std)
            return SF, EMA, mean, std


def pad_batch_to_same_length(batch_list):
    """
    Dataloader 给本函数的输入是一个 list, 其长度为 batch_size

    这个 list 中的元素都是 tuple
    - tuple 中的数据就是 Dataset 返回的数据 (SF, EMA), 数据类型都是 torch.Tensor
        - SF 和 EMA 的 size 为 (NumFrames, Dimensions)
    """

    durations, sorted_batchIndices_list = torch.sort(
        torch.LongTensor(
            [len(x[0]) for x in batch_list]
        ),
        dim=0,
        descending=True
    )
    max_duration = durations[0]
    batch_size = len(batch_list)

    input_dim = batch_list[0][0].size(-1)
    output_dim = batch_list[0][1].size(-1)

    batch_input_padded = torch.zeros(
        batch_size,
        max_duration,
        input_dim,
        dtype=torch.double
    )
    batch_output_padded = torch.zeros(
        batch_size,
        max_duration,
        output_dim,
        dtype=torch.double
    )
    batch_masks = torch.zeros(
        batch_size,
        max_duration,
        output_dim,
        dtype=torch.double
    )
    batch_durations = torch.zeros(
        batch_size,
        dtype=torch.int32
    )

    for i in range(len(sorted_batchIndices_list)):
        current_batchIndex = sorted_batchIndices_list[i]

        current_input = batch_list[current_batchIndex][0]
        current_output = batch_list[current_batchIndex][1]
        current_duration = len(current_input)

        batch_input_padded[i, :current_input.size(0), :] = current_input
        batch_output_padded[i, :current_output.size(0), :] = current_output

        batch_masks[i, :current_duration, :] = 1
        batch_durations[i] = current_duration
    
    """
    batch_input_padding (batch_size, num_frames, input_dim)
    batch_output_padding (batch_size, num_frames, output_dim)
    batch_duration (batch_size,)
    batch_masks (batch_size, num_frames, output_dim)
    """
    return batch_input_padded, batch_output_padded, batch_durations, batch_masks


def get_dataloader(
        SpeakerData_Manager,
        index_list,
        SF_name,
        batch_size,
        config
    ):
    dataset = SpeakerSpecificDataset(
        SpeakerData_Manager=SpeakerData_Manager,
        index_list=index_list,
        SF_name=SF_name,
        input_EMA_channel_list=config["input_EMA_channel_list"],
        input_EMA_TV_list=config["input_EMA_TV_list"],
        if_output_with_TV=config["if_output_with_TV"],
        more_stats=False,
        new_std=config["new_std"]
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_batch_to_same_length,
        pin_memory=True
    )
    return dataloader