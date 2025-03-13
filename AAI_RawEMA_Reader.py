import random
import os

import numpy as np
import scipy.io, scipy.interpolate


class RawEMA_Reader():
    """
    Used to read raw EMA of a speaker from a certain dataset.
    It works at the directory containing raw EMA files.
    """
    def __init__(
            self,
            raw_EMA_dir,
            speaker_name
        ):
        self.raw_EMA_dir = raw_EMA_dir
        self.speaker_name = speaker_name

        self.raw_EMA_type = None  # raw EMA file extension name
        self.EMA_index_list = None  # all indices of raw EMA
        self.EMA_NUM = None  # total num of raw EMA
        self._setup_raw_EMA_list()  # set the values for the preceding 3 attributes

        self.raw_EMA_channels = None # dictionary containing info about: sensor_name - index
        self.raw_EMA_values = None  # list specifying orders of values of an EMA channel like: x, y, z, theta...
        self._setup_raw_EMA_channels_and_values()  # set the values for the preceding 3 attributes
    

    """
    Switching between paths and indices.
    """
    def _get_raw_EMA_index_from_path(self, raw_EMA_path):
        """
        Extracting index from raw EMA path.
        """
        file_name = os.path.basename(raw_EMA_path)
        index, _ = os.path.splitext(file_name)
        return index
    
    def _get_raw_EMA_path_from_index(self, index):
        """
        Building raw EMA path from its index.
        """
        raw_EMA_path = os.path.join(
            self.raw_EMA_dir,
            index + self.raw_EMA_type
        )
        return raw_EMA_path
    

    """
    Read a random EMA.
    """
    def _randn_index_generator(self):
        index = random.sample(self.EMA_index_list, 1)
        index = index[0]
        return index

    def _randn_raw_EMA_path_generator(self):
        index = self._randn_index_generator()
        raw_EMA_path = self._get_raw_EMA_path_from_index(index)
        return raw_EMA_path
    
    def get_randn_std_EMA(self):
        raw_EMA_path = self._randn_raw_EMA_path_generator()
        std_EMA = self.get_std_EMA(raw_EMA_path)
        return std_EMA
    

    """
    General Operations
    """
    def get_std_EMA_by_index(self, index):
        """
        通过 index 获取对应的 standard EMA
        """
        EMA_path = self._get_raw_EMA_path_from_index(index)
        std_EMA = self.get_std_EMA(EMA_path)
        return std_EMA
    
    def get_std_EMA_list_by_index_list(
            self, index_list=None
        ):
        """
        Returns the list containing all std_EMA accessible by "index_list".
        If "index_list" is None, return all std EMA accessible by self.EMA_index_list.

        Parameters
        ----------
        index_list : list
            List containing indices to be loaded.

        Returns
        -------
        std_EMA_list : list
            The elements of this list are all tuples: (index, std_EMA).
        """
        if index_list is None:
            index_list = self.EMA_index_list
        std_EMA_list = []
        for index in index_list:
            std_EMA = self.get_std_EMA_by_index(index)
            print(f"std_EMA {index} loaded")
            std_EMA_list.append((index, std_EMA))
        return std_EMA_list
    

    """
    Special Operations that should be implemented in subclasses.
    """
    def _setup_raw_EMA_list(self):
        """
        Used to set the values of:
            - self.raw_EMA_type  (file extension of raw EMA)
            - self.EMA_index_list  (list containing all indices of EMA)
            - self.EMA_NUM  (total num of raw EMA)
        """
        raise NotImplementedError

    def _setup_raw_EMA_channels_and_values(self):
        """
        Used to set the values of:
            - self.raw_EMA_channels
            - self.raw_EMA_values
        """
        raise NotImplementedError

    def _read_raw_EMA(self, raw_EMA_path):
        """
        Extracting basic EMA data from raw EMA file.
        """
        raise NotImplementedError

    def get_std_EMA(self, raw_EMA_path):
        """
        Returns std EMA (num_frames, 12) as numpy.ndarray.
        12 represents x, y channel of 6 sensors, they are:
            - tt, tb, td, li, ul, ll
        """
        raise NotImplementedError


class BY2023_RawEMA_Reader(RawEMA_Reader):
    def __init__(
            self,
            raw_EMA_dir,
            speaker_name
        ):

        # 标准 EMA channel index
        self.std_raw_EMA_channels = {
            'NOSE' : 1, 'LE' : 2, 'RE' : 3,
            'TD' : 4, 'TB' : 5, 'TT' : 6,
            'LJ' : 7, 'LL' : 8, 'UL' : 9
        }

        # 记录哪些 speaker 的信息有异常
        self.speakers_with_normal_channels = [
            'F001', 'F002', 'F003', 'F004', 'F005',
            'M001', 'M002', 'M003', 'M004', 'M005',
            'L_M001', 'L_F001', 'L_F002', 'A_M001'
        ]
        self.speakers_with_irregular_channels = [
            'L_M002', 'L_M003', 'A_F001', 'L_F003',
            'L_F003_RU'
        ]

        # 异常 channels 信息
        self.L_M002_raw_EMA_channels = {
            'NOSE' : 11, 'LE' : 12, 'RE' : 13,
            'TD' : 14, 'TB' : 5, 'TT' : 6,
            'LJ' : 7, 'LL' : 8, 'UL' : 9
        }
        self.L_M003_raw_EMA_channels = self.L_M002_raw_EMA_channels
        self.A_F001_raw_EMA_channels = {
            'NOSE' : 11, 'LE' : 12, 'RE' : 13,
            'TD' : 14, 'TB' : 10, 'TT' : 16,
            "LJ" : 7, "LL" : 8, "UL" : 9
        }
        self.L_F003_raw_EMA_channels = {
            'NOSE' : 1, 'LE' : 5, 'RE' : 3,
            'TD' : 11, 'TB' : 10, 'TT' : 12,
            'LJ' : 7, 'LL' : 8, 'UL' : 9
        }

        super().__init__(
            raw_EMA_dir,
            speaker_name
        )


    """
    实现父类特殊操作
    """
    def _setup_raw_EMA_list(self):
        self.raw_EMA_type = ".pos"
        self.EMA_NUM = 962

        """
        L_F003 说话人有录制过 18 条俄语 EMA, 放在一个单独的文件夹
            - 即, 不在 L_F003 文件夹内
        如果没有找到, 就请直接忽略
        """
        if self.speaker_name == 'L_F003_RU':
            self.EMA_NUM = 18

        dir_filelist = os.listdir(self.raw_EMA_dir)
        self.EMA_index_list = [os.path.splitext(x)[0] for x in dir_filelist \
                               if os.path.splitext(x)[-1] == self.raw_EMA_type]
        self.EMA_index_list = sorted(self.EMA_index_list)
        # 只选取前 962 个 EMA, 因为之后的是上颚数据
        self.EMA_index_list = self.EMA_index_list[: self.EMA_NUM]
    
    def _setup_raw_EMA_channels_and_values(self):
        self.raw_EMA_values = [
            'x', 'y', 'z',
            'phi', 'theta', 'rms', 'extra'
        ]

        # 根据 speaker_name, 确定各个 channel 的 index
        if self.speaker_name not in self.speakers_with_normal_channels and \
                self.speaker_name not in self.speakers_with_irregular_channels:
            raise ValueError("不存在 speaker {}".format(self.speaker_name))
        
        if self.speaker_name in self.speakers_with_normal_channels:
            self.raw_EMA_channels = self.std_raw_EMA_channels
        elif self.speaker_name == 'L_M002':
            self.raw_EMA_channels = self.L_M002_raw_EMA_channels
        elif self.speaker_name == 'L_M003':
            self.raw_EMA_channels = self.L_M003_raw_EMA_channels
        elif self.speaker_name == 'A_F001':
            self.raw_EMA_channels = self.A_F001_raw_EMA_channels
        elif self.speaker_name == 'L_F003' or self.speaker_name == 'L_F003_RU':
            self.raw_EMA_channels = self.L_F003_raw_EMA_channels
    
    def _read_raw_EMA(self, raw_EMA_path):
        """
        该文件包含 header, 因此读取 EMA 数据时需要将 header 跳过
            - header 的字节数包含在 header 的第二行
        
        EMA 数据部分是一个一维 NumPy 数组
            - 16 个 channel
            - 每个 channel 7 个数据: x, y, z, phi, theta, rms, extra
        """
        header_size = self._get_header_size(raw_EMA_path)

        EMA_data = np.fromfile(
            raw_EMA_path,
            np.float32,
            offset=header_size
        ) # EMA_data 此时为一维 numpy 数组

        EMA_data = EMA_data.reshape((-1, 112))
        """
        最后所生成的 EMA_data 的形状为 (num_frams, 112)

        112 的含义:
            - 16 * 7 = 112
                - 16 代表 16 个 channels
                - 7 代表每个 channel 的 7 个值: x, y, z, phi, theta, rms, extra
        """
        return EMA_data
    
    def get_std_EMA(self, raw_EMA_path):
        """
        输入 EMA 数据的绝对路径

        输出:
            - (num_frames, 12) 形状的 EMA 数据
            - 以及 (num_frames, 2) 形状的 NOSE 数据
                - 12 代表着 tt, tb, td, lj, ul, ll 的 x, y 数据
                - 格式均为 NumPy 数组
        """

        EMA_data = self._read_raw_EMA(raw_EMA_path)

        tt_x = EMA_data[:, self._get_channel_value_index('TT', 'x')]
        tt_z = EMA_data[:, self._get_channel_value_index('TT', 'z')]
        tb_x = EMA_data[:, self._get_channel_value_index('TB', 'x')]
        tb_z = EMA_data[:, self._get_channel_value_index('TB', 'z')]
        td_x = EMA_data[:, self._get_channel_value_index('TD', 'x')]
        td_z = EMA_data[:, self._get_channel_value_index('TD', 'z')]

        lj_x = EMA_data[:, self._get_channel_value_index('LJ', 'x')]
        lj_z = EMA_data[:, self._get_channel_value_index('LJ', 'z')]
        ul_x = EMA_data[:, self._get_channel_value_index('UL', 'x')]
        ul_z = EMA_data[:, self._get_channel_value_index('UL', 'z')]
        ll_x = EMA_data[:, self._get_channel_value_index('LL', 'x')]
        ll_z = EMA_data[:, self._get_channel_value_index('LL', 'z')]

        stacked_channels = np.vstack((
            tt_x, tt_z, tb_x, tb_z, td_x, td_z,
            lj_x, lj_z, ul_x, ul_z, ll_x, ll_z
        ))
        stacked_channels = stacked_channels.T

        return stacked_channels
    

    """
    Tools
    """
    def _get_header_size(self, raw_EMA_path):
        """
        根据 .pos 的路径, 获得该 .pos 文件的【头文件大小】
        """
        with open(raw_EMA_path, 'rb') as EMA_file:
            # 取出 header 的字节数
            counter = 0
            for line in EMA_file:
                counter += 1
                if counter == 2: # header 的第二行, 就是 header 的字节数的信息
                    header_size = line.decode('latin-1').strip("\n")
                    header_size = int(header_size)
                    break
        return header_size
    
    def _get_channel_value_index(self, sensor, value):
        sensor_index = self.raw_EMA_channels[sensor] - 1
        value_index = self.raw_EMA_values.index(value)

        channel_index = sensor_index * 7 + value_index
        return channel_index
    
    # handling header files
    def print_header(self, raw_EMA_path):
        with open(raw_EMA_path, 'rb') as EMA_file:
            # 取出 header 的字节数
            counter = 0
            for line in EMA_file:
                line = line.decode('latin-1').strip("\n")
                print(line)
                counter += 1
                if counter == 14:
                    break
    
    # getting nose EMA
    def get_nose_EMA_by_path(self, raw_EMA_path):
        EMA_data = self._read_raw_EMA(raw_EMA_path)

        nose_x = EMA_data[:, self._get_channel_value_index('NOSE', 'x')]
        nose_z = EMA_data[:, self._get_channel_value_index('NOSE', 'z')]

        stacked_channels = np.vstack((
            nose_x, nose_z
        ))
        stacked_channels = stacked_channels.T

        return stacked_channels
    
    def get_nose_EMA_by_index(self, index):
        raw_EMA_path = self._get_raw_EMA_path_from_index(index)
        nose_EMA = self.get_nose_EMA_by_path(raw_EMA_path)
        return nose_EMA
    
    def get_nose_EMA_list_by_index_list(self, index_list=None):
        """
        Returns a list containing (index, nose_EMA) accessible by index_list.
        If index_list is None, change it to self.EMA_index_list.

        Returns
        -------
        nose_EMA_list : list
            The elements of this list are all tuples: (index, nose_EMA).
            Shape of "nose_EMA": (num_frames, 2).
        """
        if not index_list:
            index_list = self.EMA_index_list
        nose_EMA_list = []
        for index in index_list:
            nose_EMA = self.get_nose_EMA_by_index(index)
            nose_EMA_list.append((index, nose_EMA))
        return nose_EMA_list

        
class MOCHA_TIMIT_RawEMA_Reader(RawEMA_Reader):
    def __init__(
            self,
            raw_EMA_dir,
            speaker_name
        ):

        super().__init__(
            raw_EMA_dir,
            speaker_name
        )

        self.speakers_with_velum = [
            "fsew0", "faet0", "ffes0", "falh0",
            "msak0"
        ]
        if self.speaker_name in self.speakers_with_velum:
            self.with_velum = True
        else:
            self.with_velum = False


    """
    实现父类特殊操作
    """
    def _setup_raw_EMA_list(self):
        self.raw_EMA_type = ".ema"
        self.EMA_NUM = 460

        dir_filelist = os.listdir(self.raw_EMA_dir)
        self.EMA_index_list = [os.path.splitext(x)[0] for x in dir_filelist \
                               if os.path.splitext(x)[-1] == self.raw_EMA_type]
        self.EMA_index_list = sorted(self.EMA_index_list)
        # 超过 460 的都是无效数据
        self.EMA_index_list = [
            x for x in self.EMA_index_list \
                if x.split("_")[-1].isdigit() and \
                    int(x.split("_")[-1]) <= 460
        ]

        if self.EMA_NUM != len(self.EMA_index_list):
            # 有的 speaker 的有效数据不足 460 个
            self.EMA_NUM = len(self.EMA_index_list)
    

    def _setup_raw_EMA_channels_and_values(self):
        self.raw_EMA_channels = [
            'ui', 'li',
            'ul', 'll',
            'tt', 'tb', 'td',
            'v'
        ]
        self.raw_EMA_values = None
    

    def _read_raw_EMA(self, raw_EMA_path):
        """
        返回 (num_frames, 22) 的 EMA_data, 以及记录了其 channel_names 的列表

        22 个通道里, 前两个通道不是 EMA 数据
        """

        with open(raw_EMA_path, 'rb') as EMA_file:
            num_frames, channel_names_list = self._EST_Header_reader(EMA_file)
            EMA_data = np.fromfile(EMA_file, "float32").reshape(num_frames, -1)
            # EMA_data 的 shape 为 (num_frames, 22)

            EMA_data = EMA_data / 100 # 默认单位为 0.1 m, 因此需要除以 100 转化为毫米值
            EMA_data = self._NaN_handler(EMA_data)

            return EMA_data, channel_names_list, num_frames
    

    def get_std_EMA(self, raw_EMA_path):
        """
        输入 EMA 数据的绝对路径

        输出:
            - (num_frames, 12) 形状的 EMA 数据
            - 以及 (num_frames, 2) 形状的 NOSE 数据
                - 12 代表着 tt, tb, td, lj, ul, ll 的 x, y 数据
                - 格式均为 NumPy 数组
        """

        EMA_data, channel_names_list, num_frames = \
            self._read_raw_EMA(raw_EMA_path)

        def _get_index(sensor, value):
            channel_name = sensor + "_" + value
            # EMA_data 的通道数为 22
            # 前两个通道不是 EMA 数据, 因此要跳过
            return channel_names_list.index(channel_name) + 2
        
        tt_x = EMA_data[:, _get_index('tt', 'x')]
        tt_y = EMA_data[:, _get_index('tt', 'y')]
        tb_x = EMA_data[:, _get_index('tb', 'x')]
        tb_y = EMA_data[:, _get_index('tb', 'y')]
        td_x = EMA_data[:, _get_index('td', 'x')]
        td_y = EMA_data[:, _get_index('td', 'y')]

        li_x = EMA_data[:, _get_index('li', 'x')]
        li_y = EMA_data[:, _get_index('li', 'y')]
        ul_x = EMA_data[:, _get_index('ul', 'x')]
        ul_y = EMA_data[:, _get_index('ul', 'y')]
        ll_x = EMA_data[:, _get_index('ll', 'x')]
        ll_y = EMA_data[:, _get_index('ll', 'y')]


        stacked_channels = np.vstack((
            tt_x, tt_y, tb_x, tb_y, td_x, td_y,
            li_x, li_y, ul_x, ul_y, ll_x, ll_y
        ))
        stacked_channels = stacked_channels.T

        return stacked_channels
    

    """
    其他内部工具
    """
    def _EST_Header_reader(self, EMA_file):
        """
        EMA_file 是一个 file 对象

        读取该 file 对象的 EST Header, 获得以下信息:
            - num_frames: 该条 EMA 数据的帧数
            - channel_names_list:
                - 一个长度为 20 的列表
                - 存储着诸如 "ui_x, li_x, ul_x" 这样的 channel_name
                - 每个 channel_name 在列表中的 index 就是其在 EMA 数据中的 index
        """

        channel_names_list = [0] * 20

        for line in EMA_file:
            line = line.decode('latin-1').strip("\n")
            
            if line == 'EST_Header_End': # 若为 header 的 end, 则结束
                break

            elif line.startswith('NumFrames'): # 获得帧数
                num_frames = int(line.rsplit(' ', 1)[-1])

            elif line.startswith('Channel_'):  # 获得各个 channel 的名称
                channel_index, channel_name = line.split(' ', 1)
                """
                - channel_index 示例:
                    - Channel_0, Channel_1, ..., Channel_19
                - channel_name 示例:
                    - ui_x, li_x, ul_x, ...
                """

                # 将 channel_index 转换为数字
                # 例如: Channel_1 -> 1
                channel_index = int(channel_index.split('_', 1)[-1])

                channel_names_list[channel_index] = channel_name.replace(" ", "")
                # v_x 有时候会有空格, 因此需要替换掉
        
        return num_frames, channel_names_list
    

    def _NaN_handler(self, EMA_data):
        """
        处理 EMA 数据中有 NaN 的情况
        """
        if np.isnan(EMA_data).sum() != 0:
            print("NaN 的数量", np.isnan(EMA_data).sum())
            # Build a cubic spline out of non-NaN values.

            spline = scipy.interpolate.splrep(
                np.argwhere(~np.isnan(EMA_data).ravel()),
                EMA_data[~np.isnan(EMA_data)], k=3
            )
            """
            np.isnan 形成一个和 ema_data 同形状的 np array, 为 NaN 的地方为 True, 其他地方为 False
            ~np.isnan 进行非运算, 这样为 NaN 的地方为 False, 其他地方为 True
            ravel() 将该数组展平为一维数组
            np.argwhere 返回一维数组中所有非 0 元素的索引, 插值将在这些点上进行
            """
            # Interpolate missing values and replace them.
            for j in np.argwhere(np.isnan(EMA_data)).ravel():
                EMA_data[j] = scipy.interpolate.splev(j, spline)

        return EMA_data


    """
    其他外部工具
    """
    def EST_Header_printer(self, raw_EMA_path):
        with open(raw_EMA_path, 'rb') as EMA_file:
            for line in EMA_file:
                line = line.decode('latin-1').strip("\n")
                print(line)

                if line == 'EST_Header_End': # 若为 header 的 end, 则结束
                    break


class HPRC_RawEMA_Reader(RawEMA_Reader):
    def __init__(
            self,
            raw_EMA_dir,
            speaker_name
        ):

        super().__init__(
            raw_EMA_dir,
            speaker_name
        )

    """
    实现父类特殊操作
    """
    def _setup_raw_EMA_list(self):
        self.raw_EMA_type = ".mat"

        dir_filelist = os.listdir(self.raw_EMA_dir)
        self.EMA_index_list = [os.path.splitext(name)[0] for name in dir_filelist \
                               if "palate" not in name]
        self.EMA_NUM = len(self.EMA_index_list)
    

    def _setup_raw_EMA_channels_and_values(self):
        self.raw_EMA_channels = [
            'AUDIO',
            'TR', 'TB', 'TT',
            'UL', 'LL', 'ML',
            'JAW', 'JAWL'
        ]
        self.raw_EMA_values = [
            'pos_x', 'pos_y', 'pos_z',
            'rotation_x', 'rotation_y', 'rotation_z'
        ]
    

    def _read_raw_EMA(self, raw_EMA_path):
        index = self._get_raw_EMA_index_from_path(raw_EMA_path)
        EMA_data = scipy.io.loadmat(raw_EMA_path)[index][0]
        # shape (9,)
        return EMA_data


    def get_std_EMA(self, raw_EMA_path):
        """
        输入 EMA 数据的绝对路径

        输出:
            - (num_frames, 12) 形状的 EMA 数据
            - 以及 (num_frames, 2) 形状的 NOSE 数据
                - 12 代表着 tt, tb, td, lj, ul, ll 的 x, y 数据
                - 格式均为 NumPy 数组
        """
        EMA_data = self._read_raw_EMA(raw_EMA_path)

        tt_x = EMA_data[self.raw_EMA_channels.index('TT')][2][:, self.raw_EMA_values.index('pos_x')]
        tt_z = EMA_data[self.raw_EMA_channels.index('TT')][2][:, self.raw_EMA_values.index('pos_z')]
        tb_x = EMA_data[self.raw_EMA_channels.index('TB')][2][:, self.raw_EMA_values.index('pos_x')]
        tb_z = EMA_data[self.raw_EMA_channels.index('TB')][2][:, self.raw_EMA_values.index('pos_z')]
        tr_x = EMA_data[self.raw_EMA_channels.index('TR')][2][:, self.raw_EMA_values.index('pos_x')]
        tr_z = EMA_data[self.raw_EMA_channels.index('TR')][2][:, self.raw_EMA_values.index('pos_z')]

        jaw_x = EMA_data[self.raw_EMA_channels.index('JAW')][2][:, self.raw_EMA_values.index('pos_x')]
        jaw_z = EMA_data[self.raw_EMA_channels.index('JAW')][2][:, self.raw_EMA_values.index('pos_z')]
        ul_x = EMA_data[self.raw_EMA_channels.index('UL')][2][:, self.raw_EMA_values.index('pos_x')]
        ul_z = EMA_data[self.raw_EMA_channels.index('UL')][2][:, self.raw_EMA_values.index('pos_z')]
        ll_x = EMA_data[self.raw_EMA_channels.index('LL')][2][:, self.raw_EMA_values.index('pos_x')]
        ll_z = EMA_data[self.raw_EMA_channels.index('LL')][2][:, self.raw_EMA_values.index('pos_z')]

        stacked_channels = np.vstack((
            tt_x, tt_z, tb_x, tb_z, tr_x, tr_z,
            jaw_x, jaw_z, ul_x, ul_z, ll_x, ll_z
        ))
        stacked_channels = stacked_channels.T

        return stacked_channels
    

    """
    其他外部工具

    HPRC 数据集的 wav 与 xtrm_list 都在 EMA 里面, 需要专门提取
    """
    def get_wav_from_raw_EMA(self, index):
        raw_EMA_path = self._get_raw_EMA_path_from_index(index)
        EMA_data = self._read_raw_EMA(raw_EMA_path)
        wav = EMA_data[self.raw_EMA_channels.index('AUDIO')][2][:, 0]

        return wav
    
    def get_all_wav(self):
        """
        提取所有的 wav, 放入列表返回
        """
        all_wav_list = []

        for index in self.EMA_index_list:
            wav = self.get_wav_from_raw_EMA(index)
            print(f"wav for '{index}' extracted")
            all_wav_list.append((index, wav))
        
        return all_wav_list
    
    
    def get_xtrm_list_from_index(self, index):
        raw_EMA_path = self._get_raw_EMA_path_from_index(index)
        EMA_data = self._read_raw_EMA(raw_EMA_path)
        wav_data = EMA_data[self.raw_EMA_channels.index('AUDIO')]

        for k in [5, 6]:
            """
            5, 6 为 word 和 phone 的标注文件

            有的文件可能 5, 6 不同？
            所以用了 try / except 的结构
            """
            try:
                first_sp = wav_data[k][0][0]
                last_sp = wav_data[k][0][-1]

                start_time = first_sp[1][0][1]
                end_time = last_sp[1][0][0]
            except:
                pass
        
        return [start_time, end_time]
    
    def extract_annotation_info_from_index(self, index):
        raw_EMA_path = self._get_raw_EMA_path_from_index(index)
        EMA_data = self._read_raw_EMA(raw_EMA_path)
        wav_data = EMA_data[self.raw_EMA_channels.index('AUDIO')]

        tier_index = [5, 6]
        if not isinstance(wav_data[5][0], np.ndarray):
            tier_index = [x + 1 for x in tier_index]

        # word tier
        words = []
        for sp in wav_data[tier_index[0]][0]:
            word = sp[0][0]
            start_time = sp[1][0][0]
            end_time = sp[1][0][1]
            words.append((start_time, end_time, word))
        
        # phoneme tier
        phonemes = []
        for sp in wav_data[tier_index[1]][0]:
            phoneme = sp[0][0]
            start_time = sp[1][0][0]
            end_time = sp[1][0][1]
            phonemes.append((start_time, end_time, phoneme))
        
        return (words, phonemes)
    
    def extract_all_annotation_info(self):
        all_annotation_info = []
        for index in self.EMA_index_list:
            annotation_info = self.extract_annotation_info_from_index(index)
            print(f"annotation info for {index} extracted")
            all_annotation_info.append((index, annotation_info))
        return all_annotation_info



    def get_all_xtrm_list(self):
        """
        all_xtrm_list 的元素都是 list
            - [start_time, end_time]
        """
        all_xtrm_list = []

        for index in self.EMA_index_list:
            xtrm_list = self.get_xtrm_list_from_index(index)
            all_xtrm_list.append(xtrm_list)
        
        return all_xtrm_list


