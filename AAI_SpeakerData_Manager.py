import os
import json
import shutil
import random

import tqdm
import librosa
import soundfile as sf

from AAI_RawEMA_Reader import *
from AAI_Annotation_Reader import *


class SpeakerData_Manager():
    def __init__(self, dataset_path, speaker_name):
        self.dataset_path = dataset_path
        self.speaker_name = speaker_name
        self.dataset_name = os.path.basename(os.path.normpath(dataset_path))

        print("---------")
        print(f"Building Data_Manager for speaker '{self.speaker_name}' in '{self.dataset_name}'")

        """
        load dataset information
        """
        Dataset_Info_path = r"AAI_Dataset_Info.json"
        with open(Dataset_Info_path, "r") as f:
            Dataset_Info = json.load(f)
        self.Current_Dataset_Info = Dataset_Info["Dataset_Info_List"]\
            [Dataset_Info["Dataset_Name_List"].index(self.dataset_name)]
        
        self.speaker_list = self.Current_Dataset_Info["Speaker_List"]
        if self.speaker_name not in self.speaker_list:
            raise ValueError(f"speaker name {self.speaker_name} is not in the name list!")
        
        # get information about Sample Rate (sr)
        self.raw_EMA_sr = self.Current_Dataset_Info["raw_EMA_sr"]
        self.wav_sr = self.Current_Dataset_Info["wav_sr"]
        self.with_nose = self.Current_Dataset_Info["with_nose"]
        self.with_annotation = self.Current_Dataset_Info["with_annotation"]
        self.silence_token = self.Current_Dataset_Info["silence_token"]

        # get the absolute path of the directories "raw_data" and "prep_data"
        self.raw_data_dir_path = os.path.join(self.dataset_path, self.speaker_name, "raw_data")
        self.prep_data_dir_path = os.path.join(self.dataset_path, self.speaker_name, "prep_data")

        # get the absolute path of directories containing raw data
        self.raw_EMA_dir_path = self._get_path_from_Dataset_Info("raw_EMA_dir_path")
        self.wav_dir_path = self._get_path_from_Dataset_Info("wav_dir_path")
        self.Annotation_dir_path = self._get_path_from_Dataset_Info("Annotation_dir_path")

        # choose the corresponding RawEMA_Reader for the dataset
        RawEMA_Reader_name = self.dataset_name + "_RawEMA_Reader"
        if RawEMA_Reader_name in globals():
            self.RawEMA_Reader = globals()[RawEMA_Reader_name]\
                (self.raw_EMA_dir_path, self.speaker_name)
            self.all_index_list = self.RawEMA_Reader.EMA_index_list
            print("RawEMA_Reader loaded")
        else:
            raise ValueError(f"{RawEMA_Reader_name} not found")
        
        # choose the corresponding annotation reader
        if self.with_annotation:
            Annotation_Reader_name = self.dataset_name + "_Annotation_Reader"
            if Annotation_Reader_name in globals():
                if not os.path.exists(self.Annotation_dir_path):
                    self._initialise_dir(self.Annotation_dir_path)
                self.Annotation_Reader = globals()[Annotation_Reader_name]\
                    (self.Annotation_dir_path)
                print("Annotation_Reader loaded")
            else:
                raise ValueError(f"{Annotation_Reader_name} not found")
        else:
            self.Annotation_Reader = None
        
        """
        std EMA information
        """
        self.std_EMA_channels = [
            'tt_x', 'tt_y', 'tb_x', 'tb_y', 'td_x', 'td_y',
            'li_x', 'li_y', 'ul_x', 'ul_y', 'll_x', 'll_y'
        ]
        self.TVs = [
            "LA", "LP", "JA",
            "TTCL", "TMCL", "TRCL"
        ]
        self.EMA_TVs_channels = self.std_EMA_channels + self.TVs
        self.std_EMA_channels_TVs_indices_dict = \
            {channel : idx for idx, channel in enumerate(self.EMA_TVs_channels)}

        print(f"SpeakerData_Manager for {speaker_name} in {self.dataset_name} initialised.")
        print("---------")

    def _get_path_from_Dataset_Info(self, key):
        """
        key = "*_dir_path" from self.Current_Dataset_Info
        its corresponding value specifies how one can get to a certain dir from self.dataset_path

        get the corresponding value
        and concatenate them with self.dataset_path to get the absolute path of that certain dir
        """
        string_list = [self.speaker_name if x == "speaker_name" else x \
                       for x in self.Current_Dataset_Info[key]]
        absolute_path = os.path.join(self.dataset_path, *string_list)
        return absolute_path
    
    """
    Get raw data info

    raw data contains:
        - raw EMA
        - wav
        - annotation information
    """
    # for EMA
    def RAW_get_std_EMA_by_index(self, index):
        std_EMA = self.RawEMA_Reader.get_std_EMA_by_index(index)
        return std_EMA

    def RAW_get_std_EMA_list_by_index_list(self, index_list=None):
        """
        Get the list containing std_EMA accessible by index_list.
        If index_list is None, load all std EMA.

        Parameters
        ----------
        index_list : list
            List containing indices.

        Returns
        -------
        std_EMA_list : list
            Its elements are all tuples: (index, std_EMA).
            Shape of "std_EMA": (num_frames, 12).
        """
        std_EMA_list = self.RawEMA_Reader.\
            get_std_EMA_list_by_index_list(index_list=index_list)
        return std_EMA_list

    def RAW_get_nose_EMA_list_by_index_list(self, index_list=None):
        """
        Returns a list containing all nose_EMA accessible by self.all_index_list.
        If the dataset has no nose_EMA (determined by self.with_nose), returns None.

        Returns
        -------
        list_of_all_nose_EMA : list
            The elements of this list are all tuples: (index, nose_EMA).
            Shape of "nose_EMA": (num_frames, 2).
        """
        if self.with_nose:
            nose_EMA_list = self.RawEMA_Reader.\
                get_nose_EMA_list_by_index_list(index_list=index_list)
            return nose_EMA_list
        else:
            return None
    
    # for wav
    def RAW_get_wav_by_index(self, index):
        wav_path = os.path.join(self.wav_dir_path, index + ".wav")
        wav, _ = librosa.load(wav_path, sr=self.wav_sr)
        return wav
    
    def RAW_get_wav_list_by_index_list(self, index_list=None):
        wav_list = []
        if not index_list:
            index_list = self.all_index_list
        for index in index_list:
            current_wav_path = os.path.join(self.wav_dir_path, index + ".wav")
            wav, _ = librosa.load(current_wav_path, sr=self.wav_sr)
            print(f"wav {index} loaded")
            wav_list.append((index, wav))
        return wav_list
    
    # for annotation
    def RAW_get_xtrm_list_from_Annotations_by_index(self, index):
        xtrm_list = self.Annotation_Reader.get_xtrm_list_for_index(index)
        return xtrm_list
    
    def RAW_get_xtrm_list_from_wav_by_index(
            self, index,
            top_db=21,
            frame_length=512,
            hop_length=128
        ):
        wav = self.RAW_get_wav_by_index(index)

        wav_trimmed, frame_indices = librosa.effects.trim(
            wav,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )

        start_time = frame_indices[0] / self.wav_sr
        end_time = frame_indices[1] / self.wav_sr

        return [start_time, end_time]
    
    def _remove_silence(self, EMA, SF, sr, index):
        """
        Removing the preceding and trailing silence from EMA and SF.
        EMA should be downsampled with SF beforehand,
        Resulting in EMA and SF having the same sampling rate and length.
        If with annotations, extract xtrm_list from annotations.
        If not, extract xtrm_list from wav.

        Parameters
        ----------
        EMA : numpy.ndarray
            Shape: (num_frames, channels).
        SF : numpy.ndarray
            Shape: (num_frames, dimensions).
        sr : int
            Sampling rate of EMA and SF.
        index : str
            Index of EMA and SF.
            Needed for extracting wav or annotation.
        
        Returns
        -------
        trimmed_EMA : numpy.ndarray
            Shape: (num_frames, channels).
        trimmed_SF : numpy.ndarray
            Shape: (num_frames, dimensions).
        """
        if self.with_annotation:
            xtrm_list = self.RAW_get_xtrm_list_from_Annotations_by_index(index)
        else:
            xtrm_list = self.RAW_get_xtrm_list_from_wav_by_index(index)
        
        xtrm_frames = [
            int(xtrm_list[0] * sr),
            min(int(np.ceil(xtrm_list[1] * sr)), len(EMA))
        ]

        trimmed_EMA = EMA[xtrm_frames[0], xtrm_frames[1]]
        trimmed_SF = SF[xtrm_frames[0], xtrm_frames[1]]

        return trimmed_EMA, trimmed_SF
    

    """
    Functions related to channels and sensors of std EMA.
    """
    def _get_std_EMA_channels_TVs_indices(self, channels):
        if not channels:
            raise ValueError("No channels specified for lookup.")
        
        try:
            channels_indices = [self.std_EMA_channels_TVs_indices_dict[channel] for channel in channels]
        except KeyError as e:
            raise ValueError(f"Channel '{e.args[0]}' not found in self.std_EMA_channels.")
        return channels_indices
    
    def _get_std_EMA_sensors_indices(self, *sensors):
        channels = []
        for sensor in sensors:
            channels.extend([sensor + "_x", sensor + "_y"])
        channels_indices = self._get_std_EMA_channels_TVs_indices(channels)
        return channels_indices


    """
    Initialise directories under "prep_data".
    """
    def _initialise_dir(self, dir_path):
        """
        If dir_path exists, delete its content.
        If not, create this directory.
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            self._delete_dir_recursively(dir_path)
            os.makedirs(dir_path)
        print(f"{dir_path} initialised")
    
    def _delete_dir_recursively(self, dir_path):
        """
        Delete dir_path and its contents.
        """
        if not os.path.isdir(dir_path):
            raise ValueError(f"{dir_path} is not a dir")

        try:
            shutil.rmtree(dir_path)
            print(f"{dir_path} and its content deleted")
        except Exception as e:
            print(f"when deleting {dir_path}, error {e} occured")
    
    def create_prep_data_dir(self):
        """
        Check if prep_data exists.
        If not, create it.
        """
        if os.path.exists(self.prep_data_dir_path):
            print("prep_data already exists, would not make any changes to it")
        else:
            os.makedirs(self.prep_data_dir_path)
            print("prep_data dir created")
    
    def initialise_dir_under_prep_data(self, dir_name):
        """
        Initialise "prep_data/dir_name".
        If the directory exists, delete all its contents.
        If not, create the directory.
        """
        dir_path = os.path.join(self.prep_data_dir_path, dir_name)
        self._initialise_dir(dir_path)
    

    """
    Save data to prep_data.
    """
    def save_data_list_to_dir_under_prep_data(self, data_list, dir_name):
        """
        Save the data contained in "all_list" to "dir_name".

        Parameters
        ----------
        data_list : list
                   Its elements: (index, data).
                   "index" is needed to name the resulting file.
        dir_name : str
                   The name of the directory under which the data are to be saved.
                   The directory should be under "dataset_path/speaker_name/prep_data/".
                   The directory should already exists (this function will check it).
        """
        dir_path = os.path.join(self.prep_data_dir_path, dir_name)
        if not os.path.exists(dir_path):
            raise ValueError(f"{dir_path} does not exist")
        
        print(f"saving data to {dir_name}")
        for index, data in tqdm(data_list):
            data_path = os.path.join(dir_path, index + ".npy")
            np.save(data_path, data)
            print(f"{index}.npy saved")
        print("saving completed")

  
    """
    Load preped data
    """
    def load_data_list_from_dir_under_prep_data(self, dir_name, index_list=None):
        """
        Load all data accessible by index_list from a directory under "prep_data".
        If index_list is None, return all data accessible by self.all_index_list

        Parameters
        ----------
        dir_name : str
            Name of the directory under "prep_data".
        
        Returns
        -------
        all_data_list : list
            The elements of this list are all tuples: (index, data).
        """
        if not index_list:
            index_list = self.all_index_list
        data_list = []
        dir_path = os.path.join(self.prep_data_dir_path, dir_name)
        if not os.path.exists(dir_path):
            raise ValueError(f"{dir_path} not existed")
        print(f"loading data from {dir_name}")
        for index in tqdm(index_list):
            current_file_path = os.path.join(dir_path, index + ".npy")
            current_data = np.load(current_file_path)
            data_list.append((index, current_data))
        return data_list
    
    def load_EMA_sensors_from_dir_under_prep_data(
            self, dir_name, sensors_list=[], TVs_list=[], index_list=None
        ):
        if (sensors_list == []) and (TVs_list == []):
            raise ValueError("sensors_list and TVs_list cannot be both empty.")

        if sensors_list:
            required_indices = self._get_std_EMA_sensors_indices(*sensors_list)
        else:
            required_indices = []
        if TVs_list:
            TVs_indices = self._get_std_EMA_channels_TVs_indices(TVs_list)
            required_indices += TVs_indices
        
        if not index_list:
            index_list = self.all_index_list

        data_list = []

        dir_path = os.path.join(self.prep_data_dir_path, dir_name)
        if not os.path.exists(dir_path):
            raise ValueError(f"{dir_path} not existed")
        
        print(f"loading EMA from {dir_name}")
        for index in tqdm(index_list):
            current_file_path = os.path.join(dir_path, index + ".npy")
            current_data = np.load(current_file_path)[:, required_indices]
            data_list.append((index, current_data))
        return data_list
    

    """
    remove silence with phone sequence
    and z-score normalise (optional)
    """
    def last_process(
            self,
            data_list,
            PS_list,
            if_normalise=False,
            new_std=2,
            more_stats=False
        ):
        """
        根据从标注文件中提取出来的 Phone Sequence 信息, 对 data_list 中的 data 进行去静音段操作
            - Phone Sequence 中表示静音段的 silence token, 需要在 AAI_Dataset_Info.json 中进行指定
            - Data_list 中的元素都是这样的元组: (index, data)
                - data 都是 numpy.ndarray, 形状为 (num_frames, channels)

        若 if_normalise 为 True, 则对 data 去均值, 方差设为 new_std
        若 more_stats 为 True, 返回 stats_list, stats_list 的元素皆为元组, 其内容为 (index, mean, std)
        """
        if len(data_list) != len(PS_list):
            raise ValueError("Data_list_1 与 PS_list 长度不一致")
        
        data_list_indices = [idx for idx, _ in data_list]
        PS_indices = [idx for idx, _ in PS_list]
        if data_list_indices != PS_indices:
            raise ValueError("EMA 与 PS 的索引序列不一致")
        
        data_list_data = [data for _, data in data_list]
        PS_data = [data for _, data in PS_list]

        new_data_list = []
        stats_list = []
        new_PS_list = []

        print("performing last process")
        for i in tqdm(range(len(PS_data))):
            index = PS_indices[i]

            # 去除静音段
            ps = PS_data[i]
            mask = ps != self.silence_token
            filtered_data = data_list_data[i][mask]
            filtered_ps = ps[mask]

            # z-score 归一化
            if if_normalise:
                filtered_data, mean, std = normalise_np(
                    filtered_data, new_std=new_std
                )
            
            new_data_list.append((index, filtered_data))
            stats_list.append((index, mean, std))
            new_PS_list.append((index, filtered_ps))

        if more_stats:
            return new_data_list, new_PS_list, stats_list
        else:
            return new_data_list, new_PS_list

    """
    split data
    """
    def split_speaker_data(
            self, train_ratio, val_ratio,
            test_ratio=0.0, with_speaker_name=False
        ):
        """
        Split speaker data according to self.all_index_list.
        Returns the resulting index list.
        If test_ratio=0, test_list would be [].
        If with speaker_name, the elements of the resulting list would be (speaker_name, index).

        Parameters
        ----------
        train_ratio : float
        val_ratio : float
        test_ratio : float
            test_ratio could be 0.
        
        Returns
        -------
        train_list : list
            If with_speaker_name is False, the elements of this list would simply be indices.
            If it is True, the elements would be (speaker_name, index).
        val_list : list
            Same with train_list.
        test_list : list
            If test_ratio is 0, test_list would be [].
        """
        if train_ratio + val_ratio + test_ratio > 1:
            raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must not exceed 1.")
        
        total_num = len(self.all_index_list)
        train_num = int(total_num * train_ratio)
        val_num = int(total_num * val_ratio)

        if not with_speaker_name:
            index_list = self.all_index_list.copy()
        else:
            index_list = [(self.speaker_name, index) for index in self.all_index_list]
        random.shuffle(index_list)

        train_list = index_list[:train_num]

        if test_ratio:
            val_list = index_list[train_num : train_num + val_num]
            test_list = index_list[train_num + val_num:]
        else:
            val_list = index_list[train_num :]
            test_list = []

        return train_list, val_list, test_list
    
    def _save_split_list_to_txt(self, split_list, save_path):
        """
        The saved format: dataset_name speaker_name index.
        """
        with open(save_path, "w", encoding="utf-8") as f:
            for index in split_list:
                str_to_write = self.dataset_name + " " + self.speaker_name + " " + index
                f.write(str_to_write + "\n")
        print(f"{save_path} saved")

    def split_speaker_data_and_save(
            self, train_ratio, val_ratio, test_ratio, save_dir_path=None
        ):
        if save_dir_path is None:
            save_dir_path = os.getcwd()

        train_list, val_list, test_list = self.split_speaker_data(
            train_ratio, val_ratio, test_ratio
        )
        print(f"Data split completed: {len(train_list)} training samples, {len(val_list)} validation samples, {len(test_list)} testing samples.")

        train_list_save_path = os.path.join(save_dir_path, "train_list.txt")
        val_list_save_path = os.path.join(save_dir_path, "val_list.txt")

        self._save_split_list_to_txt(train_list, train_list_save_path)
        self._save_split_list_to_txt(val_list, val_list_save_path)

        if test_list:
            test_list_save_path = os.path.join(save_dir_path, "test_list.txt")
            self._save_split_list_to_txt(test_list, test_list_save_path)


class MOCHA_TIMIT_RawData_Extractor(SpeakerData_Manager):
    """
    Used to convert NIST files into wav files,
    and .lab files into .TextGrid files.
    """
    def __init__(self, dataset_path, speaker_name):
        super().__init__(dataset_path, speaker_name)
    
    def read_lab(self, lab_path):
        phones = []
        with open(lab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    start, end, phone = line.split()
                    phones.append([start, end, phone])
        return phones
    
    def convert_all_labs_into_TextGrid(self):
        self._initialise_dir(self.Annotation_dir_path)
        for index in self.all_index_list:
            print(f"converting lab '{index}'")
            lab_path = os.path.join(self.raw_EMA_dir_path, index + ".lab")
            phones = self.read_lab(lab_path)
            annotation_path = os.path.join(
                self.Annotation_dir_path, index + ".TextGrid"
            )
            self.generate_textgrid(phones, annotation_path)
        print("all annotation info extracted and saved")

    def generate_textgrid(self, phones, output_file):
        # 创建 TextGrid 文件的头部
        with open(output_file, 'w') as f:
            f.write('File type = "ooTextFile"\n')
            f.write('Object class = "TextGrid"\n\n')
            
            # 假设整个音频的时长是从 0 到最大结束时间
            xmin = 0.0
            xmax = max([x[1] for x in phones])
            f.write(f'xmin = {xmin}\n')
            f.write(f'xmax = {xmax}\n')

            # 输出层数
            f.write('tiers? <exists>\n')
            f.write('size = 1\n')
            f.write('item []:\n')

            # 单词层
            f.write('    item [1]:\n')
            f.write('        class = "IntervalTier"\n')
            f.write('        name = "phones"\n')
            f.write(f'        xmin = {xmin}\n')
            f.write(f'        xmax = {xmax}\n')
            f.write(f'        intervals: size = {len(phones)}\n')
            for i, (start, end, phoneme) in enumerate(phones):
                f.write(f'        intervals [{i + 1}]:\n')
                f.write(f'            xmin = {start}\n')
                f.write(f'            xmax = {end}\n')
                f.write(f'            text = "{phoneme}"\n')
    


class HPRC_RawData_Extractor(SpeakerData_Manager):
    """
    Used to extract wav and annotation from HPRC .mat files.

    wav_save_dir : raw_data/wav
    annotation_save_dir : raw_data/TextGrid
    """
    def __init__(self, dataset_path, speaker_name):
        super().__init__(dataset_path, speaker_name)
    
    def extract_all_wav_and_save(self):
        self._initialise_dir(self.wav_dir_path)
        print(f"dir '{self.wav_dir_path}' initialised")
        all_wav_list = self.RawEMA_Reader.get_all_wav()
        for index, wav in all_wav_list:
            wav_path = os.path.join(self.wav_dir_path, index + ".wav")
            sf.write(wav_path, wav, samplerate=self.wav_sr)
            print(f"wav for '{index}' saved")
        print("wav extraction completed")

    def extract_all_Annotation_and_save(self):
        self._initialise_dir(self.Annotation_dir_path)
        all_annotation_info = self.RawEMA_Reader.extract_all_annotation_info()
        for index, annotation_info in all_annotation_info:
            annotation_path = os.path.join(self.Annotation_dir_path, index + ".TextGrid")
            self.generate_textgrid(annotation_info[0], annotation_info[1], annotation_path)
        print("all annotation info extracted and saved")

    def generate_textgrid(self, words, phones, output_file):
        # 创建 TextGrid 文件的头部
        with open(output_file, 'w') as f:
            f.write('File type = "ooTextFile"\n')
            f.write('Object class = "TextGrid"\n\n')
            
            # 假设整个音频的时长是从 0 到最大结束时间
            xmin = 0.0
            xmax = max(max([x[1] for x in phones]), max([x[1] for x in words]))
            f.write(f'xmin = {xmin}\n')
            f.write(f'xmax = {xmax}\n')

            # 输出层数
            f.write('tiers? <exists>\n')
            f.write('size = 2\n')
            f.write('item []:\n')
            
            # word tier
            f.write('    item [1]:\n')
            f.write('        class = "IntervalTier"\n')
            f.write('        name = "words"\n')
            f.write(f'        xmin = {xmin}\n')
            f.write(f'        xmax = {xmax}\n')
            f.write(f'        intervals: size = {len(words)}\n')
            for i, (start, end, word) in enumerate(words):
                f.write(f'        intervals [{i + 1}]:\n')
                f.write(f'            xmin = {start}\n')
                f.write(f'            xmax = {end}\n')
                f.write(f'            text = "{word}"\n')

            # 单词层
            f.write('    item [2]:\n')
            f.write('        class = "IntervalTier"\n')
            f.write('        name = "phones"\n')
            f.write(f'        xmin = {xmin}\n')
            f.write(f'        xmax = {xmax}\n')
            f.write(f'        intervals: size = {len(phones)}\n')
            for i, (start, end, phoneme) in enumerate(phones):
                f.write(f'        intervals [{i + 1}]:\n')
                f.write(f'            xmin = {start}\n')
                f.write(f'            xmax = {end}\n')
                f.write(f'            text = "{phoneme}"\n')


def normalise_np(np_array, new_std=1):
    """
    Along axis 0, set the mean of "np_array" to 0, and std to "new_std".

    Parameters
    ----------
    np_array : numpy.ndarray
        Has to be the shape of (num_frames, dimensions).
    new_std : int
        The new std.
    
    Returns
    -------
    normalised_np_array : numpy.ndarray
        Normalised np_array, whose shape is the same as the input "np_array".
    mean : numpy.ndarray
        Original mean along axis 0, its shape is (dimensions,).
    std : numpy.ndarray
        Original std along aixs 0, its shape is (dimensions,).
    """
    mean = np.mean(np_array, axis=0)  # shape (dimension,)
    std = np.std(np_array, axis=0)    # shape (dimension,)

    if np.any(np.isclose(std, 0)):
        std[np.isclose(std, 0)] = 1e-10

    normalised_np_array = (np_array - mean) * (new_std / std)

    return normalised_np_array, mean, std


class PhoneSequenceExtractor(SpeakerData_Manager):
    """
    Used to extract phone sequence according to
    the sampling rate of a certain speech feature.
    """
    def __init__(
            self,
            dataset_path, speaker_name,
            item_name="phones"
        ):
        super().__init__(dataset_path, speaker_name)
        self.item_name = item_name
    
    def extract_phone_sequence_for(
            self, feature_name=None
        ):
        """
        需要标注文件
        若 feature_name 为 None, 那么所生成的 Phone Sequence, 采样率和 df_preped_EMA 相同
        """
        if feature_name:
            numpy_ndarray_list = self.load_data_list_from_dir_under_prep_data(
                feature_name
            )
            phone_sequence_dir = os.path.join(
                self.prep_data_dir_path,
                "PhoneSeq_" + feature_name
            )
        else:
            numpy_ndarray_list = self.load_data_list_from_dir_under_prep_data(
                "df_preped_EMA"
            )
            phone_sequence_dir = os.path.join(
                self.prep_data_dir_path,
                "PhoneSeq"
            )

        self._initialise_dir(phone_sequence_dir)

        for index, numpy_ndarray in numpy_ndarray_list:
            print(f"extracting phone sequence for '{index}'")
            textgrid_path = os.path.join(
                self.Annotation_dir_path, index + ".TextGrid"
            )
            phone_sequence = self._generate_phone_sequence_for_numpy_ndarray(
                numpy_ndarray, textgrid_path
            )
            np.save(
                os.path.join(phone_sequence_dir, index),
                phone_sequence
            )
        print("phone sequence extraction completed")
    
    def _generate_phone_sequence_for_numpy_ndarray(
            self, numpy_ndarray, textgrid_path 
        ):
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        phone_tier = tg.getFirst(self.item_name)

        num_frames = numpy_ndarray.shape[0]
        audio_duration = phone_tier.maxTime
        frame_time_step = audio_duration / num_frames

        phone_sequence = np.full(num_frames, fill_value='sil', dtype='<U20')

        for interval in phone_tier:
            start_time = interval.minTime
            end_time = interval.maxTime
            phoneme = interval.mark
            
            start_frame = int(np.floor(start_time / frame_time_step))
            end_frame = int(np.floor(end_time / frame_time_step))

            phone_sequence[start_frame:end_frame+1] = phoneme
        
        return phone_sequence
    
    def collect_all_phones(self):
        speaker_phone_set = set()
        for index in self.all_index_list:
            current_set = set()
            print(f'collecting phones for {index}')
            tg_path = os.path.join(self.Annotation_dir_path, index + ".TextGrid")

            tg = textgrid.TextGrid.fromFile(tg_path)
            phone_tier = tg.getFirst(self.item_name)

            for interval in phone_tier:
                phone = interval.mark
                current_set.add(phone)
            speaker_phone_set.update(current_set)
        return speaker_phone_set