import os
import json
import random
from collections import defaultdict

from AAI_SpeakerData_Manager import SpeakerData_Manager


class Dataset_Manager():
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset_name = os.path.basename(os.path.normpath(dataset_path))

        """
        load dataset information
        """
        Dataset_Info_path = r"AAI_Dataset_Info.json"
        with open(Dataset_Info_path, "r") as f:
            Dataset_Info = json.load(f)
        self.Current_Dataset_Info = Dataset_Info["Dataset_Info_List"]\
            [Dataset_Info["Dataset_Name_List"].index(self.dataset_name)]
        
        self.speaker_list = self.Current_Dataset_Info["Speaker_List"]
        self.speaker_num = len(self.speaker_list)
        self.speaker_name_to_idx_dict = {
            speaker_name:idx for idx, speaker_name in enumerate(self.speaker_list)
        }

        self.SpeakerData_Manager_list = []
        for speaker_name in self.speaker_list:
            My_SpeakerData_Manager = SpeakerData_Manager(
                self.dataset_path, speaker_name
            )
            self.SpeakerData_Manager_list.append(My_SpeakerData_Manager)
        print("各 SpeakerData Manager 加载完成")
    
    def split_data_for_speaker_independent_scenario(
            self,
            train_ratio,
            val_ratio,
            test_speaker_num,
            save_dir=None
        ):

        if (train_ratio + val_ratio) != 1.0:
            raise ValueError("train ratio 和 val ratio 之和不是 1")
        if save_dir is None:
            save_dir = os.getcwd()
        
        test_speakers = random.sample(self.speaker_list, k=test_speaker_num)
        train_speakers = [x for x in self.speaker_list if x not in test_speakers]

        all_train_list = []
        all_val_list = []

        for speaker_name in train_speakers:
            speaker_idx = self.speaker_name_to_idx_dict[speaker_name]
            current_Manager = self.SpeakerData_Manager_list[speaker_idx]
            current_train_list, current_val_list, _ = current_Manager.split_speaker_data(
                train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=0.0,
                with_speaker_name=True
            )
            all_train_list += current_train_list
            all_val_list += current_val_list
        
        train_list_save_path = os.path.join(save_dir, "train_list.txt")
        val_list_save_path = os.path.join(save_dir, "val_list.txt")
        test_speakers_save_path = os.path.join(save_dir, "test_speaker.json")
        
        self._save_list_to_txt(all_train_list, train_list_save_path)
        self._save_list_to_txt(all_val_list, val_list_save_path)
        with open(test_speakers_save_path, "w") as f:
            json.dump(test_speakers, f)
    
    def convert_txt_to_speaker_indices_dict(self, txt_path):
        """
        返回一个字典, 键为 speaker_name, 值为该 speaker 的 index_list
        """
        all_index_list = self._convert_txt_to_list(txt_path)
        speaker_indices_dict = defaultdict(list)

        for speaker_name, idx in all_index_list:
            speaker_indices_dict[speaker_name].append(idx)
        return speaker_indices_dict

    def _save_list_to_txt(self, the_list, save_path):
        with open(save_path, "w", encoding="utf-8") as f:
            for speaker_name, idx in the_list:
                str_to_write = speaker_name + " " + idx
                f.write(str_to_write + "\n")
        print(f"{save_path} saved")
    
    def _convert_txt_to_list(self, txt_path):
        resulting_list = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                speaker_name, idx = line.split()
                resulting_list.append((speaker_name, idx))
        return resulting_list