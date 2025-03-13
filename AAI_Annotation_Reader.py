import os
import bisect

import textgrid


class Annotation_Reader():
    def __init__(
            self,
            annotation_dir,
            annotation_file_type,
            all_index_list=None
        ):
        self.annotation_dir = annotation_dir
        self.annotation_file_type = annotation_file_type

        if all_index_list:
            self.all_index_list = all_index_list
        else:
            self.all_index_list = [os.path.splitext(x)[0] for x in os.listdir(annotation_dir) \
                                    if x.endswith(self.annotation_file_type)]

    def get_xtrm_list_for_index(self, index):
        raise NotImplementedError
    
    def get_phone_from_time_for_index(self, index, time_point):
        raise NotImplementedError
    
    def get_all_xtrm_list(self):
        list_for_xtrm_list = []
        for index in self.all_index_list:
            xtrm_list = self.get_xtrm_list_for_index(index)
            list_for_xtrm_list.append((index, xtrm_list))
        return list_for_xtrm_list


class BY2023_Annotation_Reader(Annotation_Reader):
    def __init__(
            self,
            annotation_dir,
            annotation_file_type=".TextGrid",
            all_index_list=None
        ):
        
        super().__init__(
            annotation_dir,
            annotation_file_type,
            all_index_list
        )
    
    def get_xtrm_list_for_index(self, index):
        annotation_path = os.path.join(
            self.annotation_dir, index + self.annotation_file_type
        )

        tg = textgrid.TextGrid.fromFile(annotation_path)
        word_tier = tg.getFirst("words")

        if word_tier.intervals[0].mark == "":
            start_time = word_tier.intervals[0].maxTime
        else:
            raise ValueError(f"the first part of {index} is not empty")
        if word_tier.intervals[-1].mark == "":
            end_time = word_tier.intervals[-1].minTime
        else:
            raise ValueError(f"the last part of {index} is not empty")
        
        return [start_time, end_time]
    
    def get_phone_from_time_for_index(self, index, time_point):
        annotation_path = os.path.join(
            self.annotation_dir, index + self.annotation_file_type
        )

        tg = textgrid.TextGrid.fromFile(annotation_path)

        if time_point < tg.minTime or time_point > tg.maxTime:
            raise ValueError(
                f"time_point {time_point} out of bounds for {index}."
                f"Valid range: [{tg.minTime}, {tg.maxTime}]"
            )
        
        phone_tier = tg.getFirst("phones")
        intervals = phone_tier.intervals
        start_times = [interval.minTime for interval in intervals]
        idx = bisect.bisect_left(start_times, time_point)
        if idx > 0 and intervals[idx - 1].minTime <= time_point <= intervals[idx - 1].maxTime:
            interval = intervals[idx - 1]
            return interval.mark
        
        raise ValueError(f"time_point {time_point} not found in any interval for {index}")


class MOCHA_TIMIT_Annotation_Reader(Annotation_Reader):
    def __init__(
            self,
            annotation_dir,
            annotation_file_type=".lab",
            all_index_list=None
        ):
        
        super().__init__(
            annotation_dir,
            annotation_file_type,
            all_index_list
        )
    
    def get_xtrm_list_for_index(self, index):
        margin = 0
        annotation_path = os.path.join(
            self.annotation_dir, index + self.annotation_file_type
        )

        with open(annotation_path, "r", encoding="utf-8") as file:
            labels = [
                row.strip('\n').strip('\t').replace(' 26 ', '').split(' ') \
                    for row in file
            ]

        xtrm_list = [
            max(float(labels[0][1]) - margin, 0),
            float(labels[-1][0]) + margin
        ]

        return xtrm_list
    
    def get_phone_from_time_for_index(self, index, time_point):
        annotation_path = os.path.join(
            self.annotation_dir, index + self.annotation_file_type
        )

        with open(annotation_path, "r", encoding="utf-8") as file:
            labels = [
                row.strip('\n').strip('\t').replace(' 26 ', '').split(' ') \
                    for row in file
            ]
        
        start_times = [float(label[0]) for label in labels]
        idx = bisect.bisect_left(start_times, time_point)
        if idx > 0 and float(labels[idx - 1][0]) <= time_point <= float(labels[idx - 1][1]):
            label = labels[idx - 1]
        return label[-1]


class HPRC_Annotation_Reader(Annotation_Reader):
    def __init__(
            self,
            annotation_dir,
            annotation_file_type=".TextGrid",
            all_index_list=None
        ):
        
        super().__init__(
            annotation_dir,
            annotation_file_type,
            all_index_list
        )