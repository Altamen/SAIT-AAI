import os
from datetime import datetime


# 用于初始化文件夹
def initialise_folder(folder_path):
    """
    如果路径所指定的文件夹存在, 就清空该文件夹

    如果不存在, 就创建该文件夹
    """

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        delete_folder_and_contents(folder_path)
        os.makedirs(folder_path)

def delete_folder_and_contents(folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            delete_folder_and_contents(item_path)
    
    os.rmdir(folder_path)


# 用于计时
class Timer():
    def __init__(self):
        self.start_time = None
        self.formatted_start_time = None

        self.end_time = None
        self.formatted_end_time = None

        self.time_delta = None
        self.time_delta_in_hours = None
    
    def _get_formatted_time(self, time):
        return time.strftime("%Y-%m-%d %H:%M:%S")
    
    def start(self):
        current_time = datetime.now()
        self.start_time = current_time
        self.formatted_start_time = \
            self._get_formatted_time(self.start_time)
    
    def end(self):
        current_time = datetime.now()
        self.end_time = current_time
        self.formatted_end_time = \
            self._get_formatted_time(self.end_time)

        self.time_delta = self.end_time - self.start_time
        self.time_delta_in_hours = \
            self.time_delta.total_seconds() / 3600