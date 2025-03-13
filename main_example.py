import json

from AAI_SpeakerData_Manager import SpeakerData_Manager
from AAI_Dataloder_Simple import get_dataloader
from Trainer_Simple import SimpleTrainer


dataset_path = r""
speaker_name = ""


# loading configuration file for the experiment
config_path = r"config.json"
with open(config_path, "r") as f:
    config = json.load(f)


def main():
    MyManager = SpeakerData_Manager(dataset_path, speaker_name)

    # dividing train- and val-dataset
    train_list, val_list, _ = MyManager.split_speaker_data(
        train_ratio=0.8, val_ratio=0.2
    )

    # loading data
    MyTrainLoader = get_dataloader(
        SpeakerData_Manager=MyManager,
        index_list=train_list,
        SF_name=config["SF_name"],
        batch_size=config["train_batch_size"],
        config=config
    )
    MyValLoader = get_dataloader(
        SpeakerData_Manager=MyManager,
        index_list=val_list,
        SF_name=config["SF_name"],
        batch_size=config["val_batch_size"],
        config=config
    )

    # model training
    MyTrainer = SimpleTrainer(
        train_dataloader=MyTrainLoader,
        val_dataloader=MyValLoader,
        config=config,
        save_ckpt=True
    )
    MyTrainer.train()


if __name__ == "__main__":
    main()