import os

import torch
import numpy as np

from AAI_Utils import *
from AAI_Base_Models import AAIModel_BBF as AAI_Model
from AAI_Loss_Function import *
from AAI_RMSE_and_CC import *


class SimpleTrainer():
    def __init__(
            self,
            train_dataloader,
            val_dataloader,
            config,
            exp_dir=None,
            save_ckpt=True
        ):

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.exp_dir = exp_dir if exp_dir else os.getcwd()
        self.save_ckpt = save_ckpt

        self.epoch_num = self.config["epoch_num"]
        self.input_dim = self.config["input_dim"]
        self.output_dim = self.config["output_dim"]
        self.learning_rate = self.config["learning_rate"]
        self.beta1 = self.config["beta1"]
        self.beta2 = self.config["beta2"]

        # 创建 train_record 和 val_record
        self.train_record_path = os.path.join(
            self.exp_dir, "train_record.txt"
        )
        self.val_record_path = os.path.join(
            self.exp_dir, "val_record.txt"
        )
        if os.path.exists(self.train_record_path):
            os.remove(self.train_record_path)
        if os.path.exists(self.val_record_path):
            os.remove(self.val_record_path)
        
        # 创建 checkpoint dir
        self.ckpt_dir = os.path.join(self.exp_dir, "ckpts")
        initialise_folder(self.ckpt_dir)

        # 设置 device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"using device {self.device}")

        # 创建模型与优化器
        self.build_model()
    

    def build_model(self):
        self.model = AAI_Model(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            self.learning_rate, [self.beta1, self.beta2]
        )
        self.loss_function = Masked_MSE_Loss()

        self.model.to(self.device)
        self.model.double()


    def train(self):
        # 获取训练开始时间
        MyTimer = Timer()
        MyTimer.start()

        print("开始训练")

        with open(self.train_record_path, 'w') as f:
            f.write("实验文件夹: {}\n".format(self.exp_dir))
            f.write("训练开始时间: {}\n".format(MyTimer.formatted_start_time))
            f.write("\n")
        
        # 对 epoch 进行迭代
        for current_epoch in range(self.epoch_num):
            print(f"Epoch【{current_epoch}】")

            # epoch 训练阶段
            self._epoch_step(mode="train", current_epoch=current_epoch)

            # epoch 验证阶段
            self._epoch_step(mode="val", current_epoch=current_epoch)

        # 获取训练结束时间
        MyTimer.end()
        with open(self.train_record_path, 'a') as f:
            f.write("\n")
            f.write("训练结束时间: {}\n".format(MyTimer.formatted_end_time))
            f.write("训练花费了 {:.2f} 个小时\n".format(MyTimer.time_delta_in_hours))

        print("【模型训练结束】")
    

    def _epoch_step(self, mode, current_epoch):
        if mode == "train":
            self.model.train()
            dataloader = self.train_dataloader
            record_path = self.train_record_path
        else:
            self.model.eval()
            dataloader = self.val_dataloader
            record_path = self.val_record_path
        
        epoch_loss_list = []
        epoch_RMSE_list = []
        epoch_PCC_list = []

        # 在一个 epoch 中, 对所有 batch 进行迭代
        for current_batch, batch_data in enumerate(dataloader):
            print(f"Epoch【{current_epoch}】, {mode} Batch【{current_batch}】")

            loss_batch_avg, RMSE_batch_avg, PCC_batch_avg = self._batch_step(batch_data, mode)
            epoch_loss_list.append(loss_batch_avg)
            epoch_RMSE_list.append(RMSE_batch_avg)
            epoch_PCC_list.append(PCC_batch_avg)
            print("  Batch Loss = {:.6F}".format(loss_batch_avg))

        # 计算该 epoch 的平均 loss
        loss_epoch_avg = np.mean(epoch_loss_list)
        RMSE_epoch_avg = np.mean(epoch_RMSE_list)
        PCC_epoch_avg = np.mean(epoch_PCC_list)
        # 打印
        print("Epoch {} Loss: {:.6F}".format(mode, loss_epoch_avg))
        # 将 loss 写入 record
        with open(record_path, "a") as f:
            f.write("Epoch {}, Loss=【{:.6f}】, RMSE=【{:.6f}】, CC=【{:.6f}】".format(
                current_epoch,
                loss_epoch_avg,
                RMSE_epoch_avg,
                PCC_epoch_avg
            ))
            f.write("\n")
        
        # 若为 train 阶段, 就保存模型参数
        if mode == "train" and self.save_ckpt:
            ckpt_name = f"epoch_{current_epoch}.pth"
            ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
            torch.save({"model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict()}, ckpt_path)
    

    def _batch_step(self, batch_data, mode):
        batch_input, batch_output, batch_durations, batch_masks = batch_data

        batch_input = batch_input.to(self.device)
        batch_output = batch_output.to(self.device)
        batch_durations = batch_durations.to(self.device)
        batch_masks = batch_masks.to(self.device)

        batch_pred = self.model(batch_input)
        batch_pred.to(self.device)

        loss_batch_avg = self.loss_function(
            batch_pred, batch_output, batch_masks, self.device
        )
        RMSE_batch_avg, CC_batch_avg = get_batch_avg_RMSE_and_PCC(
            batch_pred, batch_output, batch_durations
        )

        if mode == "train":
            self.optimizer.zero_grad()
            loss_batch_avg.backward()
            self.optimizer.step()
        
        return loss_batch_avg.item(), RMSE_batch_avg.item(), CC_batch_avg.item()