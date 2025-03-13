import torch
import torch.nn as nn
import torch.nn.functional as F


class AAIModel_BBF(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AAIModel_BBF, self).__init__()

        self.hidden_size_BLSTM1 = 128
        self.hidden_size_BLSTM2 = 128

        self.BLSTM1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_size_BLSTM1,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        self.BLSTM2 = nn.LSTM(
            input_size=self.hidden_size_BLSTM1 * 2,
            hidden_size=self.hidden_size_BLSTM2,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=0.5)
        self.FC = nn.Linear(
            self.hidden_size_BLSTM2 * 2,
            output_dim
        )
    
    def forward(self, x):
        out, _ = self.BLSTM1(x)
        out, _ = self.BLSTM2(out)
        out = self.dropout(out)
        out = self.FC(out)

        return out


class AAIModel_FFBBF(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AAIModel_FFBBF, self).__init__()

        self.FC1= nn.Linear(input_dim, 512)
        self.FC2 = nn.Linear(512, 512)

        self.BLSTM1 = nn.LSTM(
            input_size=512,
            hidden_size=300,
            num_layers=4,
            batch_first=True,
            bidirectional=True
        )
        self.BLSTM2 = nn.LSTM(
            input_size=600,
            hidden_size=300,
            num_layers=4,
            batch_first=True,
            bidirectional=True
        )

        self.outputLayer = nn.Linear(600, output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.FC1(x))
        x = self.dropout(x)

        x = F.relu(self.FC2(x))
        x = self.dropout(x)

        out, _ = self.BLSTM1(x)
        out, _ = self.BLSTM2(out)

        out = self.outputLayer(out)

        return out


class AAIModel_CNN_BBF(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AAIModel_CNN_BBF, self).__init__()

        # 一维卷积在时间维度上做卷积, 因此需要将 length 维度放到最后
        self.Conv_1 = nn.Conv1d(input_dim, 128, kernel_size=1, stride=1, padding=0)
        self.Conv_2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.Conv_3 = nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)
        self.Conv_4 = nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3)
        self.Conv_5 = nn.Conv1d(128, 128, kernel_size=9, stride=1, padding=4)

        # 640 是 128 的 5 倍
        self.BLSTM_1 = nn.LSTM(
            input_size=640,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        self.BLSTM_2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        self.FC = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self , x):
        """
        x 的形状为 (batch_size, length, input_dim)
        一维卷积在时间维度上做卷积, 因此需要先将 length 维度放到最后
        """
        x = x.permute(0, 2, 1)

        x1 = F.relu(self.Conv_1(x))
        x2 = F.relu(self.Conv_2(x1))
        x3 = F.relu(self.Conv_3(x2))
        x4 = F.relu(self.Conv_4(x3))
        x5 = F.relu(self.Conv_5(x4))

        x_cat = torch.cat((x1, x2, x3, x4, x5), 1)
        # 把维度再变回来
        x_cat = x_cat.permute(0, 2, 1)

        out, _ = self.BLSTM_1(x_cat)
        out, _ = self.BLSTM_2(out)

        out = self.FC(out)

        return out