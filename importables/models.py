# pylint: disable-all
"""This module contains all experimented models and train per batch functions to keep track changes"""
import os
import json
import torch
import torch.nn
import torchvision
import torch.utils.data
from tqdm.auto import tqdm
import sklearn.preprocessing
import torch.utils.tensorboard.writer
from importables.utilities import Utilities
from importables.custom_types import (
    Tensor
)
# GROUND RULES:
# 1: CNN IS PRETRAINED ONLY

# Parameter not to test:
# 1: Learning rate (default is suffice)
# 2: Batch size (not much to test, GPU memory limited)
# 3: Epoch (just use the best target)

# Parameters to test:
# 1: Timestep (How much frame for detection, less = faster, more = smoother)
# 2: Model layers
# 3: With mask / not
# 4: With cropping / Not
# 5: bounding value
# 6: mv vs of


def default_train_one_epoch(
    model: torch.nn.Module,
    current_epoch: int,
    train_dl: torch.utils.data.DataLoader,
    test_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Adam,
    criterion: torch.nn.CrossEntropyLoss,
    summary_writer: torch.utils.tensorboard.writer.SummaryWriter,
    training_parameters: dict,
    encoder: sklearn.preprocessing.LabelEncoder,
    dataset_map: dict,
) -> None:

    train_length: int = len(train_dl)
    test_length: int = len(test_dl)

    confusion_matrix_sum_epoch = {
        "train": [],
        "test": []
    }

    memory_avg: dict = {}
    with tqdm(total=train_length) as pbar:

        for i, data in enumerate(train_dl):
            start_time: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
            end_time: torch.cuda.Event = torch.cuda.Event(enable_timing=True)

            start_time.record()  # type: ignore

            inputs: Tensor = data[training_parameters["data_selector"]].to("cuda", non_blocking=True)
            labels: Tensor = data["label"].to("cuda", non_blocking=True)

            optimizer.zero_grad()
            output: Tensor = model(inputs)
            loss: Tensor = criterion(output, labels)
            loss.backward()
            optimizer.step()

            end_time.record()  # type: ignore

            torch.cuda.synchronize()

            total_seconds: float = start_time.elapsed_time(end_time) / 1000

            with torch.no_grad():
                y_true = labels.detach().cpu().numpy()
                y_pred = torch.argmax(output.detach().softmax(dim=1), dim=1).cpu().numpy()
                memory_avg, summary_writer = Utilities.write_all_summary_iteration(
                    summary_writer=summary_writer,
                    y_true=y_true,
                    y_pred=y_pred,
                    loss=loss.item(),
                    fps=1/total_seconds,
                    encoder=encoder,
                    dataset_map=dataset_map,
                    step=current_epoch*train_length+i,
                    epoch=current_epoch,
                    memory_average=memory_avg,
                    is_train=True,
                )

            pbar.set_description("Training: Epoch {} | Loss: {:.3f} | Accuracy A/Wa: {:.3f}/{:.3f} | FPS {:.3f}".format(
                current_epoch,
                memory_avg[current_epoch]["Loss"]/(i+1),
                memory_avg[current_epoch]["Accuracy"]["Average"]/(i+1),
                memory_avg[current_epoch]["Accuracy"]["Weighted Average"]/(i+1),
                memory_avg[current_epoch]["FPS"]/(i+1))
            )
            pbar.update()

    Utilities.write_confusion_matrix_epoch(
        summary_writer=summary_writer,
        encoder=encoder,
        dataset_map=dataset_map,
        epoch=current_epoch,
        memory_average=memory_avg,
        is_train=True,
    )
    confusion_matrix_sum_epoch["train"] = memory_avg[current_epoch]["Confusion Matrix"].tolist()

    criterion_eval: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss()
    memory_avg: dict = {}
    with torch.no_grad():

        with tqdm(total=test_length) as pbar:
            for i, data in enumerate(test_dl):
                start_time: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
                end_time: torch.cuda.Event = torch.cuda.Event(enable_timing=True)

                start_time.record()  # type: ignore

                inputs: Tensor = data[training_parameters["data_selector"]].to("cuda", non_blocking=True)
                labels: Tensor = data["label"].to("cuda", non_blocking=True)

                output: Tensor = model(inputs)
                loss: Tensor = criterion_eval(output, labels)

                end_time.record()  # type: ignore

                torch.cuda.synchronize()

                total_seconds: float = start_time.elapsed_time(end_time) / 1000

                memory_avg, summary_writer = Utilities.write_all_summary_iteration(
                    summary_writer=summary_writer,
                    y_true=labels.detach().cpu().numpy(),
                    y_pred=torch.argmax(output.detach().softmax(dim=1), dim=1).cpu().numpy(),
                    loss=loss.item(),
                    fps=1/total_seconds,
                    encoder=encoder,
                    dataset_map=dataset_map,
                    step=current_epoch*test_length+i,
                    epoch=current_epoch,
                    memory_average=memory_avg,
                    is_train=False,
                )

                pbar.set_description("Testing: Epoch {} | Loss: {:.3f} | Accuracy A/Wa: {:.3f}/{:.3f} | FPS {:.3f}".format(
                    current_epoch,
                    memory_avg[current_epoch]["Loss"]/(i+1),
                    memory_avg[current_epoch]["Accuracy"]["Average"]/(i+1),
                    memory_avg[current_epoch]["Accuracy"]["Weighted Average"]/(i+1),
                    memory_avg[current_epoch]["FPS"]/(i+1))
                )
                pbar.update()

    Utilities.write_confusion_matrix_epoch(
        summary_writer=summary_writer,
        encoder=encoder,
        dataset_map=dataset_map,
        epoch=current_epoch,
        memory_average=memory_avg,
        is_train=False,
    )
    confusion_matrix_sum_epoch["test"] = memory_avg[current_epoch]["Confusion Matrix"].tolist()
    with open(os.path.join(summary_writer.get_logdir(), f"confusion_matrix_epoch_{current_epoch}.json"), "w") as f:
        json.dump(confusion_matrix_sum_epoch, f)


class HARMV_CNNLSTM_ResNet18_Single(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int) -> None:
        super(HARMV_CNNLSTM_ResNet18_Single, self).__init__()
        self.__lstm_hidden: int = hidden_cell
        self.__timesteps: int = timesteps
        self.feature_extractor: torchvision.models.ResNet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.feature_extractor.conv1 = torch.nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor.fc = torch.nn.Identity()  # type: ignore
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = self.feature_extractor.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.eval()
        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(512,  self.__lstm_hidden, bidirectional=True, batch_first=True)
        self.fc1: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2,  self.__lstm_hidden*2)
        self.output: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, category_count)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, timesteps, C, H, W = x.shape
        assert self.__timesteps == timesteps, f"Timesteps not equal to {self.__timesteps}"
        with torch.no_grad():
            x = x.view(batch_size * self.__timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, self.__timesteps, -1)

        self.lstm1.flatten_parameters()
        _, (x, _) = self.lstm1(x)

        x = torch.cat((x[0], x[1]), dim=1)

        x = torch.nn.functional.relu(self.fc1(x))
        x = self.output(x)

        return x

    @staticmethod
    def comment() -> str:
        return """CNN Model Shape with single layer of lstm, 2 layer of fc for classification
        ResNet18 is used for cnn feature extraction"""

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args, **kwargs
    ) -> None:
        return default_train_one_epoch(
            *args, **kwargs
        )


class HARMV_CNNLSTM_MobileNetv3_Single(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int) -> None:
        super(HARMV_CNNLSTM_MobileNetv3_Single, self).__init__()
        self.__lstm_hidden: int = hidden_cell
        self.__timesteps: int = timesteps
        self.feature_extractor: torchvision.models.MobileNetV3 = torchvision.models.mobilenet_v3_small(torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
        self.feature_extractor.features[0][0] = torch.nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.feature_extractor.classifier = torch.nn.Identity()  # type: ignore
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = self.feature_extractor.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.eval()
        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(576,  self.__lstm_hidden, bidirectional=True, batch_first=True)
        self.fc1: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2,  self.__lstm_hidden*2)
        self.output: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, category_count)

    def forward(self, x: Tensor):
        batch_size, timesteps, C, H, W = x.shape
        assert self.__timesteps == timesteps, f"Timesteps not equal to {self.__timesteps}"
        with torch.no_grad():
            x = x.view(batch_size * self.__timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, self.__timesteps, -1)

        self.lstm1.flatten_parameters()
        _, (x, _) = self.lstm1(x)

        x = torch.cat((x[0], x[1]), dim=1)

        x = torch.nn.functional.relu(self.fc1(x))
        x = self.output(x)

        return x

    @staticmethod
    def comment() -> str:
        return """CNN Model Shape with single layer of lstm, 2 layer of fc for classification
        MobileNetV3 is used for cnn feature extraction
        """

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args, **kwargs
    ) -> None:
        return default_train_one_epoch(
            *args, **kwargs
        )


class HARMV_CNNLSTM_ShuffleNetv2x1_Single(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int) -> None:
        super(HARMV_CNNLSTM_ShuffleNetv2x1_Single, self).__init__()
        self.__lstm_hidden: int = hidden_cell
        self.__timesteps: int = timesteps
        self.feature_extractor: torchvision.models.ShuffleNetV2 = torchvision.models.shufflenet_v2_x1_0(torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT)
        self.feature_extractor.conv1[0] = torch.nn.Conv2d(2, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # shufflenet avgpool using mean, see implementation
        self.feature_extractor.fc = torch.nn.Identity()  # type: ignore
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = self.feature_extractor.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.eval()
        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(1024,  self.__lstm_hidden, bidirectional=True, batch_first=True)
        self.fc1: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2,  self.__lstm_hidden*2)
        self.output: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, category_count)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, timesteps, C, H, W = x.shape
        assert self.__timesteps == timesteps, f"Timesteps not equal to {self.__timesteps}"
        with torch.no_grad():
            x = x.view(batch_size * self.__timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, self.__timesteps, -1)

        self.lstm1.flatten_parameters()
        _, (x, _) = self.lstm1(x)

        x = torch.cat((x[0], x[1]), dim=1)

        x = torch.nn.functional.relu(self.fc1(x))
        x = self.output(x)

        return x

    @staticmethod
    def comment() -> str:
        return """CNN Model Shape with single layer of lstm, 2 layer of fc for classification
        ShuffleNetv2x1 is used for cnn feature extraction
        """

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args, **kwargs
    ) -> None:
        return default_train_one_epoch(
            *args, **kwargs
        )


class HARMV_CNNLSTM_MNASNetx0_5_Single(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int) -> None:
        super(HARMV_CNNLSTM_MNASNetx0_5_Single, self).__init__()
        self.__lstm_hidden: int = hidden_cell
        self.__timesteps: int = timesteps
        self.feature_extractor: torchvision.models.MNASNet = torchvision.models.mnasnet0_5(torchvision.models.MNASNet0_5_Weights.DEFAULT)
        self.feature_extractor.layers[0] = torch.nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # mnasnet avgpool using mean, see implementation
        self.feature_extractor.classifier = torch.nn.Identity()  # type: ignore
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = self.feature_extractor.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.eval()
        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(1280,  self.__lstm_hidden, bidirectional=True, batch_first=True)
        self.fc1: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2,  self.__lstm_hidden*2)
        self.output: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, category_count)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, timesteps, C, H, W = x.shape
        assert self.__timesteps == timesteps, f"Timesteps not equal to {self.__timesteps}"
        with torch.no_grad():
            x = x.view(batch_size * self.__timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, self.__timesteps, -1)

        self.lstm1.flatten_parameters()
        _, (x, _) = self.lstm1(x)

        x = torch.cat((x[0], x[1]), dim=1)

        x = torch.nn.functional.relu(self.fc1(x))
        x = self.output(x)

        return x

    @staticmethod
    def comment() -> str:
        return """CNN Model Shape with single layer of lstm, 2 layer of fc for classification
        MNASNetx0_5 is used for cnn feature extraction
        """

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args, **kwargs
    ) -> None:
        return default_train_one_epoch(
            *args, **kwargs
        )


class HARMV_CNNLSTM_SqueezeNet1_1_Single(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int) -> None:
        super(HARMV_CNNLSTM_SqueezeNet1_1_Single, self).__init__()
        self.__lstm_hidden: int = hidden_cell
        self.__timesteps: int = timesteps
        self.feature_extractor: torchvision.models.SqueezeNet = torchvision.models.squeezenet1_1(torchvision.models.SqueezeNet1_1_Weights.DEFAULT)
        self.feature_extractor.features[0] = torch.nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(2, 2))
        self.feature_extractor.classifier[0] = torch.nn.Identity()  # type: ignore
        self.feature_extractor.classifier[1] = torch.nn.Identity()  # type: ignore
        self.feature_extractor.classifier[2] = torch.nn.Identity()  # type: ignore
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = self.feature_extractor.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.eval()
        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(512,  self.__lstm_hidden, bidirectional=True, batch_first=True)
        self.fc1: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2,  self.__lstm_hidden*2)
        self.output: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, category_count)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, timesteps, C, H, W = x.shape
        assert self.__timesteps == timesteps, f"Timesteps not equal to {self.__timesteps}"
        with torch.no_grad():
            x = x.view(batch_size * self.__timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, self.__timesteps, -1)

        self.lstm1.flatten_parameters()
        _, (x, _) = self.lstm1(x)

        x = torch.cat((x[0], x[1]), dim=1)

        x = torch.nn.functional.relu(self.fc1(x))
        x = self.output(x)

        return x

    @staticmethod
    def comment() -> str:
        return """CNN Model Shape with single layer of lstm, 2 layer of fc for classification
        SqueezeNet1_1 is used for cnn feature extraction
        """

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args, **kwargs
    ) -> None:
        return default_train_one_epoch(
            *args, **kwargs
        )


class HARMV_CNNLSTM_ShuffleNetv2x0_5_Single(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int) -> None:
        super(HARMV_CNNLSTM_ShuffleNetv2x0_5_Single, self).__init__()
        self.__lstm_hidden: int = hidden_cell
        self.__timesteps: int = timesteps
        self.feature_extractor: torchvision.models.ShuffleNetV2 = torchvision.models.shufflenet_v2_x0_5(torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        self.feature_extractor.conv1[0] = torch.nn.Conv2d(2, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # shufflenet avgpool using mean, see implementation
        self.feature_extractor.fc = torch.nn.Identity()  # type: ignore
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = self.feature_extractor.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.eval()
        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(1024,  self.__lstm_hidden, bidirectional=True, batch_first=True)
        self.fc1: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2,  self.__lstm_hidden*2)
        self.output: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, category_count)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, timesteps, C, H, W = x.shape
        assert self.__timesteps == timesteps, f"Timesteps not equal to {self.__timesteps}"
        with torch.no_grad():
            x = x.view(batch_size * self.__timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, self.__timesteps, -1)

        self.lstm1.flatten_parameters()
        _, (x, _) = self.lstm1(x)

        x = torch.cat((x[0], x[1]), dim=1)

        x = torch.nn.functional.relu(self.fc1(x))
        x = self.output(x)

        return x

    @staticmethod
    def comment() -> str:
        return """CNN Model Shape with single layer of lstm, 2 layer of fc for classification
        ShuffleNetv2x0_5 is used for cnn feature extraction
        """

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args, **kwargs
    ) -> None:
        return default_train_one_epoch(
            *args, **kwargs
        )


class HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_With_Cell(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int) -> None:
        super(HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_With_Cell, self).__init__()
        self.__lstm_hidden: int = hidden_cell
        self.__timesteps: int = timesteps
        self.feature_extractor: torchvision.models.ShuffleNetV2 = torchvision.models.shufflenet_v2_x0_5(torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        self.feature_extractor.conv1[0] = torch.nn.Conv2d(2, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # shufflenet avgpool using mean, see implementation
        self.feature_extractor.fc = torch.nn.Identity()  # type: ignore
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = self.feature_extractor.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.eval()
        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(1024,  self.__lstm_hidden, bidirectional=True, batch_first=True)
        self.fc1: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 4,  self.__lstm_hidden * 4)
        self.output: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 4, category_count)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, timesteps, C, H, W = x.shape
        assert self.__timesteps == timesteps, f"Timesteps not equal to {self.__timesteps}"
        with torch.no_grad():
            x = x.view(batch_size * self.__timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, self.__timesteps, -1)

        self.lstm1.flatten_parameters()
        _, (x, y) = self.lstm1(x)

        x = torch.cat((x[0], x[1], y[0], y[1]), dim=1)

        x = torch.nn.functional.relu(self.fc1(x))
        x = self.output(x)

        return x

    @staticmethod
    def comment() -> str:
        return """CNN Model Shape with single layer of lstm, uses both hidden and cell state, and 2 layer of fc for classification
        ShuffleNetv2x0_5 is used for cnn feature extraction
        """

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args, **kwargs
    ) -> None:
        return default_train_one_epoch(
            *args, **kwargs
        )


class HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_Output(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int) -> None:
        super(HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_Output, self).__init__()
        self.__lstm_hidden: int = hidden_cell
        self.__timesteps: int = timesteps
        self.feature_extractor: torchvision.models.ShuffleNetV2 = torchvision.models.shufflenet_v2_x0_5(torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        self.feature_extractor.conv1[0] = torch.nn.Conv2d(2, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # shufflenet avgpool using mean, see implementation
        self.feature_extractor.fc = torch.nn.Identity()  # type: ignore
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = self.feature_extractor.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.eval()
        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(input_size=1024, hidden_size=self.__lstm_hidden, bidirectional=True, batch_first=True)
        self.fc1: torch.nn.Linear = torch.nn.Linear(self.__timesteps * 1024,  self.__lstm_hidden)
        self.output: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden, category_count)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, timesteps, C, H, W = x.shape
        assert self.__timesteps == timesteps, f"Timesteps not equal to {self.__timesteps}"
        with torch.no_grad():
            x = x.view(batch_size * self.__timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, self.__timesteps, -1)

        self.lstm1.flatten_parameters()
        X, (_, _) = self.lstm1(x)

        x = x.view(batch_size, self.__timesteps * 1024)

        x = torch.nn.functional.relu(self.fc1(x))
        x = self.output(x)

        return x

    @staticmethod
    def comment() -> str:
        return """CNN Model Shape with single layer of lstm, 2 layer of fc for classification
        ShuffleNetv2x0_5 is used for cnn feature extraction
        """

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args, **kwargs
    ) -> None:
        return default_train_one_epoch(
            *args, **kwargs
        )


class HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int) -> None:
        super(HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP, self).__init__()
        self.__lstm_hidden: int = hidden_cell
        self.__timesteps: int = timesteps
        self.feature_extractor: torchvision.models.ShuffleNetV2 = torchvision.models.shufflenet_v2_x0_5(torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        self.feature_extractor.conv1[0] = torch.nn.Conv2d(2, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # shufflenet avgpool using mean, see implementation
        self.feature_extractor.fc = torch.nn.Identity()  # type: ignore
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = self.feature_extractor.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.eval()
        self.num_layers = 1
        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(1024,  self.__lstm_hidden, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.conv1: torch.nn.Conv1d = torch.nn.Conv1d(self.__timesteps, 128, kernel_size=5, stride=2)
        self.conv2: torch.nn.Conv1d = torch.nn.Conv1d(128, 256, kernel_size=5, stride=2)
        self.conv3: torch.nn.Conv1d = torch.nn.Conv1d(256, 512, kernel_size=5, stride=2)
        self.conv4: torch.nn.Conv1d = torch.nn.Conv1d(512, 1024, kernel_size=5, stride=2)
        self.batchnorm: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(1024)
        self.output: torch.nn.Linear = torch.nn.Linear(1024, category_count)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, timesteps, C, H, W = x.shape
        assert self.__timesteps == timesteps, f"Timesteps not equal to {self.__timesteps}"
        with torch.no_grad():
            x = x.view(batch_size * self.__timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, self.__timesteps, -1)

        self.lstm1.flatten_parameters()
        x, (_, _) = self.lstm1(x)

        x = torch.nn.functional.relu(x)
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.conv4(x)
        x = torch.nn.functional.relu(x)

        x = torch.mean(x, dim=2)  # GAP

        x = self.batchnorm(x)

        x = self.output(x)

        return x

    @staticmethod
    def comment() -> str:
        return """CNN Model Shape with 1 layer of lstm, last output from lstm as input 
        for CNN layer that has kernel size of 5x5, and uses GAP layer and fc for classification
        ShuffleNetv2x0_5 is used for cnn feature extraction
        """

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args, **kwargs
    ) -> None:
        return default_train_one_epoch(
            *args, **kwargs
        )


class HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP_With_Residual_Network(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int) -> None:
        super(HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP_With_Residual_Network, self).__init__()
        self.__lstm_hidden: int = hidden_cell
        self.__timesteps: int = timesteps
        self.__num_layers: int = 1
        self.__feature_extractor_size: int = 1024  # last layer output from shuffle net

        self.queue: Tensor = torch.empty((0))
        self.reset_queue()

        self.feature_extractor: torchvision.models.ShuffleNetV2 = torchvision.models.shufflenet_v2_x0_5(torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        self.feature_extractor.conv1[0] = torch.nn.Conv2d(2, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # shufflenet avgpool using mean, see implementation
        self.feature_extractor.fc = torch.nn.Identity()  # type: ignore
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = self.feature_extractor.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.eval()

        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(self.__feature_extractor_size,  self.__lstm_hidden, num_layers=self.__num_layers, bidirectional=True, batch_first=True)
        self.conv1: torch.nn.Conv1d = torch.nn.Conv1d(self.__timesteps, 128, kernel_size=5, stride=2, bias=False)
        self.conv2: torch.nn.Conv1d = torch.nn.Conv1d(128, 256, kernel_size=5, stride=2, bias=False)
        self.conv3: torch.nn.Conv1d = torch.nn.Conv1d(256, 512, kernel_size=5, stride=2, bias=False)
        self.conv4: torch.nn.Conv1d = torch.nn.Conv1d(512, 1024, kernel_size=5, stride=2, bias=False)
        self.batchnorm1: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(128)
        self.batchnorm2: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(256)
        self.batchnorm3: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(512)
        self.batchnorm4: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(1024)
        self.fc_lstm: torch.nn.Linear = torch.nn.Linear(self.__timesteps*2, 1024)
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(0.5, inplace=True)
        self.relu: torch.nn.ReLU = torch.nn.ReLU()
        self.output: torch.nn.Linear = torch.nn.Linear(category_count)

    def forward(self, x: Tensor,) -> Tensor:
        x = self.feature_extract(x)
        x = self.forward_classifier(x)
        return x

    def forward_with_queue(self, x: Tensor) -> Tensor:
        assert not self.training, "Not compatible with training mode"
        assert x.shape[0] == 1, "Only single batch compatible"
        x = self.feature_extract(x)
        x = self.push_tensor_to_queue(x)
        x = self.forward_classifier(x)
        return x

    def forward_classifier(self, x: Tensor) -> Tensor:
        self.lstm1.flatten_parameters()

        x, (h, _) = self.lstm1(x)

        x_lstm = torch.concat((h[-2], h[-1]), dim=1)  # concat backward and forward hidden cell
        x_lstm = self.fc_lstm(x_lstm)

        x = self.relu(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)

        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)

        x = self.relu(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)

        x = self.relu(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)

        x = torch.mean(x, dim=2)  # GAP
        x += x_lstm

        x = self.relu(x)

        x = self.output(x)
        return x

    def feature_extract(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            assert len(x.shape) == 5, "Requires batched input"
            batch_size, timesteps, C, H, W = x.shape
            x = x.view(batch_size * timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, timesteps, self.__feature_extractor_size)
            return x

    def push_tensor_to_queue(self, x: Tensor) -> Tensor:
        assert not self.training, "Not compatible with training mode"
        assert x.shape[0] == 1, "Only single batch compatible"
        with torch.no_grad():
            assert len(x.shape) == 3 and x.shape[-1] == self.__feature_extractor_size, "Requires input from feature extractor"
            self.queue = torch.cat((self.queue.cuda(), x), dim=1)
            self.queue = self.queue[:, -self.__timesteps:, ...]
            return torch.clone(self.queue)

    def reset_queue(self) -> Tensor:
        with torch.no_grad():
            self.queue = torch.ones((1, self.__timesteps, self.__feature_extractor_size)) * 0.5
            return torch.clone(self.queue)

    @staticmethod
    def comment() -> str:
        return """Final model"""

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args, **kwargs
    ) -> None:
        return default_train_one_epoch(
            *args, **kwargs
        )


class HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP_With_Residual_Network_Minified(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int) -> None:
        super(HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP_With_Residual_Network_Minified, self).__init__()
        self.__lstm_hidden: int = hidden_cell
        self.__timesteps: int = timesteps
        self.__num_layers: int = 1
        self.__channel_output: int = 128
        self.__feature_extractor_size: int = 1024  # last layer output from shuffle net

        self.queue: Tensor = torch.empty((0))
        self.reset_queue()

        self.feature_extractor: torchvision.models.ShuffleNetV2 = torchvision.models.shufflenet_v2_x0_5(torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        self.feature_extractor.conv1[0] = torch.nn.Conv2d(2, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # shufflenet avgpool using mean, see implementation
        self.feature_extractor.fc = torch.nn.Identity()  # type: ignore
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = self.feature_extractor.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.eval()

        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(self.__feature_extractor_size,  self.__lstm_hidden, num_layers=self.__num_layers, bidirectional=True, batch_first=True)
        self.conv1: torch.nn.Conv1d = torch.nn.Conv1d(self.__timesteps, 128, kernel_size=3, stride=2, bias=False)
        self.conv2: torch.nn.Conv1d = torch.nn.Conv1d(128, 128, kernel_size=1, stride=1, bias=False)
        self.batchnorm1: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(128)
        self.batchnorm2: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(128)
        self.fclstm: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden*2, self.__channel_output)
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(0.5, inplace=True)
        self.silu: torch.nn.SiLU = torch.nn.SiLU()
        self.output: torch.nn.Linear = torch.nn.Linear(self.__channel_output, category_count)

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extract(x)
        x = self.forward_classifier(x)
        return x

    def forward_with_queue(self, x: Tensor) -> Tensor:
        assert not self.training, "Not compatible with training mode"
        assert x.shape[0] == 1, "Only single batch compatible"
        x = self.feature_extract(x)
        x = self.push_tensor_to_queue(x)
        x = self.forward_classifier(x)
        return x

    def forward_classifier(self, x: Tensor) -> Tensor:
        self.lstm1.flatten_parameters()
        x, (h, _) = self.lstm1(x)

        x = self.silu(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)

        x = self.silu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)

        x = torch.mean(x, dim=2)  # GAP

        x_lstm = torch.concat((h[-2], h[-1]), dim=1)  # concat backward and forward hidden cell
        x_lstm = self.fclstm(x_lstm)

        x += x_lstm

        x = self.silu(x)

        x = self.dropout(x)
        x = self.output(x)
        return x

    def feature_extract(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            assert len(x.shape) == 5, "Requires batched input"
            batch_size, timesteps, C, H, W = x.shape
            x = x.view(batch_size * timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, timesteps, self.__feature_extractor_size)
            return x

    def push_tensor_to_queue(self, x: Tensor) -> Tensor:
        assert not self.training, "Not compatible with training mode"
        assert x.shape[0] == 1, "Only single batch compatible"
        with torch.no_grad():
            assert len(x.shape) == 3 and x.shape[-1] == self.__feature_extractor_size, "Requires input from feature extractor"
            self.queue = torch.cat((self.queue.cuda(), x), dim=1)
            self.queue = self.queue[:, -self.__timesteps:, ...]
            return torch.clone(self.queue)

    def reset_queue(self) -> Tensor:
        with torch.no_grad():
            self.queue = torch.ones((1, self.__timesteps, self.__feature_extractor_size)) * 0.5
            return torch.clone(self.queue)

    @staticmethod
    def comment() -> str:
        return """Final model"""

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args, **kwargs
    ) -> None:
        return default_train_one_epoch(
            *args, **kwargs
        )


class HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP_With_Residual_Network_Minified_BeforeLSTM(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int, channel_size: int = 128) -> None:
        super(HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP_With_Residual_Network_Minified_BeforeLSTM, self).__init__()
        self.__lstm_hidden: int = hidden_cell
        self.__timesteps: int = timesteps
        self.__num_layers: int = 1
        self.__feature_extractor_size: int = 1024  # last layer output from shuffle net
        self.__output_channel_size: int = channel_size  # fc size

        self.queue: Tensor = torch.empty((0))
        self.reset_queue()

        self.feature_extractor: torchvision.models.ShuffleNetV2 = torchvision.models.shufflenet_v2_x0_5(torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        self.feature_extractor.conv1[0] = torch.nn.Conv2d(2, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # shufflenet avgpool using mean, see implementation
        self.feature_extractor.fc = torch.nn.Identity()  # type: ignore
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = self.feature_extractor.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.eval()

        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(self.__feature_extractor_size,  self.__lstm_hidden, num_layers=self.__num_layers, bidirectional=True, batch_first=True)

        self.conv_sequence_1: torch.nn.Conv1d = torch.nn.Conv1d(self.__timesteps, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_sequence_2: torch.nn.Conv1d = torch.nn.Conv1d(128, self.__output_channel_size, kernel_size=1, stride=1, bias=False)
        self.batchnorm_sequence_1: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(128)
        self.batchnorm_sequence_2: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(self.__output_channel_size)

        self.fc_final_state: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, self.__output_channel_size)

        self.dropout: torch.nn.Dropout = torch.nn.Dropout(0.5, inplace=True)
        self.silu: torch.nn.SiLU = torch.nn.SiLU()
        self.output: torch.nn.Linear = torch.nn.Linear(self.__output_channel_size, category_count)

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extract(x)
        x = self.forward_classifier(x)
        return x

    def forward_with_queue(self, x: Tensor) -> Tensor:
        assert not self.training, "Not compatible with training mode"
        assert x.shape[0] == 1, "Only single batch compatible"
        x = self.feature_extract(x)
        x = self.push_tensor_to_queue(x)
        x = self.forward_classifier(x)
        return x

    def forward_classifier(self, x: Tensor) -> Tensor:
        # cnn
        x_cnn = self.silu(x)
        x_cnn = self.conv_sequence_1(x_cnn)
        x_cnn = self.batchnorm_sequence_1(x_cnn)

        x_cnn = self.silu(x_cnn)
        x_cnn = self.conv_sequence_2(x_cnn)
        x_cnn = self.batchnorm_sequence_2(x_cnn)

        x_cnn = torch.mean(x_cnn, dim=2)  # global avg pooling

        # final state
        self.lstm1.flatten_parameters()
        _, (h, _) = self.lstm1(x)
        x_lstm = torch.concat((h[-2], h[-1]), dim=1)  # concat backward and forward hidden cell
        x_lstm = self.fc_final_state(x_lstm)

        # additive residual
        x = x_cnn + x_lstm
        x = self.silu(x)

        # fully connected
        x = self.dropout(x)
        x = self.output(x)
        return x

    def feature_extract(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            assert len(x.shape) == 5, "Requires batched input"
            batch_size, timesteps, C, H, W = x.shape
            x = x.view(batch_size * timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, timesteps, self.__feature_extractor_size)
            return x

    def push_tensor_to_queue(self, x: Tensor) -> Tensor:
        assert not self.training, "Not compatible with training mode"
        assert x.shape[0] == 1, "Only single batch compatible"
        with torch.no_grad():
            assert len(x.shape) == 3 and x.shape[-1] == self.__feature_extractor_size, "Requires input from feature extractor"
            self.queue = torch.cat((self.queue.cuda(), x), dim=1)
            self.queue = self.queue[:, -self.__timesteps:, ...]
            return torch.clone(self.queue)

    def reset_queue(self) -> Tensor:
        with torch.no_grad():
            self.queue = torch.ones((1, self.__timesteps, self.__feature_extractor_size)) * 0.5
            return torch.clone(self.queue)

    @staticmethod
    def comment() -> str:
        return """Final model"""

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args, **kwargs
    ) -> None:
        return default_train_one_epoch(
            *args, **kwargs
        )

class HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP_With_Residual_Network_Minified_BeforeLSTM_CNNReLU(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int, channel_size: int = 128) -> None:
        super(HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP_With_Residual_Network_Minified_BeforeLSTM_CNNReLU, self).__init__()
        self.__lstm_hidden: int = hidden_cell
        self.__timesteps: int = timesteps
        self.__num_layers: int = 1
        self.__feature_extractor_size: int = 1024  # last layer output from shuffle net
        self.__output_channel_size: int = channel_size  # fc size

        self.queue: Tensor = torch.empty((0))
        self.reset_queue()

        self.feature_extractor: torchvision.models.ShuffleNetV2 = torchvision.models.shufflenet_v2_x0_5(torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT)
        self.feature_extractor.conv1[0] = torch.nn.Conv2d(2, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        # shufflenet avgpool using mean, see implementation
        self.feature_extractor.fc = torch.nn.Identity()  # type: ignore
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = self.feature_extractor.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.eval()

        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(self.__feature_extractor_size,  self.__lstm_hidden, num_layers=self.__num_layers, bidirectional=True, batch_first=True)

        self.conv_sequence_1: torch.nn.Conv1d = torch.nn.Conv1d(self.__timesteps, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_sequence_2: torch.nn.Conv1d = torch.nn.Conv1d(128, self.__output_channel_size, kernel_size=1, stride=1, bias=False)
        self.batchnorm_sequence_1: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(128)
        self.batchnorm_sequence_2: torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(self.__output_channel_size)

        self.fc_final_state: torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, self.__output_channel_size)

        self.dropout: torch.nn.Dropout = torch.nn.Dropout(0.5, inplace=True)
        self.relu: torch.nn.ReLU = torch.nn.ReLU()
        self.silu: torch.nn.SiLU = torch.nn.SiLU()
        self.output: torch.nn.Linear = torch.nn.Linear(self.__output_channel_size, category_count)

    def forward(self, x: Tensor) -> Tensor:
        x = self.feature_extract(x)
        x = self.forward_classifier(x)
        return x

    def forward_with_queue(self, x: Tensor) -> Tensor:
        assert not self.training, "Not compatible with training mode"
        assert x.shape[0] == 1, "Only single batch compatible"
        x = self.feature_extract(x)
        x = self.push_tensor_to_queue(x)
        x = self.forward_classifier(x)
        return x

    def forward_classifier(self, x: Tensor) -> Tensor:
        # cnn
        x_cnn = self.relu(x)
        x_cnn = self.conv_sequence_1(x_cnn)
        x_cnn = self.batchnorm_sequence_1(x_cnn)

        x_cnn = self.relu(x_cnn)
        x_cnn = self.conv_sequence_2(x_cnn)
        x_cnn = self.batchnorm_sequence_2(x_cnn)

        x_cnn = torch.mean(x_cnn, dim=2)  # global avg pooling

        # final state
        self.lstm1.flatten_parameters()
        _, (h, _) = self.lstm1(x)
        x_lstm = torch.concat((h[-2], h[-1]), dim=1)  # concat backward and forward hidden cell
        x_lstm = self.fc_final_state(x_lstm)

        # additive residual
        x = x_cnn + x_lstm
        x = self.silu(x)

        # fully connected
        x = self.dropout(x)
        x = self.output(x)
        return x

    def feature_extract(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            assert len(x.shape) == 5, "Requires batched input"
            batch_size, timesteps, C, H, W = x.shape
            assert self.__timesteps == timesteps, f"Timesteps not equal to {self.__timesteps}"
            x = x.view(batch_size * self.__timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, self.__timesteps, self.__feature_extractor_size)
            return x

    def push_tensor_to_queue(self, x: Tensor) -> Tensor:
        assert not self.training, "Not compatible with training mode"
        assert x.shape[0] == 1, "Only single batch compatible"
        with torch.no_grad():
            assert len(x.shape) == 3 and x.shape[-1] == self.__feature_extractor_size, "Requires input from feature extractor"
            _, current_to_push, _ = x.shape
            if current_to_push >= self.__timesteps:
                self.queue = torch.clone(x[:, -self.__timesteps:])
            else:
                to_take: int = self.__timesteps - current_to_push
                self.queue = torch.cat((self.queue[:, -to_take:], x))
            return torch.clone(self.queue)

    def reset_queue(self) -> Tensor:
        with torch.no_grad():
            self.queue = torch.ones((1, self.__timesteps, self.__feature_extractor_size)) * 0.5
            return torch.clone(self.queue)

    @staticmethod
    def comment() -> str:
        return """Final model"""

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args, **kwargs
    ) -> None:
        return default_train_one_epoch(
            *args, **kwargs
        )

register_model = [
    HARMV_CNNLSTM_ResNet18_Single,
    HARMV_CNNLSTM_MobileNetv3_Single,
    HARMV_CNNLSTM_ShuffleNetv2x1_Single,
    HARMV_CNNLSTM_MNASNetx0_5_Single,
    HARMV_CNNLSTM_SqueezeNet1_1_Single,
    HARMV_CNNLSTM_ShuffleNetv2x0_5_Single,
    HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_Output,
    HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_With_Cell,
    HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP,
    HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP_With_Residual_Network,
    HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP_With_Residual_Network_Minified,
    HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP_With_Residual_Network_Minified_BeforeLSTM,
    HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_CNNGAP_With_Residual_Network_Minified_BeforeLSTM_CNNReLU
]
