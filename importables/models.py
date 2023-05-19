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
            "train" : [],
            "test" : []
        }

        memory_avg: dict = {}
        with tqdm(total=train_length) as pbar:

            for i, data in enumerate(train_dl):
                start_time: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
                end_time: torch.cuda.Event = torch.cuda.Event(enable_timing=True)

                start_time.record() # type: ignore

                inputs: Tensor = data[training_parameters["data_selector"]].to("cuda")
                labels: Tensor = data["label"].to("cuda")

                optimizer.zero_grad()
                output: Tensor = model(inputs)
                loss: Tensor = criterion(output, labels)
                loss.backward()
                optimizer.step()

                end_time.record() # type: ignore

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

                    start_time.record() # type: ignore

                    inputs: Tensor = data[training_parameters["data_selector"]].to("cuda")
                    labels: Tensor = data["label"].to("cuda")

                    output: Tensor = model(inputs)
                    loss: Tensor = criterion_eval(output, labels)

                    end_time.record() # type: ignore

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
        with open(os.path.join(summary_writer.get_logdir(),f"confusion_matrix_epoch_{current_epoch}.json"), "w") as f:
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
        self.fc1:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2,  self.__lstm_hidden*2)
        self.output:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, category_count)
        
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
        *args,**kwargs
    ) -> None:
        return default_train_one_epoch(
            *args,**kwargs
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
        self.fc1:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2,  self.__lstm_hidden*2)
        self.output:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, category_count)

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
        *args,**kwargs
    ) -> None:
        return default_train_one_epoch(
            *args,**kwargs
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
        self.fc1:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2,  self.__lstm_hidden*2)
        self.output:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, category_count)

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
        *args,**kwargs
    ) -> None:
        return default_train_one_epoch(
            *args,**kwargs
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
        self.fc1:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2,  self.__lstm_hidden*2)
        self.output:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, category_count)

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
        *args,**kwargs
    ) -> None:
        return default_train_one_epoch(
            *args,**kwargs
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
        self.fc1:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2,  self.__lstm_hidden*2)
        self.output:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, category_count)

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
        *args,**kwargs
    ) -> None:
        return default_train_one_epoch(
            *args,**kwargs
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
        self.fc1:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2,  self.__lstm_hidden*2)
        self.output:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, category_count)

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
        *args,**kwargs
    ) -> None:
        return default_train_one_epoch(
            *args,**kwargs
        )

class HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_Rev(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int) -> None:
        super(HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_Rev, self).__init__()
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
        self.lstm1: torch.nn.LSTM = torch.nn.LSTM(input_size= 1024, hidden_size=self.__lstm_hidden, bidirectional=True, batch_first=True)
        self.fc1:torch.nn.Linear = torch.nn.Linear(self.__timesteps * 1024,  self.__lstm_hidden)
        self.output:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden, category_count)

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
        *args,**kwargs
    ) -> None:
        return default_train_one_epoch(
            *args,**kwargs
        )

class HARMV_CNNLSTM_ShuffleNetv2x0_5_DoubleLSTM(torch.nn.Module):
    def __init__(self, category_count: int, timesteps: int, hidden_cell: int) -> None:
        super(HARMV_CNNLSTM_ShuffleNetv2x0_5_DoubleLSTM, self).__init__()
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
        self.lstm2: torch.nn.LSTM = torch.nn.LSTM(self.__lstm_hidden * 2,  self.__lstm_hidden, bidirectional=True, batch_first=True)
        self.fc1:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2,  self.__lstm_hidden*2, bidirectional=True, batch_first=True)
        self.output:torch.nn.Linear = torch.nn.Linear(self.__lstm_hidden * 2, category_count)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, timesteps, C, H, W = x.shape
        assert self.__timesteps == timesteps, f"Timesteps not equal to {self.__timesteps}"
        with torch.no_grad():
            x = x.view(batch_size * self.__timesteps, C, H, W)
            x = self.feature_extractor(x)
            x = x.view(batch_size, self.__timesteps, -1)

        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        x, (h, c) = self.lstm1(x)
        x, (_, _) = self.lstm2(x, (h, c))

        x = torch.nn.functional.relu(self.fc1(x))
        x = self.output(x)

        return x

    @staticmethod
    def comment() -> str:
        return """CNN Model Shape with double layer of lstm, 2 layer of fc for classification
        ShuffleNetv2x0_5 is used for cnn feature extraction
        """

    @staticmethod
    def resolution() -> int:
        return 224

    @staticmethod
    def train_one_epoch(
        *args,**kwargs
    ) -> None:
        return default_train_one_epoch(
            *args,**kwargs
        )

register_model = [
    HARMV_CNNLSTM_ResNet18_Single,
    HARMV_CNNLSTM_MobileNetv3_Single,
    HARMV_CNNLSTM_ShuffleNetv2x1_Single,
    HARMV_CNNLSTM_MNASNetx0_5_Single,
    HARMV_CNNLSTM_SqueezeNet1_1_Single,
    HARMV_CNNLSTM_ShuffleNetv2x0_5_Single,
    HARMV_CNNLSTM_ShuffleNetv2x0_5_Single_Rev,
]
