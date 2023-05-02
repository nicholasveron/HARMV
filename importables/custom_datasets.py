import h5py
import numpy
import torch
import pandas
import sklearn
import torch.utils.data
import sklearn.preprocessing
from tqdm.auto import tqdm
from .utilities import Utilities
from .preprocessing_managers import PreprocessingManagers, DatasetDictionary
from .custom_types import (
    Any,
    List,
    Tuple,
    Union,
    Tensor,
    ndarray,
    ColorXY,
    OpticalFlowFrame,
    SegmentationMask,
    BoundingBoxXY1XY2,
    MotionVectorFrame,

)


class FlowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_mapping: dict,
        label_encoder: sklearn.preprocessing.LabelEncoder,
        loader: PreprocessingManagers.Loader,
        dataframe: pandas.DataFrame,
        h5py_base_file: h5py.File,
        timestep: int,
        transform: Union[callable, None] = None,  # type: ignore
        label_key: str = "A",
        with_pre_padding: bool = False
    ):

        self.__timestep: int = timestep
        self.__label_key: str = label_key
        self.__h5py_base_file: h5py.File = h5py_base_file
        self.__dataset_mapping: dict = dataset_mapping
        self.__index_mapping: List[Tuple[str, int, int, int]] = []
        self.__loader: PreprocessingManagers.Loader = loader
        self.__transform:  Union[callable, None] = transform  # type: ignore
        self.__label_encoder: sklearn.preprocessing.LabelEncoder = label_encoder
        self.__with_pre_padding: bool = with_pre_padding

        for index in tqdm(range(len(dataframe)), desc="Generating index mapping..."):

            current_series: pandas.Series = dataframe.iloc[index]
            label: int = current_series[self.__dataset_mapping[self.__label_key]]  # type: ignore
            frame_count: int = current_series[self.__dataset_mapping[DatasetDictionary.FRAME_COUNT_KEY]]  # type: ignore
            file_path: str = current_series[self.__dataset_mapping[DatasetDictionary.GENERATED_FILENAME_KEY]]  # type: ignore

            for frame_index in range(frame_count):
                start_index: int = frame_index - self.__timestep + 1
                stop_index: int = frame_index + 1

                if not self.__with_pre_padding:
                    if start_index < 0:
                        continue

                self.__index_mapping.append(
                    (
                        file_path,
                        label,
                        start_index,
                        stop_index,
                    )
                )

    def __len__(self) -> int:
        return len(self.__index_mapping)

    def __getitem__(self, idx: Any) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path, label, start_index, stop_index = self.__index_mapping[idx]

        file_group: h5py.Group = self.__h5py_base_file[file_path]  # type: ignore
        _, mask_generator, motion_vector_processor, optical_flow_generator = self.__loader.load_from_h5(file_group, start_index, stop_index)

        label_label: ndarray = self.__label_encoder.transform([[label]])[0]  # type: ignore
        returns_list: List[dict[str, Union[ndarray, Tensor]]] = []

        motion_vector_output: bool = motion_vector_processor is not None
        optical_flow_output: bool = optical_flow_generator is not None

        if self.__with_pre_padding:
            if len(mask_generator) < self.__timestep:
                padding_returns: dict[str, Union[ndarray, Tensor]] = {}
                padding_returns["frame_label"] = numpy.zeros_like(label_label)

                if motion_vector_output:
                    padding_returns["motion_vector"] = motion_vector_processor.generate_blank()

                if optical_flow_output:
                    padding_returns["optical_flow"] = optical_flow_generator.generate_blank()

                if self.__transform:
                    padding_returns = self.__transform(padding_returns)

                returns_list = [padding_returns] * int(self.__timestep-len(mask_generator))

        while len(mask_generator) > 0:
            current_frame_returns: dict[str, Union[ndarray, Tensor]] = {}
            current_frame_returns["frame_label"] = numpy.zeros_like(label_label)

            is_mask_available, segmentation_mask, bounding_box = mask_generator.forward_once_with_mcbb()
            if is_mask_available:
                current_frame_returns["frame_label"] = label_label
                current_frame_returns["segmentation_mask"] = segmentation_mask
                current_frame_returns["bounding_box"] = bounding_box

            if motion_vector_output:
                motion_vector_frame: MotionVectorFrame = motion_vector_processor.process()
                current_frame_returns["motion_vector"] = motion_vector_frame

            if optical_flow_output:
                optical_flow_frame: OpticalFlowFrame = optical_flow_generator.forward_once_auto()
                current_frame_returns["optical_flow"] = optical_flow_frame

            if self.__transform:
                current_frame_returns = self.__transform(current_frame_returns)

            if "segmentation_mask" in current_frame_returns:
                del current_frame_returns["segmentation_mask"]
                del current_frame_returns["bounding_box"]

            returns_list.append(current_frame_returns)

        collated_timestep: dict[str, Tensor] = torch.utils.data.default_collate(returns_list)
        collated_timestep["label"] = torch.tensor(label_label)

        return collated_timestep

    class MaskCropTransform(object):
        def __init__(self, mask: bool = True, crop: bool = True, replace_with: ColorXY = (128, 128)) -> None:
            self.__mask: bool = mask
            self.__crop: bool = crop
            self.__replace_with: ColorXY = replace_with

        def __call__(self, X: dict) -> dict:
            if "segmentation_mask" not in X:
                return X

            segmentation_mask: SegmentationMask = X["segmentation_mask"]
            bounding_box: BoundingBoxXY1XY2 = X["bounding_box"]

            if self.__mask:
                if "motion_vector" in X:
                    X["motion_vector"][~segmentation_mask] = self.__replace_with

                if "optical_flow" in X:
                    X["optical_flow"][~segmentation_mask] = self.__replace_with

            if self.__crop:
                if "motion_vector" in X:
                    X["motion_vector"] = Utilities.crop_to_bb(X["motion_vector"], bounding_box)

                if "optical_flow" in X:
                    X["optical_flow"] = Utilities.crop_to_bb(X["optical_flow"], bounding_box)

            del X["segmentation_mask"]
            del X["bounding_box"]

            return X

    class PadResize(object):
        def __init__(self, output_size: int, pad_with: ColorXY = (128, 128)) -> None:
            self.__output_size: int = output_size
            self.__pad_with: ColorXY = pad_with

        def __call__(self, X: dict) -> dict:

            if "motion_vector" in X:
                X["motion_vector"], _, _, _, _, _ = Utilities.letterbox(X["motion_vector"], self.__output_size, self.__pad_with, 0, True)

            if "optical_flow" in X:
                X["optical_flow"], _, _, _, _, _ = Utilities.letterbox(X["optical_flow"], self.__output_size, self.__pad_with, 0, True)

            return X

    class ToCHW(object):

        def __call__(self, X: dict) -> dict:

            if "motion_vector" in X:
                X["motion_vector"] = X["motion_vector"].transpose((2, 0, 1))

            if "optical_flow" in X:
                X["optical_flow"] = X["optical_flow"].transpose((2, 0, 1))

            return X

    class Rescale(object):

        def __init__(self, scale: float = 1.0) -> None:
            self.__scale: float = scale

        def __call__(self, X: dict) -> dict:

            if "motion_vector" in X:
                X["motion_vector"] = (X["motion_vector"] * self.__scale).astype(numpy.float32)

            if "optical_flow" in X:
                X["optical_flow"] = (X["optical_flow"] * self.__scale).astype(numpy.float32)

            return X
