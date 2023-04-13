"""Module for preprocessing managers such as generator, viewer, loader, etc"""
import os
import h5py
import numpy
import pandas
from typing import Union
from copy import deepcopy
from tqdm.auto import tqdm
from numpy import ndarray
from .video_decoder import VideoDecoderProcessSpawner
from .mask_generator import MaskGenerator, MaskGeneratorMocker, MaskWithMostCenterBoundingBoxData
from .optical_flow_generator import OpticalFlowGenerator, OpticalFlowGeneratorMocker, OpticalFlowFrame
from .motion_vector_processor import MotionVectorProcessor, MotionVectorProcessorMocker, MotionVectorFrame
from .custom_types import (
    DecodedData,
    MotionVectorFrame,
    RawMotionVectors,
    FrameRGB
)


class DatasetDictionary:
    """Pandas wrapper that manages dataset dictionary"""

    class Mappings:
        NTU_ACTION_RECOGNITION_DATASET = {
            "A": "ActionID",
            "C": "CameraID",
            "P": "PerformerID",
            "R": "ReplicationID",
            "S": "SetupID",
            "/": "SourceFilePath",
            ".": "GeneratedFileName"
        }

        def __new__(cls):
            raise TypeError('DatasetDictionary.Mappings is a constant class and cannot be instantiated')

    UNKNOWN_KEY = "_"
    GENERATED_FILENAME_KEY = "."
    SOURCE_FILEPATH_KEY = "/"

    def __init__(self, dictionary_path: str, dataset_mapping: dict) -> None:
        self.__dictionary_path: str = dictionary_path
        self.__dataset_mapping: dict = dataset_mapping
        sample_dataset: dict = {}
        for keys in dataset_mapping.values():
            sample_dataset[keys] = [1]
        sample_dataset[dataset_mapping[self.GENERATED_FILENAME_KEY]] = ["a"]
        sample_dataset[dataset_mapping[self.SOURCE_FILEPATH_KEY]] = ["a"]

        try:
            self.__dictionary: pandas.DataFrame = pandas.read_csv(dictionary_path, index_col=False).convert_dtypes()
            assert set(self.__dictionary.columns) - set(dataset_mapping.values()) == set(), "Dataset mapping is not matching with the loaded dictionary, replacing..."
        except:
            self.__dictionary: pandas.DataFrame = pandas.DataFrame(sample_dataset).convert_dtypes()
            self.clear()
            self.save()

    def save(self) -> bool:
        try:
            self.__dictionary.to_csv(self.__dictionary_path, index=False)
        except:
            return False
        return True

    def parse(self, dataset_filename: str, source_filepath: str) -> dict:
        filtered: dict = {}
        current_parsing: str = self.UNKNOWN_KEY
        current_column: str = self.UNKNOWN_KEY
        dataset_mapping_vals: set = set(self.__dataset_mapping.keys())

        for ch in dataset_filename:
            if not ch.isalnum():
                if current_column != "_":
                    filtered[current_column] = int(filtered[current_column])
                break

            if not ch.isdigit():
                current_parsing = self.UNKNOWN_KEY

                if current_column != self.UNKNOWN_KEY:
                    filtered[current_column] = int(filtered[current_column])

                if ch in dataset_mapping_vals:
                    current_parsing = ch
                    current_column = self.__dataset_mapping[current_parsing]

            if current_parsing != self.UNKNOWN_KEY:
                if current_column not in filtered.keys():
                    filtered[current_column] = ""
                else:
                    filtered[current_column] += ch

        filtered[self.__dataset_mapping[self.SOURCE_FILEPATH_KEY]] = source_filepath
        filtered[self.__dataset_mapping[self.GENERATED_FILENAME_KEY]] = dataset_filename

        return filtered

    def append(self, dataset_dict: dict) -> bool:
        find_criterion: pandas.Series = self.find_row(dataset_dict)
        if any(find_criterion):
            self.__dictionary = self.__dictionary[~find_criterion]
        try:
            self.__dictionary = self.__dictionary.append(dataset_dict, ignore_index=True)  # type: ignore
        except:
            return False
        return True

    def parse_and_append(self, dataset_filename: str, source_filepath: str) -> bool:
        dataset_dict: dict = self.parse(dataset_filename, source_filepath)
        return self.append(
            dataset_dict
        )

    def find_row(self, dataset_dict: dict, ignore_source=True) -> pandas.Series:

        # all attributes (except source path) should be defined by its generated filename
        # check generated filename only, check source filepath if requested
        generated_filename_key: str = self.__dataset_mapping[self.GENERATED_FILENAME_KEY]
        check_criterion: pandas.Series = self.__dictionary[generated_filename_key] == dataset_dict[generated_filename_key]

        if not ignore_source:
            source_filepath_key: str = self.__dataset_mapping[self.SOURCE_FILEPATH_KEY]
            check_criterion &= self.__dictionary[source_filepath_key] == dataset_dict[source_filepath_key]

        return check_criterion

    def as_DataFrame(self) -> pandas.DataFrame:
        return self.__dictionary

    def clear(self) -> bool:
        try:
            self.__dictionary = self.__dictionary[0:0]
        except:
            return False
        return self.save()

    def __del__(self):
        try:
            self.save()
        except:
            pass


class PreprocessingManagers:
    """Preprocessing managers parent class"""

    def __new__(cls):
        raise TypeError('PreprocessingManagers is a static base class and cannot be instantiated')

    @staticmethod
    def generate_dataset_dictionary_recursively(current_path: str, dataset_mapping: dict, current_dictionary: Union[DatasetDictionary, None] = None):
        current_path = os.path.abspath(current_path)
        if current_dictionary == None:
            assert os.path.isdir(current_path), "Path is not exist or a directory"
            _, original_filename = os.path.split(current_path)
            target_name = original_filename.split(".")[0] + ".csv"
            return PreprocessingManagers.generate_dataset_dictionary_recursively(
                current_path,
                dataset_mapping,
                DatasetDictionary(
                    os.path.abspath(
                        os.path.join(current_path, target_name)
                    ),
                    dataset_mapping
                )
            )

        if os.path.isdir(current_path):
            for path in tqdm(os.listdir(current_path)):
                path = os.path.join(current_path, path)
                current_dictionary = PreprocessingManagers.generate_dataset_dictionary_recursively(
                    path,
                    dataset_mapping,
                    current_dictionary
                )

        if os.path.isfile(current_path):
            _, current_filename = os.path.split(current_path)
            target_name = current_filename.split(".")[0] + ".h5"
            if current_dictionary != None and current_filename.split(".")[-1] != "csv":
                current_dictionary.parse_and_append(target_name, current_path)

        return current_dictionary

    class Pregenerator:
        """Pregenerate preprocessing to h5 file(s) from a video"""

        DICTIONARY_FILENAME = "generated_dictionary.csv"

        def __init__(self, target_folder: str, motion_vector_processor_kwargs: dict, mask_generator_kwargs: dict, optical_flow_kwargs: dict, directory_mapping: dict) -> None:
            self.__motion_vector_processor_kwargs: dict = motion_vector_processor_kwargs
            self.__mask_generator_kwargs: dict = mask_generator_kwargs
            self.__optical_flow_kwargs: dict = optical_flow_kwargs
            self.__target_folder: str = target_folder
            os.makedirs(os.path.dirname(target_folder), exist_ok=True)

            self.__dataset_dictionary: DatasetDictionary = DatasetDictionary(
                os.path.abspath(os.path.join(target_folder, self.DICTIONARY_FILENAME)),
                directory_mapping
            )
            self.__initialized: bool = False

        def init_models(self, sample_video_path: str) -> "PreprocessingManagers.Pregenerator":
            """Initialize all preprocessing model"""

            print("Pregenerator is initializing and warming models...")

            decoder = VideoDecoderProcessSpawner(path=sample_video_path).start()
            frame_data: DecodedData = decoder.read()
            assert frame_data[0], "DECODER ERROR, ABORTING..."

            rgb_frame: FrameRGB = frame_data[1]
            raw_motion_vector: RawMotionVectors = frame_data[2]

            self.__motion_vector_processor_kwargs["input_size"] = rgb_frame.shape[:2]

            self.__motion_vector_processor: MotionVectorProcessor = MotionVectorProcessor(**self.__motion_vector_processor_kwargs)
            self.__mask_generator: MaskGenerator = MaskGenerator(**self.__mask_generator_kwargs)
            self.__optical_flow_generator: OpticalFlowGenerator = OpticalFlowGenerator(**self.__optical_flow_kwargs)

            self.__motion_vector_processor.process(raw_motion_vector)
            self.__motion_vector_processor.process(raw_motion_vector)
            self.__motion_vector_processor.process(raw_motion_vector)
            self.__mask_generator.forward_once_with_mcbb(rgb_frame)
            self.__mask_generator.forward_once_with_mcbb(rgb_frame)
            self.__mask_generator.forward_once_with_mcbb(rgb_frame)
            self.__optical_flow_generator.forward_once_auto(rgb_frame, rgb_frame)
            self.__optical_flow_generator.forward_once_auto(rgb_frame, rgb_frame)
            self.__optical_flow_generator.forward_once_auto(rgb_frame, rgb_frame)
            self.__initialized = True

            print("Initializing and warming models done")

            return self

        def pregenerate_preprocessing(self, video_path: str, target_name: str = "", frame_length: int = 100) -> bool:
            """pregenerate preprocessing data and save as h5 file from given video path"""
            if not self.__initialized:
                self.init_models(video_path)
            if target_name == "":
                _, original_filename = os.path.split(video_path)
                target_name = original_filename.split(".")[0] + ".h5"

            abs_video_path = os.path.abspath(video_path)
            abs_target_path = os.path.abspath(os.path.join(self.__target_folder, target_name))
            rel_video_path = os.path.relpath(abs_video_path, abs_target_path)

            h5_handle: h5py.File = h5py.File(abs_target_path, mode="w")
            mock_mve: MotionVectorProcessorMocker = MotionVectorProcessorMocker(h5_handle, **self.__motion_vector_processor_kwargs)
            mock_maskgen: MaskGeneratorMocker = MaskGeneratorMocker(h5_handle, **self.__mask_generator_kwargs)
            mock_opflowgen: OpticalFlowGeneratorMocker = OpticalFlowGeneratorMocker(h5_handle, **self.__optical_flow_kwargs)

            decoder: VideoDecoderProcessSpawner = VideoDecoderProcessSpawner(path=abs_video_path).start()
            frame_data: DecodedData = decoder.read()
            last_frame_data: DecodedData = deepcopy(frame_data)
            assert frame_data[0], "DECODER ERROR, ABORTING..."

            with tqdm(total=frame_length, desc=target_name) as progress_bar:
                while True:
                    available, rgb_frame, raw_motion_vectors, _ = frame_data
                    _, last_rgb_frame, _, _ = last_frame_data

                    if not available:
                        break

                    motion_vector_frame: MotionVectorFrame = self.__motion_vector_processor.process(raw_motion_vectors)
                    mock_mve.append(motion_vector_frame)

                    mask_data: MaskWithMostCenterBoundingBoxData = self.__mask_generator.forward_once_with_mcbb(rgb_frame)
                    mock_maskgen.append(mask_data)

                    optical_flow_frame: OpticalFlowFrame = self.__optical_flow_generator.forward_once_auto(last_rgb_frame, rgb_frame)
                    mock_opflowgen.append(optical_flow_frame)

                    last_frame_data = frame_data
                    frame_data = decoder.read()

                    progress_bar.update(1)

            decoder.stop()
            del decoder

            dictionary_dict: dict = self.__dataset_dictionary.parse(target_name, rel_video_path)
            for k, v in dictionary_dict.items():
                h5_handle.attrs[k] = v

            mock_mve.save()
            mock_maskgen.save()
            mock_opflowgen.save()
            h5_handle.close()

            return self.__dataset_dictionary.append(dictionary_dict)
