"""Module for preprocessing managers such as generator, viewer, loader, etc"""
import os
import cv2
import tables
import h5py
import pandas
from copy import deepcopy
from tqdm.auto import tqdm
from .video_decoder import VideoDecoderProcessSpawner
from .mask_generator import MaskGenerator, MaskGeneratorMocker, MaskWithMostCenterBoundingBoxData
from .optical_flow_generator import OpticalFlowGenerator, OpticalFlowGeneratorMocker, OpticalFlowFrame
from .motion_vector_processor import MotionVectorProcessor, MotionVectorProcessorMocker, MotionVectorFrame
from .custom_types import (
    Union,
    Tuple,
    ndarray,
    FrameRGB,
    DecodedData,
    MotionVectorFrame,
    RawMotionVectors
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
            ".": "GeneratedFileName",
            "#": "FrameCount",
        }

        def __new__(cls):
            raise TypeError('DatasetDictionary.Mappings is a constant class and cannot be instantiated')

    UNKNOWN_KEY = "_"
    GENERATED_FILENAME_KEY = "."
    SOURCE_FILEPATH_KEY = "/"
    FRAME_COUNT_KEY = "#"

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

    def parse(self, dataset_filename: str, source_filepath: str, frame_count: int = -1) -> dict:
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

        if frame_count == -1:
            video_capturer: cv2.VideoCapture = cv2.VideoCapture()
            video_capturer.open(source_filepath)
            frame_count = int(video_capturer.get(cv2.CAP_PROP_FRAME_COUNT))
        filtered[self.__dataset_mapping[self.FRAME_COUNT_KEY]] = frame_count

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

    def parse_and_append(self, dataset_filename: str, source_filepath: str, frame_count: int = -1) -> bool:
        dataset_dict: dict = self.parse(dataset_filename, source_filepath, frame_count)
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

    @staticmethod
    def consolidate_pregenerated_preprocessing_files(root_folder: str, dataset_mapping: dict) -> None:
        abs_root_folder: str = os.path.abspath(root_folder)
        dataset_dictionary_filename: str = os.path.join(abs_root_folder, "generated_dictionary.csv")
        dataset_dictionary_pd: pandas.DataFrame = DatasetDictionary(dataset_dictionary_filename, dataset_mapping).as_DataFrame()

        h5py_base_filename: str = os.path.join(abs_root_folder, "generated_consolidation.h5")
        h5py_base_file: h5py.File = h5py.File(h5py_base_filename, "w", rdcc_nbytes=1024**2*4000, rdcc_nslots=1e7)

        for index in tqdm(range(len(dataset_dictionary_pd)), desc="Consolidating pregenerated files..."):
            current_series: pandas.Series = dataset_dictionary_pd.iloc[index]
            generated_filename: str = current_series[dataset_mapping[DatasetDictionary.GENERATED_FILENAME_KEY]]  # type: ignore
            h5py_current_filename: str = os.path.join(abs_root_folder, generated_filename)
            current_h5py_file: h5py.File = h5py.File(h5py_current_filename, rdcc_nbytes=1024**2*4000, rdcc_nslots=1e7)
            current_group: h5py.Group = h5py_base_file.create_group(generated_filename)
            for key, value in current_h5py_file.attrs.items():
                current_group.attrs[key] = value
            for dataset_name, value in current_h5py_file.items():
                if isinstance(value, h5py.Dataset):
                    data: ndarray = value[()]
                    current_group.create_dataset(name=dataset_name, data=data, chunks=data.shape, compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)  # blosc compression

        h5py_base_file.close()

    class Pregenerator:
        """Pregenerate preprocessing to h5 file(s) from a video. This class is designed to be independent from other preprocessing classes"""

        DICTIONARY_FILENAME = "generated_dictionary.csv"

        def __init__(self, target_folder: str, mask_generator_kwargs: dict, motion_vector_processor_kwargs: dict, optical_flow_generator_kwargs: dict, directory_mapping: dict) -> None:
            self.__mask_generator_kwargs: dict = mask_generator_kwargs
            self.__motion_vector_processor_kwargs: dict = motion_vector_processor_kwargs
            self.__optical_flow_generator_kwargs: dict = optical_flow_generator_kwargs
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
            self.__optical_flow_generator: OpticalFlowGenerator = OpticalFlowGenerator(**self.__optical_flow_generator_kwargs)

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

            abs_video_path: str = os.path.abspath(video_path)
            abs_target_path: str = os.path.abspath(os.path.join(self.__target_folder, target_name))
            rel_video_path: str = os.path.relpath(abs_video_path, abs_target_path)

            h5py_instance: h5py.File = h5py.File(abs_target_path, mode="w", rdcc_nbytes=1024**2*4000, rdcc_nslots=1e7)
            mock_mask_generator: MaskGeneratorMocker = MaskGeneratorMocker(h5py_instance, **self.__mask_generator_kwargs)
            mock_motion_vector_processor: MotionVectorProcessorMocker = MotionVectorProcessorMocker(h5py_instance, **self.__motion_vector_processor_kwargs)
            mock_optical_flow_generator: OpticalFlowGeneratorMocker = OpticalFlowGeneratorMocker(h5py_instance, **self.__optical_flow_generator_kwargs)

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
                    mock_motion_vector_processor.append(motion_vector_frame)

                    mask_data: MaskWithMostCenterBoundingBoxData = self.__mask_generator.forward_once_with_mcbb(rgb_frame)
                    mock_mask_generator.append(mask_data)

                    optical_flow_frame: OpticalFlowFrame = self.__optical_flow_generator.forward_once_auto(last_rgb_frame, rgb_frame)
                    mock_optical_flow_generator.append(optical_flow_frame)

                    last_frame_data = frame_data
                    frame_data = decoder.read()

                    progress_bar.update(1)

            decoder.stop()
            del decoder

            dictionary_dict: dict = self.__dataset_dictionary.parse(target_name, rel_video_path, len(mock_mask_generator))
            for k, v in dictionary_dict.items():
                h5py_instance.attrs[k] = v

            mock_mask_generator.save()
            mock_motion_vector_processor.save()
            mock_optical_flow_generator.save()
            h5py_instance.file.close()

            return self.__dataset_dictionary.append(dictionary_dict) and self.__dataset_dictionary.save()

    class Loader:
        """Loads preprocessed data (h5 File) with its video and spawns mock classes"""

        def __init__(self, dataset_mapping: dict,  mask_generator_kwargs: dict, motion_vector_processor_kwargs: Union[dict, None] = None, optical_flow_kwargs: Union[dict, None] = None, load_rgb_video: bool = False) -> None:
            self.__dataset_mapping: dict = dataset_mapping
            self.__mask_generator_kwargs: dict = mask_generator_kwargs
            self.__motion_vector_processor_kwargs: Union[dict, None] = motion_vector_processor_kwargs
            self.__optical_flow_kwargs: Union[dict, None] = optical_flow_kwargs
            self.__load_rgb_video: bool = load_rgb_video
            self.__previously_loaded: bool = False
            assert self.__motion_vector_processor_kwargs or self.__optical_flow_kwargs, "Either motion vector arguments or optical flow arguments must be given"

        def load_from_h5(self, h5py_instance: Union[h5py.File, h5py.Group], start_index: int = -1, stop_index: int = -1) -> Tuple[Union[VideoDecoderProcessSpawner, None], MaskGeneratorMocker, Union[MotionVectorProcessorMocker, None], Union[OpticalFlowGeneratorMocker, None]]:

            self.close()

            self.__last_h5py_instance: Union[h5py.File, h5py.Group] = h5py_instance
            abs_file_path: str = self.__last_h5py_instance.file.filename

            assert self.__dataset_mapping["/"] in self.__last_h5py_instance.attrs
            rel_video_path: str = str(self.__last_h5py_instance.attrs[self.__dataset_mapping["/"]])
            abs_video_path: str = os.path.abspath(os.path.join(abs_file_path, rel_video_path))

            self.__last_video_spawner: Union[VideoDecoderProcessSpawner, None] = None
            if self.__load_rgb_video:
                if start_index < 0 and stop_index < 0:
                    self.__last_video_spawner = VideoDecoderProcessSpawner(abs_video_path).start()
                else:
                    print("Loader currently doesn't support rgb video partial load, skipping rgb video")

            self.__last_mask_generator: MaskGeneratorMocker = MaskGeneratorMocker(self.__last_h5py_instance, **self.__mask_generator_kwargs)
            assert self.__last_mask_generator.load(start_index, stop_index), "Mask Generator Mocker Failed to Load"

            self.__last_motion_vector_processor: Union[MotionVectorProcessor, None] = None
            self.__last_optical_flow_generator: Union[OpticalFlowGenerator, None] = None

            if self.__motion_vector_processor_kwargs:
                self.__last_motion_vector_processor = MotionVectorProcessorMocker(self.__last_h5py_instance, **self.__motion_vector_processor_kwargs)
                assert self.__last_motion_vector_processor.load(start_index, stop_index), "Motion Vector Processor Mocker Failed to Load"

            if self.__optical_flow_kwargs:
                self.__last_optical_flow_generator = OpticalFlowGeneratorMocker(self.__last_h5py_instance, **self.__optical_flow_kwargs)
                assert self.__last_optical_flow_generator.load(start_index, stop_index), "Optical Flow Generator Failed to Load"

            return self.__last_video_spawner, self.__last_mask_generator, self.__last_motion_vector_processor, self.__last_optical_flow_generator

        def close(self) -> None:
            """Properly close previous loaded preprocessed data"""

            if self.__previously_loaded:
                del self.__last_video_spawner
                del self.__last_mask_generator
                del self.__last_motion_vector_processor
                del self.__last_optical_flow_generator

                # only close h5py file
                if isinstance(self.__last_h5py_instance, h5py.File):
                    self.__last_h5py_instance.close()

        def __del__(self) -> None:
            self.close()
