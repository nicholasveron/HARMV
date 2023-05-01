from importables.preprocessing_managers import PreprocessingManagers, DatasetDictionary
import cv2
import h5py
import numpy
from importables.custom_types import DecodedData, FrameRGB, RawMotionVectors, BoundingBoxXY1XY2, SegmentationMask, MotionVectorFrame, MaskWithMostCenterBoundingBoxData, OpticalFlowFrame
from copy import deepcopy
from importables.utilities import Utilities
import time

# test_video = "/mnt/c/Skripsi/dataset-h264/R002A120/S018C001P008R002A120_rgb.mp4"
# test_target_path = "/mnt/c/Skripsi/dataset-pregen/"

target_size = 416

# args_mvprocessor = {
#     "raw_motion_vectors": True,
#     "target_size": target_size,
# }

# args_maskgenerator = {
#     "weight_path": './libs/yolov7-mask/yolov7-mask.pt',
#     "hyperparameter_path": './libs/yolov7-mask/data/hyp.scratch.mask.yaml',
#     "confidence_threshold": 0.5,
#     "iou_threshold": 0.45,
#     "target_size": target_size,
#     "optimize_model": True,
#     "bounding_box_grouping_range_scale": 10
# }

# args_ofgenerator = {
#     "model_type": "flowformer",
#     "model_pretrained": "sintel",
#     "raw_optical_flows": True,
#     "target_size": target_size,
#     "overlap_grid_mode": True,
#     "overlap_grid_scale": 2
# }

# pmgr = PreprocessingManagers.Pregenerator(
#     target_folder=test_target_path,
#     motion_vector_processor_kwargs=args_mvprocessor,
#     mask_generator_kwargs=args_maskgenerator,
#     optical_flow_generator_kwargs=args_ofgenerator,
#     directory_mapping=DatasetDictionary.Mappings.NTU_ACTION_RECOGNITION_DATASET
# )

# pmgr.pregenerate_preprocessing(test_video)

# PreprocessingManagers.generate_dataset_dictionary_recursively("/mnt/c/Skripsi/dataset-h264", DatasetDictionary.Mappings.NTU_ACTION_RECOGNITION_DATASET)

# PreprocessingManagers.consolidate_pregenerated_preprocessing_files("/mnt/c/Skripsi/dataset-pregen", DatasetDictionary.Mappings.NTU_ACTION_RECOGNITION_DATASET)

h5_example_path = "/mnt/c/Skripsi/dataset-pregen/S013C001P019R002A016_rgb.h5"
h5py_handle = h5py.File(h5_example_path, rdcc_nbytes=1024**2*4000, rdcc_nslots=1e7)

args_mvprocessor = {
    "raw_motion_vectors": False,
    "target_size": target_size,
}

args_maskgenerator = {
    "weight_path": './libs/yolov7-mask/yolov7-mask.pt',
    "hyperparameter_path": './libs/yolov7-mask/data/hyp.scratch.mask.yaml',
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "target_size": target_size,
    "optimize_model": True,
    "bounding_box_grouping_range_scale": 10
}

args_ofgenerator = {
    "model_type": "flowformer",
    "model_pretrained": "sintel",
    "raw_optical_flows": False,
    "target_size": target_size,
    "overlap_grid_mode": True,
    "overlap_grid_scale": 2
}

pmld = PreprocessingManagers.Loader(DatasetDictionary.Mappings.NTU_ACTION_RECOGNITION_DATASET, args_maskgenerator, args_mvprocessor, args_ofgenerator, True)

# v, ma, mv, op = pmld.load_from_h5(h5_example_path,10,20)
# print(len(ma), len(mv), len(op))

v, ma, mv, op = pmld.load_from_h5(h5py_handle)
assert v is not None and mv is not None and op is not None
frame_data: DecodedData = v.read()
last_frame_data: DecodedData = deepcopy(frame_data)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q') or not frame_data[0]:
        v.stop()
        break

    last_rgb_frame: FrameRGB = last_frame_data[1]
    rgb_frame: FrameRGB = frame_data[1]
    raw_mv: RawMotionVectors = frame_data[2]
    mv_frame: MotionVectorFrame = mv.process(raw_mv)
    rgb_frame = cv2.resize(rgb_frame, (mv_frame.shape[1], mv_frame.shape[0]))
    op_frame: OpticalFlowFrame = op.forward_once_auto(last_rgb_frame, rgb_frame)
    mask_data: MaskWithMostCenterBoundingBoxData = ma.forward_once_with_mcbb(rgb_frame)

    mcbb: BoundingBoxXY1XY2 = mask_data[2]
    mask_frame: SegmentationMask = mask_data[1]

    masked_rgb_frame = deepcopy(rgb_frame)
    masked_rgb_frame[~mask_frame] = (255, 255, 255)
    masked_rgb_frame = Utilities.crop_to_bb_and_resize(masked_rgb_frame, mcbb, rgb_frame.shape[:2], rgb_frame.shape[:2])

    rgb_stacked = numpy.vstack((rgb_frame,
                               masked_rgb_frame))

    mv_fl_x = mv_frame[:, :, 0]
    mv_fl_y = mv_frame[:, :, 1]

    mv_fl_x_s = numpy.dstack((mv_fl_x, mv_fl_x, mv_fl_x))
    mv_fl_y_s = numpy.dstack((mv_fl_y, mv_fl_y, mv_fl_y))

    mv_stacked = numpy.vstack((mv_fl_x_s,
                               mv_fl_y_s))

    op_fl_x = op_frame[:, :, 0]
    op_fl_y = op_frame[:, :, 1]

    # Display the resulting frame

    op_fl_x_s = numpy.dstack((op_fl_x, op_fl_x, op_fl_x))
    op_fl_y_s = numpy.dstack((op_fl_y, op_fl_y, op_fl_y))

    op_stacked = numpy.vstack((op_fl_x_s,
                               op_fl_y_s))

    fshow = numpy.hstack(
        (
            rgb_stacked,
            mv_stacked,
            op_stacked,
            # flxyrgb
        )
    )

    cv2.imshow('frame', fshow)
    last_frame_data = frame_data
    frame_data = v.read()
    time.sleep(1/30)


v.stop()
cv2.destroyAllWindows()
