from importables.preprocessing_managers import PreprocessingManagers, DatasetDictionary

test_video = "/mnt/c/Skripsi/dataset-h264/R002A120/S018C001P008R002A120_rgb.mp4"
test_target_path = "/mnt/c/Skripsi/dataset-pregen/"
args_mvex = {
    "bound":  32,
    "raw_motion_vectors": True,
    "camera_realtime": False,
    "camera_update_rate":  60,
    "camera_buffer_size":  0,
    "letterboxed": True,
    "new_shape": 320,
    "box": False,
    "color": (114, 114, 114, 128, 128),
    "stride":  32,
}

args_yolo = {
    "weight_path": './libs/yolov7-mask/yolov7-mask.pt',
    "hyperparameter_path": './libs/yolov7-mask/data/hyp.scratch.mask.yaml',
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
}

args_opgen = {
    "model_type": "raft_small",
    "model_pretrained": "things",
    "bound": 32,
    "raw_optical_flows": True,
}


pmgr = PreprocessingManagers.Pregenerator(
    target_folder=test_target_path,
    motion_vector_kwargs=args_mvex,
    mask_generator_kwargs=args_yolo,
    optical_flow_kwargs=args_opgen,
    directory_mapping=DatasetDictionary.Mappings.NTU_ACTION_RECOGNITION_DATASET
)

pmgr.pregenerate_preprocessing(test_video)

# PreprocessingManagers.generate_dataset_dictionary_recursively("/mnt/c/Skripsi/dataset-h264", DatasetDictionary.Mappings.NTU_ACTION_RECOGNITION_DATASET)
