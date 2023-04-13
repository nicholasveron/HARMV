from importables.preprocessing_managers import PreprocessingManagers, DatasetDictionary

test_video = "/mnt/c/Skripsi/dataset-h264/R002A120/S018C001P008R002A120_rgb.mp4"
test_target_path = "/mnt/c/Skripsi/dataset-pregen/"

target_size = 416

args_mvprocessor = {
    "raw_motion_vectors": True,
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
    "raw_optical_flows": True,
    "target_size": target_size,
    "overlap_grid_mode": True,
    "overlap_grid_scale": 2
}

pmgr = PreprocessingManagers.Pregenerator(
    target_folder=test_target_path,
    motion_vector_processor_kwargs=args_mvprocessor,
    mask_generator_kwargs=args_maskgenerator,
    optical_flow_kwargs=args_ofgenerator,
    directory_mapping=DatasetDictionary.Mappings.NTU_ACTION_RECOGNITION_DATASET
)

pmgr.pregenerate_preprocessing(test_video)

# PreprocessingManagers.generate_dataset_dictionary_recursively("/mnt/c/Skripsi/dataset-h264", DatasetDictionary.Mappings.NTU_ACTION_RECOGNITION_DATASET)
