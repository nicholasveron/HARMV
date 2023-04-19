PREPROCESSING_TARGET_RESOLUTION = 416
PREPROCESSING_ARGUMENTS_MOTION_VECTOR_PROCESSOR = {
    "raw_motion_vectors": True,
    "target_size": PREPROCESSING_TARGET_RESOLUTION,
}

PREPROCESSING_ARGUMENTS_MASK_GENERATOR = {
    "weight_path": './libs/yolov7-mask/yolov7-mask.pt',
    "hyperparameter_path": './libs/yolov7-mask/data/hyp.scratch.mask.yaml',
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "target_size": PREPROCESSING_TARGET_RESOLUTION,
    "optimize_model": True,
    "bounding_box_grouping_range_scale": 10
}

PREPROCESSING_ARGUMENTS_OPTICAL_FLOW_GENERATOR = {
    "model_type": "flowformer",
    "model_pretrained": "sintel",
    "raw_optical_flows": True,
    "target_size": PREPROCESSING_TARGET_RESOLUTION,
    "overlap_grid_mode": True,
    "overlap_grid_scale": 2
}