import os
import json
import hdf5plugin
from tqdm.auto import tqdm
from importables.preprocessing_managers import DatasetDictionary, PreprocessingManagers
from importables.custom_datasets import register_datasets

for name, data in tqdm(register_datasets.items()):

    root_path, current_dictionary, _ = data

    dataset_splits: dict[str,dict[str,list[int]]] = json.load(open(os.path.join(root_path, "train_test_split.json")))

    for split, traintest in dataset_splits.items():
        for part, value in  traintest.items():
            PreprocessingManagers.consolidate_pregenerated_preprocessing_files(
                root_path, 
                current_dictionary,
                f"{split}_{part}",
                value
            ) 
            print(f"{split}_{part}")

# del register_datasets["NTU RGB+D 120 (limited)"]

# for name, data in tqdm(register_datasets.items()):

#     root_path, current_dictionary, _ = data

#     dataset_splits: dict[str,dict[str,list[int]]] = json.load(open(os.path.join(root_path, "train_test_split.json")))
#     del dataset_splits["preset2"]
#     del dataset_splits["preset3"]

#     for split, traintest in dataset_splits.items():
#         del traintest["train"]
#         for part, value in  traintest.items():
#             PreprocessingManagers.consolidate_pregenerated_preprocessing_files(
#                 root_path, 
#                 current_dictionary,
#                 f"{split}_{part}",
#                 value,
#                 hdf5plugin.Blosc2.SHUFFLE,
#                 9
#             ) 
#             print(f"{split}_{part}")