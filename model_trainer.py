import os
import sys
import time
import json
import copy
import h5py
from tqdm.auto import tqdm
import torch
import numpy
import argparse
import torchinfo
import torchvision
import torch.utils.data
import importables.models
import importables.constants
import sklearn.preprocessing
import sklearn.model_selection
import importables.custom_datasets
import torch.utils.tensorboard.writer
from importables.utilities import Utilities
from importables.preprocessing_managers import PreprocessingManagers, DatasetDictionary
from importables.custom_datasets import FlowDataset
from importables.constants import (
    DEFAULT_TRAINING_PARAMETERS,
)
from importables.custom_types import (
    List
)
ramdisk_manager = Utilities.RamDiskManager("/mnt/ramdisk")
def clear_ramdisk_hook(type, value, tb):
    import traceback
    traceback.print_exception(type, value, tb)
    ramdisk_manager.clear()

sys.excepthook = clear_ramdisk_hook

Utilities.set_all_seed(importables.constants.RANDOM_SEED_BYTES)
torch.cuda.empty_cache()
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.system('clear')

parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Train models via command-line arguments")

parser.add_argument("--use_cmd_args", action="store_true", help="Enable command-line arguments")
parser.add_argument("--model", type=int, nargs="?", help="Model to train")
parser.add_argument("--batch_size", type=int, nargs="?", help="Batch size")
parser.add_argument("--timesteps", type=int, nargs="?", help="Timesteps")
parser.add_argument("--hidden_cell_count", type=int, nargs="?", help="Hidden Cell Count")
parser.add_argument("--learning_rate", type=float, nargs="?", help="Learning Rate")
parser.add_argument("--epoch", type=int, nargs="?", help="Epoch")
parser.add_argument("--epoch_ask_continue", type=bool, nargs="?", help="Epoch Ask Continue")
parser.add_argument("--dataset", type=int, nargs="?", help="Dataset")
parser.add_argument("--split", type=int, nargs="?", help="Split")
parser.add_argument("--data_use", type=float, nargs="?", help="Data Use %")
parser.add_argument("--data_selector", type=str, nargs="?", help="Data Selector")
parser.add_argument("--train_in_memory", type=bool, nargs="?", help="Train In Memory")
parser.add_argument("--test_in_memory", type=bool, nargs="?", help="Test In Memory")
parser.add_argument("--mask", type=bool, nargs="?", help="Mask Data")
parser.add_argument("--crop", type=bool, nargs="?", help="Crop Data")
parser.add_argument("--bounding_value", type=float, nargs="?", help="Bounding Value")
parser.add_argument("--run_comment", type=str, nargs="?", help="Run Comment")

parsed_args: argparse.Namespace = parser.parse_args()

current_parameters: dict = copy.deepcopy(DEFAULT_TRAINING_PARAMETERS)

print("Select model to train")
model_list: List = []
for obj in importables.models.register_model:
    print(f"\t[{len(model_list)}]:", obj.__name__)
    model_list.append(obj)

print(f"(Latest)[0 - {len(model_list)-1}]: ", end="")
if parsed_args.use_cmd_args:
    model_train_index_input = parsed_args.model if parsed_args.model is not None else ""
    print(model_train_index_input)
else:
    model_train_index_input = input().strip().lower()
model_train_index = int(model_train_index_input if model_train_index_input != "" else -1)

current_model_class = model_list[model_train_index]
print("Trainer will train model:", current_model_class, f"({current_model_class.__name__})")
current_parameters["model"] = current_model_class.__name__

print("")
print("Training Parameters")

print(f"\tBatch Size ({current_parameters['batch_size']})[1-inf]: ", end="")
if parsed_args.use_cmd_args:
    batch_size_input= parsed_args.batch_size if parsed_args.batch_size is not None else ""
    print(batch_size_input)
else:
    batch_size_input = input().strip().lower()
if batch_size_input != "":
    current_parameters["batch_size"] = int(batch_size_input)

print(f"\tTimesteps ({current_parameters['timesteps']})[1-inf]: ", end="")
if parsed_args.use_cmd_args:
    timesteps_input= parsed_args.timesteps if parsed_args.timesteps is not None else ""
    print(timesteps_input)
else:
    timesteps_input = input().strip().lower()
if timesteps_input != "":
    current_parameters["timesteps"] = int(timesteps_input)

print(f"\tHidden Cell Count ({current_parameters['hidden_cell_count']})[1-inf]: ", end="")
if parsed_args.use_cmd_args:
    hidden_cell_count_input= parsed_args.hidden_cell_count if parsed_args.hidden_cell_count is not None else ""
    print(hidden_cell_count_input)
else:
    hidden_cell_count_input = input().strip().lower()
if hidden_cell_count_input != "":
    current_parameters["hidden_cell_count"] = int(hidden_cell_count_input)

print(f"\tLearning Rate ({current_parameters['learning_rate']})[0-inf]: ", end="")
if parsed_args.use_cmd_args:
    learning_rate_input= parsed_args.learning_rate if parsed_args.learning_rate is not None else ""
    print(learning_rate_input)
else:
    learning_rate_input = input().strip().lower()
if learning_rate_input != "":
    current_parameters["learning_rate"] = float(learning_rate_input)

print(f"\tEpoch (Ask each epoch)[1-inf]: ", end="")
if parsed_args.use_cmd_args:
    epoch_input= parsed_args.epoch if parsed_args.epoch is not None else ""
    print(epoch_input)
else:
    epoch_input = input().strip().lower()
if epoch_input != "":
    current_parameters["epoch"] = int(epoch_input)

if current_parameters["epoch"] > 0:
    print(f"\tEpoch Ask Continue ({current_parameters['epoch_ask_continue']})[y/n]: ", end="")
    if parsed_args.use_cmd_args:
        epoch_ask_continue_input= ("y" if parsed_args.epoch_ask_continue else "n") if parsed_args.epoch_ask_continue is not None else ""
        print(epoch_ask_continue_input)
    else:
        epoch_ask_continue_input = input().strip().lower()
    if epoch_ask_continue_input in ("y", "n"):
        current_parameters["epoch_ask_continue"] = epoch_ask_continue_input == "y"
else:
    current_parameters["epoch_ask_continue"] = True

print("\tSelect dataset to train")
dataset_list: List = []
for dataset_name, dataset in importables.custom_datasets.register_datasets.items():
    print(f"\t\t[{len(dataset_list)}]:", dataset_name)
    dataset_list.append(dataset)

print(f"\t({current_parameters['dataset']})[0 - {len(dataset_list)-1}]: ", end="")
if parsed_args.use_cmd_args:
    dataset_index_input = parsed_args.dataset if parsed_args.dataset is not None else ""
    print(dataset_index_input)
else:
    dataset_index_input = input().strip().lower()
dataset_index = int(dataset_index_input if dataset_index_input != "" else current_parameters['dataset'])
current_parameters['dataset_name'] = list(importables.custom_datasets.register_datasets.keys())[dataset_index]
current_root, current_dictionary, current_mapping = dataset_list[dataset_index]

train_test_split_dict: dict = {}
with open(os.path.join(current_root, "train_test_split.json")) as f:
    train_test_split_dict = json.load(f)
print("\tSelect train/test split to train")
split_list: List = []
for split_name, split in train_test_split_dict.items():
    print(f"\t\t[{len(split_list)}]:", split_name)
    split_list.append(split)

print(f"\t({current_parameters['split']})[0 - {len(split_list)-1}]: ", end="")
if parsed_args.use_cmd_args:
    split_index_input = parsed_args.split if parsed_args.split is not None else ""
    print(split_index_input)
else:
    split_index_input = input().strip().lower()
split_index = int(split_index_input if split_index_input != "" else current_parameters['split'])
current_parameters['split_name'] = list(train_test_split_dict.keys())[split_index]
current_split: dict = split_list[split_index]

print(f"\tData Use % ({current_parameters['data_use']})[0-1]: ", end="")
if parsed_args.use_cmd_args:
    data_use_input= parsed_args.data_use if parsed_args.data_use else ""
    print(data_use_input)
else:
    data_use_input = input().strip().lower()
if data_use_input != "":
    current_parameters["data_use"] = float(data_use_input)

print(f"\tData Selector ({current_parameters['data_selector']})[str]: ", end="")
if parsed_args.use_cmd_args:
    data_selector_input= parsed_args.data_selector if parsed_args.data_selector else ""
    print(data_selector_input)
else:
    data_selector_input = input().strip().lower()
if data_selector_input != "":
    current_parameters["data_selector"] = str(data_selector_input)

print(f"\tEnable Train In Memory ({current_parameters['train_in_memory']})[y/n]: ", end="")
if parsed_args.use_cmd_args:
    train_in_memory_input= ("y" if parsed_args.train_in_memory else "n") if parsed_args.train_in_memory is not None else ""
    print(train_in_memory_input)
else:
    train_in_memory_input = input().strip().lower()
if train_in_memory_input in ("y", "n"):
    current_parameters["train_in_memory"] = train_in_memory_input == "y"

print(f"\tEnable Test In Memory ({current_parameters['test_in_memory']})[y/n]: ", end="")
if parsed_args.use_cmd_args:
    test_in_memory_input= ("y" if parsed_args.test_in_memory else "n") if parsed_args.test_in_memory is not None else ""
    print(test_in_memory_input)
else:
    test_in_memory_input = input().strip().lower()
if test_in_memory_input in ("y", "n"):
    current_parameters["test_in_memory"] = test_in_memory_input == "y"

print(f"\tMask Data ({current_parameters['mask']})[y/n]: ", end="")
if parsed_args.use_cmd_args:
    mask_input= ("y" if parsed_args.mask else "n") if parsed_args.mask is not None else ""
    print(mask_input)
else:
    mask_input = input().strip().lower()
if mask_input in ("y", "n"):
    current_parameters["mask"] = mask_input == "y"

print(f"\tCrop Data ({current_parameters['crop']})[y/n]: ", end="")
if parsed_args.use_cmd_args:
    crop_input= ("y" if parsed_args.crop else "n") if parsed_args.crop is not None else ""
    print(crop_input)
else:
    crop_input = input().strip().lower()
if crop_input in ("y", "n"):
    current_parameters["crop"] = crop_input == "y"

print(f"\tBounding Value ({current_parameters['bounding_value']})[1-inf]: ", end="")
if parsed_args.use_cmd_args:
    bounding_value_input= parsed_args.bounding_value if parsed_args.bounding_value else ""
    print(bounding_value_input)
else:
    bounding_value_input = input().strip().lower()
if bounding_value_input != "":
    current_parameters["bounding_value"] = int(bounding_value_input)

print("")
print("Current Run Comment [str]: ", end="")
if parsed_args.use_cmd_args:
    current_parameters["run_comment"] = parsed_args.run_comment if parsed_args.run_comment else ""
    print(current_parameters["run_comment"])
else:
    current_parameters["run_comment"] = input().strip().lower()
current_parameters["model_comment"] = current_model_class.comment()

print("")
print("Preparing Dataset ...")
pd_pregen = DatasetDictionary(os.path.join(current_root, "generated_dictionary.csv"), current_dictionary).as_DataFrame()
train_pd = pd_pregen.iloc[current_split["train"]]
test_pd = pd_pregen.iloc[current_split["test"]]

if current_parameters["data_use"] < 1:
    _, train_pd = sklearn.model_selection.train_test_split(train_pd, test_size=current_parameters["data_use"], shuffle=True, stratify=train_pd[current_dictionary["A"]])
    _, test_pd = sklearn.model_selection.train_test_split(test_pd, test_size=current_parameters["data_use"], shuffle=True, stratify=test_pd[current_dictionary["A"]])

label_encoder = sklearn.preprocessing.LabelEncoder().fit(numpy.expand_dims(train_pd[current_dictionary["A"]].astype(int).unique(), 1))
class_count = len(label_encoder.classes_)

h5py_train_filename: str = f"generated_consolidation_{current_parameters['split_name']}_train.h5"
h5py_test_filename: str = f"generated_consolidation_{current_parameters['split_name']}_test.h5"
h5py_train_filepath: str = os.path.join(current_root, h5py_train_filename)
h5py_test_filepath: str = os.path.join(current_root, h5py_test_filename)

if current_parameters["train_in_memory"]:
    h5py_train_filepath = ramdisk_manager.copy(h5py_train_filepath)


if current_parameters["test_in_memory"]:
    h5py_test_filepath = ramdisk_manager.copy(h5py_test_filepath)

h5py_train = h5py.File(h5py_train_filepath, rdcc_nbytes=1024**2*4000, rdcc_nslots=1e7, rdcc_w0 = 0, swmr= True, libver='latest')
h5py_test = h5py.File(h5py_test_filepath, rdcc_nbytes=1024**2*4000, rdcc_nslots=1e7, rdcc_w0 = 0, swmr=True, libver='latest')

train_pd.reset_index()
test_pd.reset_index()

print("")
print("LabelEncoder Output: ")
print(label_encoder.classes_)
print("")
print("Train class sample counts")
print(train_pd.ActionID.value_counts().rename(index=current_mapping))
print("")
print("Test class sample counts")
print(test_pd.ActionID.value_counts().rename(index=current_mapping))

print("")
motion_vector_kwargs = None
optical_flow_kwargs = None
if current_parameters["data_selector"] == "motion_vector":
    # motion_vector_kwargs = {"bound": current_parameters["bounding_value"]}
    motion_vector_kwargs = {"raw_motion_vectors": True}
if current_parameters["data_selector"] == "optical_flow":
    # optical_flow_kwargs = {"bound": current_parameters["bounding_value"]}
    optical_flow_kwargs = {"raw_optical_flows": True}

loader = PreprocessingManagers.Loader(current_dictionary, {}, motion_vector_kwargs, optical_flow_kwargs)
train_ds = FlowDataset(
    current_dictionary,
    label_encoder,
    loader,
    train_pd,
    h5py_train,
    current_parameters["timesteps"],
    torchvision.transforms.Compose([
        FlowDataset.CropMask(crop=current_parameters["crop"], mask=current_parameters["mask"], replace_with=(0,0)),
        FlowDataset.Bound(current_parameters["bounding_value"]),
        FlowDataset.PadResize(current_model_class.resolution(), pad_with=(128,128)),
    ]),
    torchvision.transforms.Compose([
        FlowDataset.Rescale(1/255.),
        FlowDataset.BatchToCHW()
    ])
)
test_ds = FlowDataset(
    current_dictionary,
    label_encoder,
    loader,
    test_pd,
    h5py_test,
    current_parameters["timesteps"],
    torchvision.transforms.Compose([
        FlowDataset.CropMask(crop=current_parameters["crop"], mask=current_parameters["mask"], replace_with=(0,0)),
        FlowDataset.Bound(current_parameters["bounding_value"]),
        FlowDataset.PadResize(current_model_class.resolution(), pad_with=(128,128)),
    ]),
    torchvision.transforms.Compose([
        FlowDataset.Rescale(1/255.),
        FlowDataset.BatchToCHW()
    ])
)

class_weight = train_ds.get_class_weight()
class_weight = class_weight / max(class_weight)
train_frames = train_pd.groupby(current_dictionary["A"]).sum()[current_dictionary[DatasetDictionary.FRAME_COUNT_KEY]]
train_frames = train_frames.rename(index=current_mapping)
print("")
print("Train class total frames")
print(train_frames - (current_parameters["timesteps"]-1))
print("")
print("Class weight")
print(class_weight)
print("")
print("Train class total frames (balanced with weights)")
print(((train_frames - (current_parameters["timesteps"]-1))*class_weight).round().astype(int))

num_workers = (torch.get_num_threads() // 2)
train_dl = torch.utils.data.DataLoader(train_ds, current_parameters["batch_size"], num_workers=num_workers, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, current_parameters["batch_size"], num_workers=num_workers, shuffle=True)

print("")
print("Initiating Model ...")
test_input = next(iter(train_dl))
target_model: torch.nn.Module = current_model_class(class_count, current_parameters["timesteps"], current_parameters["hidden_cell_count"]).to("cuda")
torchinfo.summary(target_model, input_data=test_input[current_parameters["data_selector"]].to("cuda"))

print("")
summary_writer = torch.utils.tensorboard.writer.SummaryWriter(
    log_dir=os.path.abspath(f"./runs/{int(time.time())} - {current_parameters['data_selector']} - {current_model_class.__name__}")
)
summary_writer.add_graph(target_model, test_input[current_parameters["data_selector"]].to("cuda"))
exp, ssi, sei = torch.utils.tensorboard.writer.hparams(current_parameters, {  # type: ignore
    "Loss/Train": None,
    "FPS/Train": None,
    "Accuracy/Average/Train": None,
    "Precision/Average/Train": None,
    "Recall/Average/Train": None,
    "F1 Score/Average/Train": None,
    "Support/Average/Train": None,
    "Accuracy/Weighted Average/Train": None,
    "Precision/Weighted Average/Train": None,
    "Recall/Weighted Average/Train": None,
    "F1 Score/Weighted Average/Train": None,
    "Support/Weighted Average/Train": None,
    "Loss/Test": None,
    "FPS/Test": None,
    "Accuracy/Average/Test": None,
    "Precision/Average/Test": None,
    "Recall/Average/Test": None,
    "F1 Score/Average/Test": None,
    "Support/Average/Test": None,
    "Accuracy/Weighted Average/Test": None,
    "Precision/Weighted Average/Test": None,
    "Recall/Weighted Average/Test": None,
    "F1 Score/Weighted Average/Test": None,
    "Support/Weighted Average/Test": None,
})
summary_writer.file_writer.add_summary(exp)  # type: ignore
summary_writer.file_writer.add_summary(ssi)  # type: ignore
summary_writer.file_writer.add_summary(sei)  # type: ignore
base_path = os.path.abspath(summary_writer.get_logdir())
model_base_path = os.path.join(base_path, "model_checkpoints")
params_base_path = os.path.join(base_path, "training_parameters.json")
os.mkdir(model_base_path)
print(f"SummaryWriter created ({summary_writer.get_logdir()})")
print(f"Model checkpoint for each epoch will be created in \n\t{model_base_path}")

print("")
optimizer = torch.optim.Adam(target_model.parameters(), lr=current_parameters["learning_rate"])
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weight).to("cuda"))

actual_epoch = 0
continue_training = True
if current_parameters["epoch"] > 0:
    continue_training = False
    print(f"Running {current_parameters['epoch']} epoch")
    for _ in range(current_parameters["epoch"]):
        importables.models.default_train_one_epoch(
            target_model,
            actual_epoch,
            train_dl,
            test_dl,
            optimizer,
            criterion,
            summary_writer,
            current_parameters,
            label_encoder,
            current_mapping,
        )
        torch.save(target_model, os.path.join(model_base_path, str(actual_epoch)))
        actual_epoch += 1
        current_parameters["epoch"] = actual_epoch
        with open(params_base_path, "w") as f:
            f.write(json.encoder.JSONEncoder(indent="\t").encode(current_parameters))

    if current_parameters["epoch_ask_continue"]:
        while (continue_check := str(input(f"Current epoch ({actual_epoch}), continue? [y/n]: ")).lower().strip()) not in ["y", "n"]:
            pass
        continue_training = continue_check == "y"

while continue_training:
    importables.models.default_train_one_epoch(
        target_model,
        actual_epoch,
        train_dl,
        test_dl,
        optimizer,
        criterion,
        summary_writer,
        current_parameters,
        label_encoder,
        current_mapping,
    )
    torch.save(target_model, os.path.join(model_base_path, str(actual_epoch)))
    actual_epoch += 1
    current_parameters["epoch"] = actual_epoch
    with open(params_base_path, "w") as f:
        f.write(json.encoder.JSONEncoder(indent="\t").encode(current_parameters))

    if current_parameters["epoch_ask_continue"]:
        while (continue_check := str(input(f"Current epoch ({actual_epoch}), continue? [y/n]: ")).lower().strip()) not in ["y", "n"]:
            pass
        continue_training = continue_check == "y"

print("")
print("Training ended")
ramdisk_manager.clear()
current_parameters["epoch"] = actual_epoch
with open(params_base_path, "w") as f:
    f.write(json.encoder.JSONEncoder(indent="\t").encode(current_parameters))
summary_writer.close()
