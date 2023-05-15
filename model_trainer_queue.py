import os
import subprocess
from importables.custom_types import Union

def model_trainer_arg_wrapper(
    model: Union[int, None] = None,
    batch_size: Union[int, None] = None,
    timesteps: Union[int, None] = None,
    hidden_cell_count: Union[int, None] = None,
    learning_rate: Union[float, None] = None,
    epoch: Union[int, None] = None,
    epoch_ask_continue: Union[bool, None] = None,
    dataset: Union[int, None] = None,
    split: Union[int, None] = None,
    data_use: Union[float, None] = None,
    data_selector: Union[str, None] = None,
    train_in_memory: Union[bool, None] = None,
    test_in_memory: Union[bool, None] = None,
    mask: Union[bool, None] = None,
    crop: Union[bool, None] = None,
    bounding_value: Union[float, None] = None,
    run_comment: Union[str, None] = None,
):
    arg_list: list[str] = ["python model_trainer.py --use_cmd_args"]
    for k,v in locals().items():
        if k == "arg_list":
            continue
        if v is not None:
            arg_list.append(f"--{k}={v}")
    print("")
    print("Running: ")
    print(*arg_list, sep="\t\n")
    print("")
    subprocess.run(
        shell=True,
        args=" ".join(arg_list)
    )

# test for best cnn head (after testing which model is fastest and best)
for model_index in [1,3,5]:
    model_trainer_arg_wrapper(model=model_index, batch_size=32, epoch=5, hidden_cell_count=128)


# DEFAULT_EPOCH = 5
# DEFAULT_DATA_USE = 1

# # LIST_MODEL = [5, 4, 2, 3]
# # LIST_MODEL = [0,1,2,3,4,5]
# LIST_MODEL = [1]
# LIST_HIDDEN_COUNT = [128]
# LIST_TIMESTEPS = [30]
# LIST_MASK = [True]
# LIST_CROP = [True]
# LIST_BOUNDING_VALUE = [32]
# LIST_DATA_SELECTOR = ["motion_vector"]

# for ds in LIST_DATA_SELECTOR:
#     for ma in LIST_MASK:
#         for cr in LIST_CROP:
#             for bv in LIST_BOUNDING_VALUE:
#                 for ts in LIST_TIMESTEPS:
#                     for hc in LIST_HIDDEN_COUNT:
#                         for mo in LIST_MODEL:
#                             print("Running arguments: ", 
#                                   f"\n\tData Selector: {ds}",
#                                   f"\n\tMask: {ma}",
#                                   f"\n\tCrop: {cr}",
#                                   f"\n\tBounding Value: {bv}",
#                                   f"\n\tTimesteps: {ts}",
#                                   f"\n\tHidden Cell Count: {hc}",
#                                   f"\n\tModel: {mo}"
#                                   )
#                             subprocess.run(
#                                 shell=True,
#                                 args=f"python model_trainer.py --use_cmd_args --epoch={DEFAULT_EPOCH} --data_use={DEFAULT_DATA_USE} --data_selector='{ds}' --model={mo} --hidden_cell_count={hc} --timesteps={ts} --mask={ma} --crop={cr} --bounding_value={bv}"
#                             )
