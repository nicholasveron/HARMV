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

# test for timesteps vs lstm cells (10% data, all, 5 epoch)
timesteps = [5,10,15,20,25,30,35,40]
lstm_counts = [8,16,32,64,128,256,512,1024]
for t in timesteps:
    for l in lstm_counts:
        model_trainer_arg_wrapper(model=5, batch_size=32, epoch=5, timesteps=t, hidden_cell_count=l, data_use=0.1)

# test for mask vs crop (10% data, all, 5 epoch)
for mask in [False,True]:
    for crop in [False,True]:
        model_trainer_arg_wrapper(model=5, batch_size=32, epoch=5, hidden_cell_count=128, mask=mask, crop=crop, data_use=0.1)

# test for best bounding value (10% data, all, 5 epoch)
for bounding_value in [8,16,32,64,128]:
    model_trainer_arg_wrapper(model=5, batch_size=32, epoch=5, timesteps=30, hidden_cell_count=128, bounding_value=bounding_value, data_use=0.1)


