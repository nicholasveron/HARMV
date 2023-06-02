# encode dataset from mpeg4 to h264 using nvenc h264 codec with p4 preset, main profile and bitrate 5Mb/s

import os
import shutil
import argparse
import subprocess
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(description="""Convert whole folder(s) of video from MPEG-4 to H.264 using nvenc via ffmpeg.
WARNING: this program doesn't handle ffmpeg error, make sure ffmpeg nvenc run successfully before running this!""")

arg_dicts = [
    (
        "source_root",
        {
            "help": "Root source folder to convert"
        }
    ),
    (
        "destination_root",
        {
            "help": "Root destination folder"
        }
    ),
    (
        "--bitrate",
        {
            "default": "5M",
            "help": "Target bitrate"
        }
    )
]

_ = [parser.add_argument(arg_dict[0], **arg_dict[1]) for arg_dict in arg_dicts]

args = parser.parse_args()

print(args)

category_list = os.listdir(args.source_root)

if os.path.exists(args.destination_root):
    shutil.rmtree(args.destination_root)

os.mkdir(args.destination_root)


for category_folder in tqdm(category_list):
    category_path = os.path.join(args.source_root, category_folder)
    target_category_path = os.path.join(args.destination_root, category_folder)

    os.mkdir(target_category_path)

    video_list = os.listdir(category_path)
    for video_name in tqdm(video_list, desc=category_folder):
        video_path = os.path.join(category_path, video_name)
        target_video_path = os.path.join(args.destination_root, category_folder, video_name)

        abs_vid_path = os.path.abspath(video_path)
        abs_target_vid_path = os.path.abspath(target_video_path).split(".avi")[0] + ".mp4"

        os.system("clear")

        ffmpeg_li = [
            "ffmpeg",
            "-i", '"%s"' % abs_vid_path,
            "-vcodec", "h264_nvenc",
            "-b:v", args.bitrate,
            "-vf", '"pad=max(320\,iw):240:-1:-1, crop=min(320\,iw):240"',
            '"%s"' % abs_target_vid_path
        ]

        FFMPEG_CMD = " ".join(ffmpeg_li)

        ffmpeg_run = subprocess.run(FFMPEG_CMD, shell=True)
