import torch
from glob import glob
from tqdm import tqdm
import json
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_file", default="./data/splits/all_data_test.json", type=str)
parser.add_argument("--feature_folder",
                    default="./data/eva_clip_features/", type=str)

args = parser.parse_args()

durations = {}

with open(args.data_file, 'r') as f:
    prompt2video_anns = json.load(f)
    for i, (prompt, video_anns) in enumerate(prompt2video_anns.items()):
        has_relevant_videos = False
        for video_fname, video_ann in video_anns.items():
            if not video_fname.endswith("mp4"):
                continue
            #print(video_fname)
            video_duration = video_ann['v_duration']
            video_duration = round(video_duration)
            durations[video_fname] = video_duration


feature_files = glob(f"{args.feature_folder}/*.pt")
cnt = 0
fixed_cnt = 0
new_dir_path = args.feature_folder + "_corrected"

if not os.path.exists(new_dir_path):
    os.makedirs(new_dir_path)

for f in tqdm(feature_files):
    name = f.split("/")[-1].replace(".pt", "")
    if name not in durations:
        continue

    features = torch.load(f, map_location='cpu')
    if not torch.is_tensor(features):
        cnt +=1
        continue
    if name in durations and features.shape[0] != durations[name]:
        fixed_cnt += 1
        features = features[0:durations[name]]
    torch.save(features, os.path.join(new_dir_path, name + ".pt"))
print(cnt)
