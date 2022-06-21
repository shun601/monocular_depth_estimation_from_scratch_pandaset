#!/usr/bin/env python3
from glob import glob
import gzip
import json
import os
from pathlib import Path
import pickle
from tqdm import tqdm
from typing import overload, List, Dict

from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
import pandas as pd
import transforms3d as t3d
         

class Pandaset(Dataset):
    def __init__(
        self,
        root_dir="./data",
        variations=["camera", "lidar"],
        cameras=["back_camera", "front_camera", "front_left_camera", "front_right_camera", "left_camera", "right_camera"],
        frame_inds=[-1, 0]):

        self.root_dir = Path(root_dir)
        scenes = glob(f"{root_dir}/**")
        self.scenes = [os.path.split(scene)[-1] for scene in scenes]
        self.variations = variations
        self.cameras = cameras
        self.frame_inds = frame_inds
        print("loading sequences")
        self.load_sequences()
        self.build_samples()

    def load_sequences(self):
        self.sequences = []
        for scene in self.scenes:
            for camera in self.cameras:
                for variation in self.variations:
                    if variation == "camera":
                        rgb_dir = self.root_dir / scene / variation / camera 
                        pose_json_path = self.root_dir / scene / variation / camera / "poses.json"
                        intrinsic_json_path = self.root_dir / scene / variation / camera / "intrinsics.json"

                        rgb_list = sorted(rgb_dir.glob("*.jpg"))

                        extrinsics = pose_to_extrinsic_array(pose_json_path, int(scene))
                        intrinsics = intrinsic_to_array(intrinsic_json_path)
                        sequence = list(zip(rgb_list, extrinsics, intrinsics))
                        self.sequences.append(sequence)
                    elif variation == "lidar":
                        pass

    def build_samples(self):
        self.samples = []
        for sequence in self.sequences:
            for i in range(len(sequence)):
                indices = [frame_idx + i for frame_idx in self.frame_inds]
                sample = []
                for idx in indices:
                    if 0 > idx or idx > len(sequence):
                        break
                    sample.append(sequence[idx])
                else:
                    self.samples.append(sample)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        item = {}
        for i, idx in enumerate(self.frame_inds):
            item.update({
                f"rgb_{idx}": self.load_image(sample[i][0]),
                f"extrinsic_{idx}": np.array(sample[i][1]),
                f"intrinsic_{idx}": np.array(sample[i][2])})
        return item

    def __len__(self):
        return len(self.samples)

    def load_image(self, path):
        return cv2.imread(str(path))[..., ::-1]

    
def load_json(json_path):
    with open(json_path, "r") as f:
        out = json.load(f)
    return out
    

def pose_to_extrinsic_array(pose_json_path, idx):
    pose = load_json(pose_json_path)[idx]
    # クォータニオン
    quat = np.array([pose["heading"]["w"],
                     pose["heading"]["x"],
                     pose["heading"]["y"],
                     pose["heading"]["z"]])
    # カメラ位置（ワールド座標）
    world_pos = np.array([pose["position"]["x"],
                          pose["position"]["y"],
                          pose["position"]["z"]])
    # Compose translations(移動), rotations, zooms, [shears] to affine
    pose_mat = t3d.affines.compose(np.array(world_pos),
                                   t3d.quaternions.quat2mat(quat), # Calculate rotation matrix corresponding to quaternion
                                   [1.0, 1.0, 1.0])
    #外部パラメータ行列
    return np.linalg.inv(pose_mat)
    


def intrinsic_to_array(intrinsic_json_path):
    intrinsic = load_json(intrinsic_json_path)
    intrinsic_array = np.array([
        [intrinsic["fx"], 0, intrinsic["cx"]],
        [0, intrinsic["fy"], intrinsic["cy"]],
        [0, 0, 1]])
    return intrinsic_array



if __name__ == '__main__':
    pass
