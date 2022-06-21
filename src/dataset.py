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

from .sequence import Sequence
from .utils import subdirectories


class DataSet:
    """Top-level class to load PandaSet
        ``DataSet`` prepares and loads ``Sequence`` objects for every sequence found in provided directory.
        Access to a specific sequence is provided by using the sequence name as a key on the ``DataSet`` object.
        Args:
             directory: Absolute or relative path where PandaSet has been extracted to.
        Examples:
            >>> pandaset = DataSet('/data/pandaset')
            >>> s = pandaset['002']
        """

    def __init__(self, directory: str) -> None:
        self._directory: str = directory
        self._sequences: Dict[str, Sequence] = None
        self._load_sequences()

    def __getitem__(self, item) -> Sequence:
        return self._sequences[item]

    def _load_sequences(self) -> None:
        self._sequences = {}
        sequence_directories = subdirectories(self._directory)
        for sd in sequence_directories:
            seq_id = sd.split('/')[-1].split('\\')[-1]
            self._sequences[seq_id] = Sequence(sd)

    def sequences(self, with_semseg: bool = False) -> List[str]:
        """ Lists all available sequence names
        Args:
            with_semseg: Set `True` if only sequences with semantic segmentation annotations should be returned. Set `False` to return all sequences (with or without semantic segmentation).
        Returns:
            List of sequence names.
        Examples:
            >>> pandaset = DataSet('/data/pandaset')
            >>> print(pandaset.sequences())
            ['002','004','080']
        """
        if with_semseg:
            return [s for s in list(self._sequences.keys()) if self._sequences[s].semseg]
        else:
            return list(self._sequences.keys())

    def unload(self, sequence: str):
        """ Removes all sequence file data from memory if previously loaded from disk.
        This is useful if you intend to iterate over all sequences and perform some
        operation. If you do not unload the sequences, it quickly leads to sigkill.
        Args:
            sequence: The sequence name
        Returns:
            None
        Examples:
            >>> pandaset = DataSet('...')
            >>> for sequence in pandaset.sequences():
            >>>     seq = pandaset[sequence]
            >>>     seq.load()
            >>>     # do operations on sequence here...
            >>>     # when finished, unload the sequence from memory
            >>>     pandaset.unload(sequence)
        """
        if sequence in self._sequences:
            del self._sequences[sequence]
            

class Pandaset(Dataset):
    def __init__(
        self,
        root_dir="./data/",
        variations=["camera", "lidar"]
        cameras=["back_camera", "front_camera", "front_left_camera", "front_right_camera", "left_camera", "right_camera"],
        frame_inds=[-1, 0]):

        self.root_dir = Path(root_dir)
        scenes = glob(f"{root_dir}/**")        
        self.scenes = [os.path.split(scenes)[-1] for scene in scenes]
        self.variations = variations
        self.cameras = cameras
        self.frame_inds = frame_inds
        print("loading sequences")
        self.load_sequences()
        self.build_samples()

    def load_sequences(self):
        self.sequences = []
        for idx in range()
            for camera in self.cameras:
                for variation in self.variations:
                    if variation == "camera":
                        rgb_dir = self.root_dir / scene / variation / camera 
                        pose_json_path = self.root_dir / scene / variation / camera / "poses.json"
                        intrinsic_json_path = self.root_dir / scene / variation / camera / "intrinsics.json"

                        rgb_list = sorted(rgb_dir.glob("*.jpg"))

                        extrinsics = pose_to_extrinsic_array(pose_json, int(scene))
                        intrinsice = intrinsic_to_array(intrinsic_json_path)
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
