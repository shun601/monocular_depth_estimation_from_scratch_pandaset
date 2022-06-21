#!/usr/bin/env python3
from typing import Dict

from .sensors import Camera
from .sensors import Lidar
from .utils import subdirectories


class Sequence:
    """Provides all sensor and annotations for a single sequence.

    ``Sequence`` provides generic preparation and loading methods for a single PandaSet sequence folder structure.

    Args:
         directory: Absolute or relative path where annotation files are stored
    """

    @property
    def lidar(self) -> Lidar:
        """ Stores ``Lidar`` object for sequence

        Returns:
            Instance of ``Lidar`` class.
        """
        return self._lidar

    @property
    def camera(self) -> Dict[str, Camera]:
        """ Stores all ``Camera`` objects for sequence.

        Access data by entering the key of a specific camera (see example).

        Returns:
            Dictionary of all cameras available for sequence.

        Examples:
            >>> print(s.camera.keys())
                dict_keys(['front_camera', 'left_camera', 'back_camera', 'right_camera', 'front_left_camera', 'front_right_camera'])
            >>> cam_front = s.camera['front_camera']
        """
        return self._camera

    def __init__(self, directory: str) -> None:
        self._directory: str = directory
        self._lidar: Lidar = None
        self._camera: Dict[str, Camera] = None
        self._load_data_structure()

    def _load_data_structure(self) -> None:
        data_directories = subdirectories(self._directory)

        for dd in data_directories:
            if dd.endswith('lidar'):
                self._lidar = Lidar(dd)
            elif dd.endswith('camera'):
                self._camera = {}
                camera_directories = subdirectories(dd)
                for cd in camera_directories:
                    camera_name = cd.split('/')[-1].split('\\')[-1]
                    self._camera[camera_name] = Camera(cd)
            
    def load(self) -> 'Sequence':
        """Loads all sequence files from disk into memory.

        All sequence files are loaded into memory, including sensor, meta and annotation data.

        Returns:
            Current instance of ``Sequence``
        """
        self.load_lidar()
        self.load_camera()
        return self

    def load_lidar(self) -> 'Sequence':
        """Loads all LiDAR files from disk into memory.

        All LiDAR point cloud files are loaded into memory.

        Returns:
            Current instance of ``Sequence``
        """
        self._lidar.load()
        return self

    def load_camera(self) -> 'Sequence':
        """Loads all camera files from disk into memory.

        All camera image files are loaded into memory.

        Returns:
            Current instance of ``Sequence``
        """
        for cam in self._camera.values():
            cam.load()
        return self

    

if __name__ == '__main__':
    pass
