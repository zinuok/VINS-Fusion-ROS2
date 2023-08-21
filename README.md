# VINS-Fusion

## ROS2 version of VINS-Fusion.

### Notices
- code has been updated so that the vins package can be executed via ros2 run or ros2 launch
- but Rviz config cannot be saved due to some issue.. still fixing
- GPU enable/disable features also have been added: refer [EuRoC config](https://github.com/zinuok/VINS-Fusion-ROS2/blob/main/config/euroc/euroc_stereo_imu_config.yaml#L19-L21) (refered from [here](https://github.com/pjrambo/VINS-Fusion-gpu) and [here](https://github.com/pjrambo/VINS-Fusion-gpu/issues/33#issuecomment-1097642597))
  - The GPU version has some CUDA library [dependencies: OpenCV with CUDA](https://github.com/zinuok/VINS-Fusion-ROS2/blob/main/vins/src/featureTracker/feature_tracker.h#L21-L23). Therefore, if it is a bothersome to you and only need the cpu version, please comment the following compiler macro at line 14 in the 'feature_tracker.h': .
  ```bash
  #define GPU_MODE 1
  ```

### Prerequisites
- **System**
  - Ubuntu 20.04
  - ROS2 foxy
- **Libraries**
  - OpenCV 3.4.1 (with CUDA enabled option)
  - OpenCV 3.4.1-contrib
  - [Ceres Solver-2.1.0](http://ceres-solver.org/installation.html) (you can refer [here](https://github.com/zinuok/VINS-Fusion#-ceres-solver-1); just edit 1.14.0 to 2.1.0 for install.)
  - [Eigen-3.3.9](https://github.com/zinuok/VINS-Fusion#-eigen-1)


### sensor setup
- camera: Intel realsense D435i
- using following shell script, you can install realsense SDK with ROS2 package.
```bash
chmod +x realsense_install.sh
bash realsense_install.sh
```


### build
```bash
cd $(PATH_TO_YOUR_ROS2_WS)/src
git clone https://github.com/zinuok/VINS-Fusion-ROS2
cd ..
colcon build --symlink-install && source ./install/setup.bash && source ./install/local_setup.bash
```

### run
```bash
# vins
ros2 run vins $(PATH_TO_YOUR_VINS_CONFIG_FILE)

# Rviz2 visualization
ros2 launch vins vins_rviz.launch.xml
```


## play bag recorded at ROS1
Unfortunately, you can't just play back the bag file recorded at ROS1. 
This is because the filesystem structure for bag file has been changed significantly.
The bag file at ROS2 needs the folder with some meta data for each bag file, which is done using following commands.
- you have to install [this pkg](https://gitlab.com/ternaris/rosbags)
```bash
pip install rosbags
```

- run
```bash
export PATH=$PATH:~/.local/bin
rosbags-convert foo.bag --dst /path/to/bar
```






## Original Readme:

## 8. Acknowledgements
We use [ceres solver](http://ceres-solver.org/) for non-linear optimization and [DBoW2](https://github.com/dorian3d/DBoW2) for loop detection, a generic [camera model](https://github.com/hengli/camodocal) and [GeographicLib](https://geographiclib.sourceforge.io/).

## 9. License
The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.

We are still working on improving the code reliability. For any technical issues, please contact Tong Qin <qintonguavATgmail.com>.

For commercial inquiries, please contact Shaojie Shen <eeshaojieATust.hk>.
