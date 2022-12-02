
#######################
## ROS2 dependencies ##
#######################
sudo apt-get install -y ros-foxy-cv-bridge ros-foxy-librealsense2 ros-foxy-message-filters ros-foxy-image-transport
sudo apt-get install -y libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev
sudo apt-get install -y libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev

###################
## Realsense SDK ##
###################
# Register the server's public key:
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE

# Add the server to the list of repositories:
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u

# Install the libraries (see section below if upgrading packages):
sudo apt-get -y install librealsense2-utils

# Optionally install the developer and debug packages:
sudo apt-get install -y librealsense2-dev && sudo apt-get install -y librealsense2-dbg


####################
## ROS2 realsense ##
####################
cd ~/ros/catkin_ws/src
git clone --depth 1 --branch `git ls-remote --tags https://github.com/IntelRealSense/realsense-ros.git | grep -Po "(?<=tags/)3.\d+\.\d+" | sort -V | tail -1` https://github.com/IntelRealSense/realsense-ros.git
cd ..

## install dependencies
sudo apt-get install python3-rosdep -y
sudo rosdep init # "sudo rosdep init --include-eol-distros" for Dashing
rosdep update
rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y

# build
colcon build --symlink-install

