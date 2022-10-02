- Install dependency
```
sudo apt-get install cmake
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libeigen3-dev
sudo apt-get install libsuitesparse-dev
```

- Download [latest stable release Ceres Solver](http://ceres-solver.org/installation.html)
```
wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
tar zxf ceres-solver-2.1.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake ../ceres-solver-2.0.0
make -j3
make test
sudo make install
```

- Build Vins-Fusion with ROS
```
cd ~/catkin_ws/src
git clone https://github.com/HKUST-Aerial-Robotics/VINS-Fusion.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

- oakd_lite example
```
roscore
python3 oakd_node.py

```
