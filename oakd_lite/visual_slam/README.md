for ORB_SLAM2/CMakeLists.txt ORB_SLAM2/Examples/ROS/ORB_SLAM2/CMakeLists.txt ORB_SLAM2/Thirdparty/DBoW2
SET(CMAKE_CXX_STANDARD 14)
SET (OpenCV_DIR /usr/local/include/opencv4)
find_package(OpenCV QUIET)

for ORB_SLAM2/CMakeLists.txt
#find_package(Eigen3 3.1.0 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED NO_MODULE)

for ORB_SLAM2/include/ORBextractor.h
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

for ORB_SLAM2/src/Sim3Solver.cc
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>

for ORB_SLAM2/src/PnPsolver.h
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>

for ORB_SLAM2/src/PnPsolver.cc
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types_c.h>

for ORB_SLAM2/src/Tracking.cc
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgproc/types_c.h>

for ORB_SLAM2/src/Viewer.cc
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

for ORB_SLAM2/src/LocalMapping.cc
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

for ORB_SLAM2/src/LocalClosing.cc
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

for ORB_SLAM2/src/System.cc
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

for ORB_SLAM2/src/FrameDrawer.cc
#include <opencv2/imgproc/types_c.h>

for ORB_SLAM2/Examples/RGB-D/rgbd_tum.cc
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>

for ORB_SLAM2/Examples/Stereo/stereo_euroc.cc
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>

for ORB_SLAM2/Examples/Stereo/stereo_kitti.cc
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>

for ORB_SLAM2/Examples/Monocular/mono_euroc.cc
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>

for ORB_SLAM2/Examples/Monocular/mono_kitti.cc
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>

for ORB_SLAM2/Examples/Monocular/mono_tum.cc
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/imgcodecs/legacy/constants_c.h>

for ORB_SLAM2/Examples/ROS/ORB_SLAM2/src/AR/ViewerAR.cc
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake ..
cmake --build .

sudo apt install python-is-python3
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/laitathei/catkin_ws/src/ORB_SLAM2/Examples/ROS
echo $ROS_PACKAGE_PATH
pip3 install --target=/opt/ros/noetic/lib/python3/dist-packages rospkg
sudo rosdep fix-permissions
sudo rosdep init
rosdep update
./build_ros.sh
