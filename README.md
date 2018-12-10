# Car and lane detection with opencv and yolov3 c++

This is a method that detects cars and lane on road using c++ opencv and yolov3.
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### System version that I am using
opencv => 2.4.9

gcc => 5.4

yolov3

### Hardware
Intel® Core™ i5-7500 CPU @ 3.40GHz × 4 

8gb Ram

### Installing
git clone https://github.com/lamprospapav/Car-and-lane-detection-with-opencv-and-yolo-cpp.git
cd Car-and-lane-detection-with-opencv-and-yolo-cpp/

For yolo detection you need download yolo3.weights from wget https://pjreddie.com/media/files/yolov3.weights and copy it at YoloNet folder. In the next version i will train my own weights for car detection.

mkdir build

cd build

cmake ..

make

cd ..

cmake .

make

./lane_detection

## Running the tests

For your camera you need to calibrate it using your chessbord images into camera_cal directory.

![alt text](https://github.com/lamprospapav/Car-and-lane-detection-with-opencv-and-yolo-cpp/blob/master/car_lane_detection.png)

I am running code with cpu and I need 80 ms for a frame. input image yolo is 224 x 224.
In next version i will use darknet and gpu.

