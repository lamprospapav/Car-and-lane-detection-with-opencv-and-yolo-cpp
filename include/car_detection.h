#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


class car_detection
{
    
    
    
    public:
            car_detection(float confThreshold = 0.5,float nmsThreshold = 0.4, int inpWidth = 416, int inpHeight =416);
            cv::Mat getFrame(cv::Mat frame);
    private:
            std::vector<std::string> classes;    // Load names of classes
            cv::dnn::Net net;
            cv::Mat detected_frame;
            float confThreshold;  // Confidence threshold
            float nmsThreshold;  //  Non-maximum suppression threshold
            int inpWidth;       // Width of network's input image
            int inpHeight;     // Height of network's input image
            void start_process(const cv::Mat frame);
            void postprocess(cv::Mat& frame,const std::vector<cv::Mat>& outs);
            void drawPred(int classId, float conf, int left, int top, int right, int bottom,cv::Mat& frame);
            std:: vector<cv::String> getOutputNames(const cv::dnn::Net& net);

};