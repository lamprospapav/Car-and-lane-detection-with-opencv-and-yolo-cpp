#include "opencv2/opencv.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>


class lane_detection
{
    public:
        lane_detection();
        cv::Mat left_fitx;
        cv::Mat right_fitx;
        std::vector<double> ploty;
        cv::Mat lane;
        cv::Mat get_lane(cv::Mat frame);
        void polyfit(cv::Mat src_x,cv::Mat src_y,cv::Mat& dst,int order);
        template <typename T>  std::vector<double> linspace(T start_in, T end_in, int num_in);
        void set_slide_search(const cv::Mat warped);
    private:
        void slide_search(const cv::Mat img);
        cv::Mat draw_lanes(cv::Mat left_fitx,cv::Mat right_fitx,std::vector<double> ploty,const cv::Mat img);
        
};