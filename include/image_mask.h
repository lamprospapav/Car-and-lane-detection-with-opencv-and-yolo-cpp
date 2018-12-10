#include "opencv2/opencv.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>


class image_mask{
    public:
        image_mask();
        cv::Mat masked_image;
        cv::Mat get_hls_mask(const cv::Mat frame,int low_thresh = 180,int high_thresh = 255);
        cv::Mat get_x_thresh_mask(const cv::Mat frame,int low_mag = 5,int high_mag = 255);
        cv::Mat get_mag_thresh_mask(const cv::Mat frame,int low_mag = 10,int high_mag = 255);
        cv::Mat get_color_mask(const cv::Mat frame,cv::Scalar whiteMinScalar = cv::Scalar(0 , 200 , 0),cv::Scalar whiteMaxScalar = cv::Scalar(255, 255 ,255 ),cv::Scalar yellowMinScalar = cv::Scalar(10 , 0 , 100),cv::Scalar yellowMaxScalar = cv::Scalar(40, 255, 255));
        //cv::Mat get_dir_threshold(const cv::Mat frame,float low_angle = 0.0,float high_angle = 0.09);
    private:
        // Convert BGR Image to HLS and split to HSL_Channels
        cv::Mat hls_select(const cv::Mat frame,int low_thresh,int high_thresh);

        // Convert BGR2GRAY and use SOBEL MASK
        // Return gradangle in specific Range
        
        //cv::Mat  dir_threshold(const cv::Mat frame,float low_angle,float high_angle);
        
        // Convert BGR2GRAY and use SOBEL MASK
        // Calculate gradmag in specific Range
        cv::Mat mag_thresh(const cv::Mat frame,int low_mag,int high_mag);
        
        // Convert BGR2GRAY and use SOBEL MASK
        // Calculate sobelx_abs in specific Range

        cv::Mat x_thresh(const cv::Mat frame,int low_mag,int high_mag);

        // Convert BGR2GRAY and use SOBEL MASK
        // Calculate sobelx_abs in specific Range
        cv::Mat color_mask(const cv::Mat frame,cv::Scalar whiteMinScalar,cv::Scalar whiteMaxScalar,cv::Scalar yellowMinScalar,cv::Scalar yellowMaxScalar);


};