#include "opencv2/opencv.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>


class cameraHandler
{
    public:
        cameraHandler(const cv::Mat frame);
        cv::Point2f src[4];
        cv::Point2f dst[4];
        cv::Mat mtx;
        cv::Mat dist;
        cv::Mat get_warped_img(const cv::Mat frame);
        cv::Mat get_unwarped_img(const cv::Mat frame);
    private:
        // Calibrate car camera
        void calibrate();
        // Warp Perspective transform to get Bird View area
        cv::Mat warp(const cv::Mat frame);
        // UnWarp Perspective transform
        cv::Mat unwarp(const cv::Mat frame);
};