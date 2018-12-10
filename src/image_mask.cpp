#include "opencv2/opencv.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <image_mask.h>


image_mask::image_mask()
{}

cv::Mat image_mask::hls_select(const cv::Mat frame,int low_thresh,int high_thresh)
{
    cv::Mat hsl,HSL_channels[3];
    cv::cvtColor(frame,hsl,cv::COLOR_RGB2HLS);
    cv::split(frame,HSL_channels);
    cv::inRange(HSL_channels[2],low_thresh,high_thresh,HSL_channels[2]);
    return HSL_channels[2];

}

cv::Mat image_mask::get_hls_mask(const cv::Mat frame,int low_thresh,int high_thresh)
{
    image_mask::masked_image = image_mask::hls_select(frame,low_thresh,high_thresh);
    return image_mask::masked_image;
}

/*
cv::Mat image_mask::dir_threshold(const cv::Mat frame,float low_angle,float high_angle)
{
    cv:: Mat frame_gray;
    cv::Sobel(f)
    cv::cvtColor( frame, frame_gray, CV_BGR2GRAY );
    cv::Mat sobelx,sobelx_abs,sobely,sobely_abs;
    cv::Sobel(frame_gray,sobelx,CV_64F,1,0,15);
    cv::Sobel(frame_gray,sobely,CV_64F,0,1,15);
    cv:: Mat gradmag,gradangle;
    cv::cartToPolar(cv::abs(sobely),cv::abs(sobelx),gradmag,gradangle);
    //cv::convertScaleAbs(gradangle,gradangle);
    cv::inRange(gradangle,low_angle,high_angle,gradangle);
    return gradangle;

}
cv::Mat image_mask::get_dir_threshold(const cv::Mat frame,float low_angle,float high_angle)
{
    image_mask::masked_image = image_mask::dir_threshold(frame,low_angle,high_angle);
    return image_mask::masked_image;
}

*/
cv::Mat image_mask::mag_thresh(const cv::Mat frame,int low_mag,int high_mag)
{
    cv:: Mat frame_gray;
    cv::cvtColor( frame, frame_gray, CV_BGR2GRAY );
    cv::Mat sobelx,sobelx_abs,sobely,sobely_abs;
    cv::Sobel(frame_gray,sobelx,CV_64F,1,0,3);
    //cv::convertScaleAbs(sobelx,sobelx_abs);
    cv::Sobel(frame_gray,sobely,CV_64F,0,1,3);
    //cv::convertScaleAbs(sobely,sobely_abs);
    //cv::inRange(sobelx_abs,22,100,sobelx_abs);
    cv:: Mat gradmag,gradangle;
    cv::cartToPolar(sobelx,sobely,gradmag,gradangle);
    cv::convertScaleAbs(gradmag,gradmag);
    cv::inRange(gradmag,low_mag,high_mag,gradmag);
    return gradmag;
}

cv::Mat image_mask::get_mag_thresh_mask(const cv::Mat frame,int low_mag,int high_mag)
{
    image_mask::masked_image = image_mask::mag_thresh(frame,low_mag,high_mag);
    return image_mask::masked_image;
}

cv::Mat image_mask::x_thresh(const cv::Mat frame,int low_mag,int high_mag)
{
    cv:: Mat frame_gray;
    cv::cvtColor( frame, frame_gray, CV_BGR2GRAY );
    cv::Mat sobelx,sobelx_abs,HSL_channels[3];   
    cv::Sobel(frame_gray,sobelx,CV_64F,1,0,3);
    cv::convertScaleAbs(sobelx,sobelx_abs);
    cv::inRange(sobelx_abs,low_mag,high_mag,sobelx_abs);

    return sobelx_abs;
}


cv::Mat image_mask::get_x_thresh_mask(const cv::Mat frame,int low_mag,int high_mag)
{
    image_mask::masked_image = image_mask::x_thresh(frame,low_mag,high_mag);
    return image_mask::masked_image;
}

cv::Mat image_mask::color_mask(const cv::Mat frame,cv::Scalar whiteMinScalar,cv::Scalar whiteMaxScalar,cv::Scalar yellowMinScalar,cv::Scalar yellowMaxScalar)
{
    cv::Mat img_HLS;
    cv::cvtColor(frame,img_HLS,cv::COLOR_BGR2HLS);
    // White mask    
    cv::Mat WhiteMask;
    cv::inRange(img_HLS, whiteMinScalar, whiteMaxScalar,WhiteMask);

    //yellow color thresholding
    cv::Mat YellowMask;
    cv::inRange(img_HLS, yellowMinScalar, yellowMaxScalar,YellowMask);

    cv::Mat mask;
    cv::bitwise_or(WhiteMask,YellowMask,mask);
    return mask;
}

cv::Mat image_mask::get_color_mask(const cv::Mat frame,cv::Scalar whiteMinScalar,cv::Scalar whiteMaxScalar,cv::Scalar yellowMinScalar,cv::Scalar yellowMaxScalar)
{
    image_mask::masked_image = image_mask::color_mask(frame,whiteMinScalar,whiteMaxScalar,yellowMinScalar,yellowMaxScalar);
    return masked_image;
}
































