#include "opencv2/opencv.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <image_mask.h>
#include <camera.h>
#include <lane_detection.h>
#include <ShowManyImages.h>
#include <car_detection.h>


int main(int argc, char const *argv[])
{
    cv::VideoCapture cap("egnatia.mp4");
    if(!cap.isOpened())
    {
        std::cout << "Error opening video stream of file" << std::endl;
        return -1;
    }
    cv::Mat frame;
    cap >>frame;
    cameraHandler camera(frame);
    lane_detection lane;
    image_mask mask;
    car_detection car_obj;
    for(;;)
    {
        if(frame.empty()){break;}
       
        
    
        cv::Mat x_threshed,mag_threshed,hls_selected,img,color_filtered,dir_thresholded,und_frame;
        cv::undistort(frame,und_frame,camera.mtx,camera.dist);
        cv::GaussianBlur( und_frame, frame, cv::Size(5,5),0);

       
        x_threshed = mask.get_x_thresh_mask(frame);
        hls_selected = mask.get_hls_mask(frame);
        mag_threshed = mask.get_mag_thresh_mask(frame);
        color_filtered = mask.get_color_mask(frame);
        //Bitwise maskes to one masked image
        cv::bitwise_and(hls_selected,x_threshed,img);
        cv::bitwise_and(mag_threshed,img,img);
        cv::bitwise_and(img,color_filtered,img);
        cv::Mat warped = camera.get_warped_img(img);
        lane.set_slide_search(warped);
        if(lane.left_fitx.rows ==0)
        {   cap>>frame;
            continue;
        }
        cv::Mat lane_img;
        lane_img = lane.get_lane(frame);
        cv::Mat unwarped_lane;
        unwarped_lane = camera.get_unwarped_img(lane_img);
        cv::Mat detected_lane;
        cv::addWeighted(frame,1,unwarped_lane,0.3,0,detected_lane);

        cv::Mat lane_plot;
        //lane_plot = ShowManyImages("Image", 3, detected_lane,frame,warped);
        //lane_plot = ShowManyImages("Image", 2, warped,frame,img);

        //cv::imshow("Result",lane_plot);
        
        //cv::imshow("Car",im_car);
        cv::Mat im_car;
        im_car = car_obj.getFrame(detected_lane);
        
        cv::imshow("Final_result",im_car);
        cv::namedWindow("Img",cv::WINDOW_NORMAL);
        cv::resizeWindow("Img",600,480);
        cv::imshow("Img",frame);
        cv::namedWindow("Masked",cv::WINDOW_NORMAL);
        cv::resizeWindow("Masked",600,480);
        cv::imshow("Masked",img);
        cv::namedWindow("Bird View",cv::WINDOW_NORMAL);
        cv::resizeWindow("Bird View",600,480);
        cv::imshow("Bird View",warped);
        
        char c=(char) cv::waitKey(25);
        if(c==27){break;}
        
        cap>>frame;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;


}

