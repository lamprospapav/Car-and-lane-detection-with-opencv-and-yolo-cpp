#include "opencv2/opencv.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <camera.h>
#include <fstream>

cameraHandler::cameraHandler(const cv::Mat frame)
{
    float x_center = frame.cols/2;
    int xfd=54,yf=460,xoffset=180;

    
    cameraHandler::src[0] = cv::Point2f(595,yf);
    cameraHandler::src[1] = cv::Point2f(683,yf);
    cameraHandler::src[2] = cv::Point2f(1110,frame.rows);
    cameraHandler::src[3] = cv::Point2f(220,frame.rows);

    cameraHandler::dst[0] = cv::Point2f(src[3].x + xoffset,0);
    cameraHandler::dst[1] = cv::Point2f(src[2].x - xoffset,0);
    cameraHandler::dst[2] = cv::Point2f(src[2].x - xoffset,src[2].y);
    cameraHandler::dst[3] = cv::Point2f(src[3].x + xoffset,src[3].y);
        

    /*
    cameraHandler::src[0] = cv::Point2f(620,464);
    cameraHandler::src[1] = cv::Point2f(707,464);
    cameraHandler::src[2] = cv::Point2f(258,682);
    cameraHandler::src[3] = cv::Point2f(900,682);

    cameraHandler::dst[0] = cv::Point2f(450,0);
    cameraHandler::dst[1] = cv::Point2f(frame.cols-450,0);
    cameraHandler::dst[2] = cv::Point2f(450,frame.rows);
    cameraHandler::dst[3] = cv::Point2f(frame.cols-450,frame.rows);
    */


    cv::FileStorage fs("calibration.xml",cv::FileStorage::READ);
    if (fs.isOpened())
    {
        std::cout<<"Calibration file exist:"<<std::endl;
        std::cout<<"#######################"<<std::endl;
        cv::FileStorage fs("calibration.xml",cv::FileStorage::READ);
        fs["mtx"] >> cameraHandler::mtx;
        fs["dist"] >> cameraHandler::dist;
        fs.release();
    }
    else 
    {
        std::cerr << "Calibration file (not existing or failed to open)\n";
        std::cout<<"Calibration Started"<<std::endl;
        this->calibrate();
        cv::FileStorage fs("calibration.xml",cv::FileStorage::READ);
        fs["mtx"] >> cameraHandler::mtx;
        fs["dist"] >> cameraHandler::dist;
        fs.release();
    }

}

void cameraHandler::calibrate()
{
    
        std::vector< std::vector< cv::Point3f > > object_points;
        std::vector<cv::String> fn;
        cv::glob("/camera_cal/*.jpg",fn,false);
        std::vector<cv::Point2f> corners;
        std::vector<cv::Point3f> obj;
	    for(int j=0; j<6; ++j)
            for(int i=0;i<9;++i)
		        obj.push_back(cv::Point3f(i,j,0));
                //std::cout<<obj;		

        size_t count = fn.size();
        std::vector< std::vector< cv::Point2f > > imgpoints;

        for(size_t i=0; i<count;i++)
        {
            cv::Mat img = cv::imread(fn[i]);
            cv::cvtColor(img,img,cv::COLOR_BGR2GRAY);//source image
            bool patternfound = cv::findChessboardCorners(img,cv::Size(9,6),corners); 
            if(patternfound)
            {
                object_points.push_back(obj);
                cv::cornerSubPix(img,corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));

                imgpoints.push_back(cv::Mat(corners));
            //cv::drawChessboardCorners(img,cv::Size(9,6),cv::Mat(corners),patternfound);
            //cv::imshow("result",img);
            //cv::waitKey(0);
            }

        
        }
        cv::Mat img = cv::imread(fn[1]);
        cv::Mat rvecs,tvecs;
        cv::calibrateCamera(object_points,imgpoints,cv::Size(img.cols,img.rows),cameraHandler::mtx,cameraHandler::dist,rvecs,tvecs);
    
        std::cout<<"Calibrate done"<<std::endl;
        cv::FileStorage fs("calibration.xml",cv::FileStorage::WRITE);
        fs<<"mtx"<<cameraHandler::mtx;
        fs<<"dist"<<cameraHandler::dist;
        fs.release();
}


cv::Mat cameraHandler::warp(const cv::Mat frame)
{
    
    cv::Mat M = cv::getPerspectiveTransform(cameraHandler::src,cameraHandler::dst);
    //cv::Mat Minv =cv::getPerspectiveTransform(cameraHandler::dst,cameraHandler::src);
    cv::Mat warped; 
    cv::warpPerspective(frame,warped,M,cv::Size(frame.cols,frame.rows),cv::INTER_LINEAR);
    return warped;

}

cv:: Mat cameraHandler::unwarp(const cv::Mat frame)
{
    cv::Mat Minv =cv::getPerspectiveTransform(cameraHandler::dst,cameraHandler::src);
    cv::Mat unwarped; 
    cv::warpPerspective(frame,unwarped,Minv,cv::Size(frame.cols,frame.rows),cv::INTER_LINEAR);
    return unwarped;
}

cv:: Mat cameraHandler::get_unwarped_img(const cv::Mat frame)
{
    cv::Mat unwarped =cameraHandler::unwarp(frame);
    return unwarped;
}

cv::Mat cameraHandler::get_warped_img(const cv::Mat frame)
{
    cv::Mat warped =  cameraHandler::warp(frame);
    return warped;
}
