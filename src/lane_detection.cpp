#include "opencv2/opencv.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <lane_detection.h>


lane_detection::lane_detection(){}


void lane_detection::slide_search(const cv::Mat img)
{
    cv::Mat bottom_half_height_left = img(cv::Range(img.rows / 2 ,img.rows), cv::Range(0, img.cols/2));
    cv::Mat bottom_half_height_right = img(cv::Range(img.rows / 2 ,img.rows), cv::Range(img.cols/2 ,img.cols ));
    //std::cout << "Data  = " << std::endl << cv::format(bottom_half_height.col(478), cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    cv::Mat hist_left(bottom_half_height_left.cols,1,CV_32S);
    cv::Mat hist_right(bottom_half_height_right.cols,1,CV_32S);
    for(int i=0;i<bottom_half_height_left.cols;i++)
    {
        hist_left.at<int>(0,i)=cv::countNonZero(bottom_half_height_left.col(i));
        hist_right.at<int>(0,i)=cv::countNonZero(bottom_half_height_right.col(i));

    }
    cv::Point max_loc_left,max_loc_right;
    cv::minMaxLoc(hist_left, 0, 0, 0, &max_loc_left);
    cv::minMaxLoc(hist_right, 0, 0, 0, &max_loc_right);

    cv::Mat out_img;
    cv::cvtColor(img,out_img,CV_GRAY2BGR);
   // std::cout << "Width : " << out_img.cols << std::endl;
   // std::cout << "Height: " << out_img.rows << std::endl;
   // std::cout << "Channels: " << out_img.channels() << std::endl;

    cv::Mat nonZeroCoordinates;
    cv::findNonZero(img, nonZeroCoordinates);
    cv::Mat nonzerosx,nonzerosy;
    for (int i=0;i<nonZeroCoordinates.total();i++){
        nonzerosx.push_back(nonZeroCoordinates.at<cv::Point>(i).x);
        nonzerosy.push_back(nonZeroCoordinates.at<cv::Point>(i).y);
        //std::cout << "Zero#" << i << ": " << nonZeroCoordinates.at<cv::Point>(i).x << ", " << nonZeroCoordinates.at<cv::Point>(i).y << std::endl;

    }
    //std::cout<<nonzerosx;
    int nwindows = 9;
    double window_height = int(img.rows/nwindows);

    //Identify the x and y positions of all nonzero pixels in the image
    

    //  Current positions to be updated for each window

    int leftx_current = max_loc_left.y;
    int rightx_current =img.cols/2 + max_loc_right.y;
    
    // Set the width of the windows +/- margin
    int margin = 100;
    //Set minimum number of pixels found to recenter window
    int minpix = 50;
    std::vector<int> good_left_inds,good_right_inds;
    cv::Mat left_lane_inds,right_lane_inds;

    for(int window=0; window<nwindows; window++)
    {
        double win_y_low = img.rows - (window + 1)*window_height;
        double win_y_high = img.rows - window*window_height;
        double win_xleft_low = leftx_current - margin;
        double win_xleft_high = leftx_current + margin;
        double win_xright_low = rightx_current - margin;
        double win_xright_high = rightx_current + margin;


        //Draw the windows on the visualization image
        cv::rectangle(out_img,cv::Point(win_xleft_low,win_y_low),cv::Point(win_xleft_high,win_y_high),cv::Scalar(0,255,0),2);
        cv::rectangle(out_img,cv::Point(win_xright_low,win_y_low),cv::Point(win_xright_high,win_y_high),cv::Scalar(0,255,0),2);
        //cv::imshow("out image",out_img);
        //cv::waitKey(0);
        // Identify the nonzero pixels in x and y within the window
        
        for (int i=0;i<nonzerosx.rows;i++){
            if((nonzerosy.at<int>(i) >= win_y_low) & (nonzerosy.at<int>(i) < win_y_high) & 
            (nonzerosx.at<int>(i) >= win_xleft_low) & (nonzerosx.at<int>(i) < win_xleft_high)){
                good_left_inds.push_back(i);
            }
            if((nonzerosy.at<int>(i) >= win_y_low) & (nonzerosy.at<int>(i) < win_y_high) & 
            (nonzerosx.at<int>(i) >= win_xright_low) & (nonzerosx.at<int>(i) < win_xright_high)){
                good_right_inds.push_back(i);
            }
        

        }
        
         
        left_lane_inds.push_back(good_left_inds);
        right_lane_inds.push_back(good_right_inds);


        if(good_left_inds.size() > minpix)
        {
            int temp=0;
            for(int i:good_left_inds)
            {
                temp += nonzerosx.at<int>(i);
            }
            leftx_current = temp/good_left_inds.size();
        }
       

        if(good_right_inds.size() > minpix)
        {
            int temp=0;
            for(int i:good_right_inds)
            {
                temp += nonzerosx.at<int>(i);
            }
            rightx_current = temp/good_right_inds.size();
        }

        good_left_inds.clear();
        good_right_inds.clear();


    }
   cv::Mat leftx,lefty,rightx,righty;
   
   
    if(right_lane_inds.rows>0){
        for(int i=0;i<right_lane_inds.rows;i++){
            rightx.push_back(nonzerosx.at<int>(right_lane_inds.at<int>(i)));
            righty.push_back(nonzerosy.at<int>(right_lane_inds.at<int>(i)));
        }
    }
    if(left_lane_inds.rows>0){
        for(int i=0;i<left_lane_inds.rows;i++){
            leftx.push_back(nonzerosx.at<int>(left_lane_inds.at<int>(i)));
            lefty.push_back(nonzerosy.at<int>(left_lane_inds.at<int>(i)));
        }
    }
    leftx.convertTo(leftx,CV_32F);
    lefty.convertTo(lefty,CV_32F);
    rightx.convertTo(rightx,CV_32F);
    righty.convertTo(righty,CV_32F);


    lane_detection::ploty = lane_detection::linspace(0, img.rows,img.rows-1);
    cv::Mat left_fit(2+1,1,CV_32F);
    /*
    for(int i =0;i<leftx.rows;i++)
    {
        std::cout<<"leftx "<<leftxf.row(i)<<std::endl;
    //std::cout<<"lefty "<<lefty<<std::endl;
    }
    */
    //std::cout<<"leftx ="<<leftx<<std::endl;
    if (leftx.rows > 0){
        lane_detection::polyfit(lefty,leftx,left_fit,2);
        //std::cout<<"Left fit = "<<left_fit<<std::endl;
        for (auto value:ploty)
        {
            lane_detection::left_fitx.push_back( left_fit.at<float>(2)*value*value+left_fit.at<float>(1)*value+left_fit.at<float>(0));
        }
    }
    //std::cout<<left_fitx;
    cv::Mat right_fit(2+1,1,CV_32F);
    if(rightx.rows>0){
        lane_detection::polyfit(righty,rightx,right_fit,2);
        for(auto value:lane_detection::ploty)
        {
            lane_detection::right_fitx.push_back(right_fit.at<float>(2)*value*value + right_fit.at<float>(1)*value +right_fit.at<float>(0));
        }

    }

}

void lane_detection::polyfit(cv::Mat src_x,cv::Mat src_y, cv::Mat& dst, int order)
{
    CV_Assert((src_x.rows>0)&&(src_y.rows>0)&&(src_x.cols==1)&&(src_y.cols==1)
            &&(dst.cols==1)&&(dst.rows==(order+1))&&(order>=1));
    
    cv::Mat X;
    X = cv::Mat::zeros(src_x.rows, order+1,CV_32F);
   
    cv::Mat copy;
    for(int i = 0; i <=order;i++)
    {
        copy = src_x.clone();
        cv::pow(copy,i,copy);
        cv::Mat m1= X.col(i);
        copy.col(0).copyTo(m1);
    }
    cv::Mat Xt;
    cv::transpose(X,Xt);
    cv::Mat temp = Xt*X;
    cv::Mat temp2;
    cv::invert (temp,temp2);
    cv::Mat temp3 = temp2*Xt;
    cv::Mat W;
    W = temp3*src_y;
    W.copyTo(dst);
}



cv::Mat lane_detection::draw_lanes(cv::Mat left_fitx,cv::Mat right_fitx,std::vector<double> ploty,cv::Mat img)
{   
    //std::cout<<left_fitx<<std::endl;
    std::vector<cv::Point>Points_line_left;
    std::vector<cv::Point>Points_line_right;
    std::vector<std::vector<cv::Point> > fillContAll;

    cv::Mat img2 = cv::Mat::zeros(img.size(), CV_8UC3);

    for(int i=0;i<left_fitx.rows;i++)
    {   
        //std::cout<<cv::Point((int)left_fitx.at<double>(i),(int)ploty.at(i))<<std::endl;
        
        Points_line_left.push_back(cv::Point((int)left_fitx.at<double>(i,0),(int)ploty.at(i)));
       // Points_line_right.insert(Points_line_right.begin(),cv::Point((int)right_fitx.at<double>(i,0),(int)ploty.at(i)));
        Points_line_left.push_back(cv::Point((int)right_fitx.at<double>(i,0),(int)ploty.at(i)));

    
    }
    //for(int i=0;i<right_fitx.rows;i++)
   // {   
       // std::cout<<cv::Point((int)right_fitx.at<double>(i),(int)ploty.at(i))<<std::endl;
  //      Points_line_right.insert(Points_line_right.begin(),cv::Point((int)ploty.at(i),(int)right_fitx.at<double>(i,0)));
        //Points_line_right.pu(cv::Point((int)right_fitx.at<double>(i,0),(int)ploty.at(i)));
  //  }
    fillContAll.push_back(Points_line_left);
   // fillContAll.push_back(Points_line_right);
    cv::fillPoly(img2,fillContAll,cv::Scalar(0,255,0),8);
    lane_detection::left_fitx.release();
    lane_detection::right_fitx.release();
    lane_detection::ploty.clear();
    return img2;
}
void lane_detection::set_slide_search(const cv::Mat warped)
{
    lane_detection::slide_search(warped);
}

cv::Mat lane_detection::get_lane(const cv::Mat frame)
{

    lane_detection::lane = lane_detection::draw_lanes(lane_detection::left_fitx,lane_detection::right_fitx,lane_detection::ploty,frame);
    return lane_detection::lane;
}

template<typename T>
std::vector<double> lane_detection::linspace(T start_in, T end_in, int num_in)
{

  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}