#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <car_detection.h>
#include <iostream>    
#include <fstream>

car_detection::car_detection(float confThreshold,float nmsThreshold, int inpWidth, int inpHeight)
{
    car_detection::confThreshold = confThreshold;
    car_detection::nmsThreshold = nmsThreshold;
    car_detection::inpHeight = inpWidth;
    car_detection::inpHeight = inpHeight;
   
    // Load names of classes
    std::string classesFile = "/home/lamprosp/Desktop/lane_detection/YoloNet/coco.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while(std::getline(ifs,line)) car_detection::classes.push_back(line);
    cv::String modelConfiguration ="/home/lamprosp/Desktop/lane_detection/YoloNet/yolov3.cfg";
    cv::String modelWeights = "/home/lamprosp/Desktop/lane_detection/YoloNet/yolov3.weights";
    car_detection::net = cv::dnn::readNetFromDarknet(modelConfiguration,modelWeights);
    car_detection::net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    car_detection::net.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);

}

void car_detection::start_process(cv::Mat frame)
{
    cv::Mat blob;

    cv::dnn::blobFromImage(frame,blob,1/255.0,cv::Size(224,224),cv::Scalar(0,0,0),true,false);

    car_detection::net.setInput(blob);
    std:: vector<cv::Mat> outs;
    car_detection::net.forward(outs,car_detection::getOutputNames(car_detection::net));
    car_detection::postprocess(frame,outs);
    std::vector<double> layersTimes;
    double freq = cv::getTickFrequency() /1000;
    double t = car_detection::net.getPerfProfile(layersTimes) /freq;
    std::string label = cv::format("Inference time for a frame : %.2f ms", t);
    cv::putText(frame,label,cv::Point(0,15),cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

    cv::Mat detectedFrame;
    frame.convertTo(detectedFrame,CV_8U);

    car_detection::detected_frame = frame;   


}

cv::Mat car_detection::getFrame(cv::Mat frame)
{   
    car_detection::start_process(frame);
    return car_detection::detected_frame;

}

void car_detection::postprocess(cv::Mat& frame,const std::vector<cv::Mat>& outs)
{
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for(size_t i=0; i <outs.size();i++)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        car_detection::drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
}

void car_detection::drawPred(int classId, float conf, int left, int top, int right, int bottom,cv::Mat& frame)
{
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);
    
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)car_detection::classes.size());
        label = car_detection::classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top =std::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0,0,0),1);

}

std::vector<cv::String> car_detection::getOutputNames(const cv::dnn::Net& net)
{
    static std::vector<cv::String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        //get the names of all the layers in the network
        std::vector<cv::String> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;


}