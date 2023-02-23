#ifndef _BASEMODEL_H_
#define _BASEMODEL_H_

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class BaseModel
{
  public:
    ~BaseModel();
    virtual bool loadModel(std::string& aModelPath) = 0;
    virtual void preprocess(cv::Mat& aInputImage, cv::Mat& aProcessedImage) = 0;

  protected:
    BaseModel();
};

struct faceBboxStruct {
  float x1;
  float y1;
  float x2;
  float y2;
  float score;
  float landmarks[10];
};

class FaceDetection : public BaseModel
{
  public:
    FaceDetection();
    ~FaceDetection();
   
    bool loadModel(std::string& aModelPath);
    void preprocess(cv::Mat& aInputImage,
                    cv::Mat& aProcessedImage);
    void inference(cv::Mat& aProcessedImage,
                   std::vector<faceBboxStruct>& aFaceBbox);
    void alignFace(cv::Mat& aProcessedImage,
                   std::vector<faceBboxStruct>& aFaceBboxes,
                   std::vector<cv::Mat>& aAlignedImageContainer);

  protected:
    void nms(std::vector<faceBboxStruct>& input,
             std::vector<faceBboxStruct>& output,
             float iouThreshold = 0.6,
             unsigned int topk = 300);
    
    float getIOU(faceBboxStruct& box1, faceBboxStruct& box2);
};

class FaceRecognition : public BaseModel
{
  public:
    FaceRecognition();
    ~FaceRecognition();
   
    bool loadModel(std::string& aModelPath);
    void preprocess(cv::Mat& aInputImage,
                    cv::Mat& aProcessedImage);
    void inference(cv::Mat& aProcessedImage,
                   std::vector<float>& aEmbedding);
    float calculateSimilarity(std::vector<float>& aEmbedding1,
                              std::vector<float>& aEmbedding2);
  protected:
    const int mFeatureDim = 512;
    const int mInputSize = 112;
};

#endif 