#include "baseModel.h"

BaseModel::BaseModel()
{
}

BaseModel::~BaseModel()
{
}

FaceDetection::FaceDetection()
{
}

FaceDetection::~FaceDetection()
{
}

bool FaceDetection::loadModel(std::string& aModelPath)
{
  return false;
}

void FaceDetection::preprocess(cv::Mat& aInputImage,
                               cv::Mat& aProcessedImage)
{
}
                          
void FaceDetection::inference(cv::Mat& aProcessedImage,
                              std::vector<faceBboxStruct>& aFaceBbox)
{
}

void FaceDetection::alignFace(cv::Mat& aProcessedImage,
                              std::vector<faceBboxStruct>& aFaceBbox,
                              std::vector<cv::Mat>& aAlignedImageContainer)
{
}

FaceRecognition::FaceRecognition()
{
}

FaceRecognition::~FaceRecognition()
{
}

bool FaceRecognition::loadModel(std::string& aModelPath)
{
  return false;
}

void FaceRecognition::preprocess(cv::Mat& aInputImage,
                                 cv::Mat& aProcessedImage)
{
}

void inference(cv::Mat& aProcessedImage,
               std::vector<float>& aEmbedding)
{
}

float calculateSimilarity(std::vector<float>& aEmbedding1,
                          std::vector<float>& aEmbedding2)
{
  return 0.0f;
}