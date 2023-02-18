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

class FaceDetection : public BaseModel
{
  public:
    FaceDetection();
    ~FaceDetection();
   
    struct faceBboxStruct {
      int x0;
      int y0;
      int width;
      int height;
      float faceMark[10];
    };

    bool loadModel(std::string& aModelPath);
    void preprocess(cv::Mat& aInputImage,
                    cv::Mat& aProcessedImage);
    void inference(cv::Mat& aProcessedImage,
                   std::vector<faceBboxStruct>& aFaceBbox);
    void alignFace(cv::Mat& aProcessedImage,
                   std::vector<faceBboxStruct>& aFaceBbox,
                   std::vector<cv::Mat>& aAlignedImageContainer);
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