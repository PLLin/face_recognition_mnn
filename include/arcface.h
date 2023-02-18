#include <math.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

#include "baseModel.h"

class Arcface : public FaceRecognition
{
  public:
    Arcface();
    ~Arcface();
    bool loadModel(std::string& aModelPath);
    void preprocess(cv::Mat& aInputImage,
                    cv::Mat& aProcessedImage);
    void inference(cv::Mat& aProcessedImage,
                   std::vector<float>& aEmbedding);
    float calculateSimilarity(std::vector<float>& aEmbedding1,
                              std::vector<float>& aEmbedding2);

  private:
    MNN::Tensor* mTensor;
    MNN::Session* mSession;
    std::shared_ptr<MNN::Interpreter> mMnnNet;
    MNN::ScheduleConfig mNetConfig;
    std::shared_ptr<MNN::CV::ImageProcess> mPretreat;

    const float mMeanVals[3] = {127.5f, 127.5f, 127.5f}; // RGB
    const float mNormVals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
};  