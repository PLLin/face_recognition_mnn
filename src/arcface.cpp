#include "arcface.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define OUTPUT_NODE_NAME "683"
using namespace MNN;

Arcface::Arcface()
{
  mPretreat = std::shared_ptr<MNN::CV::ImageProcess>(
    MNN::CV::ImageProcess::create(
      MNN::CV::BGR,
      MNN::CV::RGB,
      mMeanVals, 3,
      mNormVals, 3
    )
  );
}

Arcface::~Arcface()
{
}

bool Arcface::loadModel(std::string& aModelPath)
{
  mMnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(aModelPath.c_str()));
  mNetConfig.type = MNN_FORWARD_CPU;
  mNetConfig.numThread = 4;
  mSession = mMnnNet->createSession(mNetConfig);
  mTensor = mMnnNet->getSessionInput(mSession, nullptr);
  if (mTensor->elementSize() <= 4) {
      mMnnNet->resizeTensor(mTensor, {1, 3, 112, 112});
      mMnnNet->resizeSession(mSession);
  }
  return true;
}

void Arcface::preprocess(cv::Mat& aInputImage, cv::Mat& aProcessedImage)
{
  cv::resize(aInputImage, aProcessedImage, cv::Size(this->mInputSize, this->mInputSize));
}

void Arcface::inference(cv::Mat& aProcessedImage, std::vector<float>& aEmbedding)
{
  int originalWidth = aProcessedImage.size().width;
  int originalHeight = aProcessedImage.size().height;
  int originChannel = aProcessedImage.channels();

  unsigned char * inputImage = aProcessedImage.data;
  const auto rgbaPtr = reinterpret_cast<uint8_t*>(inputImage);

  mPretreat->convert(rgbaPtr, originalWidth, originalHeight, 0, mTensor);
  mMnnNet->runSession(mSession);
  auto embeddings = mMnnNet->getSessionOutput(mSession, OUTPUT_NODE_NAME);
  Tensor embeddingsHost(embeddings, Tensor::CAFFE);
  embeddings->copyToHostTensor(&embeddingsHost);
  for (int i = 0; i < this->mFeatureDim; ++i) {
    aEmbedding.push_back(embeddingsHost.host<float>()[i]);
    // std::cout << aEmbedding[i] << std::endl;
  }
}

float Arcface::calculateSimilarity(std::vector<float>& feat1, std::vector<float>& feat2) {
  float inner_product = 0.0f;
  float feat_norm1 = 0.0f;
  float feat_norm2 = 0.0f;
  for (int i = 0; i < this->mFeatureDim; ++i) {
    inner_product += feat1[i] * feat2[i];
    feat_norm1 += feat1[i] * feat1[i];
    feat_norm2 += feat2[i] * feat2[i];
	}
	return inner_product / sqrt(feat_norm1) / sqrt(feat_norm2);
}