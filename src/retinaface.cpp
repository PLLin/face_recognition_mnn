#include "retinaface.h"

using namespace MNN;

Retinaface::Retinaface()
{
  mPretreat = std::shared_ptr<MNN::CV::ImageProcess>(
    MNN::CV::ImageProcess::create(
      MNN::CV::BGR,
      MNN::CV::BGR,
      mMeanVals, 3,
      mNormVals, 3
    )
  );
};

bool Retinaface::loadModel(std::string& aModelPath)
{
  mMnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(aModelPath.c_str()));
  mNetConfig.type = MNN_FORWARD_CPU;
  mNetConfig.numThread = 4;
  mSession = mMnnNet->createSession(mNetConfig);
  mTensor = mMnnNet->getSessionInput(mSession, nullptr);
  mInputBatch = mTensor->batch();
  mInputChannel = mTensor->channel();
  mInputHeight = mTensor->height();
  mInputWidth = mTensor->width();
  int dimension_type = mTensor->getDimensionType();
  if (dimension_type == MNN::Tensor::CAFFE) {
    mMnnNet->resizeTensor(mTensor, {mInputBatch, mInputChannel, mInputHeight, mInputWidth});
    mMnnNet->resizeSession(mSession);
  }
  return true;
};

void Retinaface::preprocess(cv::Mat& aInputImage,
                            cv::Mat& aProcessedImage)
{
  aProcessedImage = aInputImage;
};

void Retinaface::inference(cv::Mat& aProcessedImage,
                           std::vector<faceBboxStruct>& aFaceBbox)
{
  int originalWidth = aProcessedImage.size().width;
  int originalHeight = aProcessedImage.size().height;
  int originChannel = aProcessedImage.channels();

  // 1. prepare input tensor.
  cv::Mat resizedImage;
  cv::resize(aProcessedImage, resizedImage, cv::Size(mInputWidth, mInputHeight));
  unsigned char * inputImage = resizedImage.data;
  const auto rgbaPtr = reinterpret_cast<uint8_t*>(inputImage);
  mPretreat->convert(rgbaPtr, mInputWidth, mInputHeight, 0, mTensor);

  // 2. inference scores & boxes.
  mMnnNet->runSession(mSession);
  auto outputTensor = mMnnNet->getSessionOutputAll(mSession);

  // 3. rescale & exclude.
  std::vector<faceBboxStruct> bboxCollection;
  this->generateBboxes(bboxCollection, outputTensor, mScoreThreshold, originalHeight, originalWidth);

  // 4. hard|blend nms with topk.
  this->nms(bboxCollection, aFaceBbox);

  // for (auto bbox: aFaceBbox) {
  //   std::cout << bbox.x1 << std::endl;
  //   std::cout << bbox.x2 << std::endl;
  //   std::cout << bbox.y1 << std::endl;
  //   std::cout << bbox.y2 << std::endl;
  //   std::cout << bbox.score << std::endl;
  //   for (int i = 0; i < 5; i++) {
  //     std::cout << bbox.landmarks[2*i] << " " << bbox.landmarks[2*i+1] << std::endl;
  //   }
  // }
};

void Retinaface::generateBboxes(std::vector<faceBboxStruct> &aBboxCollection,
                                const std::map<std::string, MNN::Tensor*> &aOutputTensors,
                                float aScoreThreshold,
                                float aImgHeight,
                                float aImgWidth)
{
  auto deviceBboxesPtr = aOutputTensors.at("bbox"); // e.g (1,16800,4)
  auto deviceProbsPtr = aOutputTensors.at("conf"); // e.g (1,16800,2) after softmax
  auto deviceLandmarksPtr = aOutputTensors.at("landmarks");

  MNN::Tensor hostBboxesTensor(deviceBboxesPtr, deviceBboxesPtr->getDimensionType());
  MNN::Tensor hostProbsTensor(deviceProbsPtr, deviceProbsPtr->getDimensionType());
  MNN::Tensor hostLandmarksTensor(deviceLandmarksPtr, deviceLandmarksPtr->getDimensionType());

  deviceBboxesPtr->copyToHostTensor(&hostBboxesTensor);
  deviceProbsPtr->copyToHostTensor(&hostProbsTensor);
  deviceLandmarksPtr->copyToHostTensor(&hostLandmarksTensor);

  auto bboxDims = hostBboxesTensor.shape();
  const unsigned int bboxNum = bboxDims.at(1); // n = ?

  std::vector<RetinaAnchor> anchors;
  this->generateAnchors(mInputHeight, mInputWidth, anchors);

  const unsigned int numAnchors = anchors.size();
  if (numAnchors != bboxNum)
    throw std::runtime_error("mismatch numAnchors != bboxNum");

  const float *bboxesPtr = hostBboxesTensor.host<float>();
  const float *probsPtr = hostProbsTensor.host<float>();
  const float *landmarksPtr = hostLandmarksTensor.host<float>();

  aBboxCollection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < numAnchors; ++i)
  {
    float conf = probsPtr[2 * i + 1];
    if (conf < aScoreThreshold) continue; // filter first.

    float priorCX = anchors.at(i).cx;
    float priorCY = anchors.at(i).cy;
    float priorSKX = anchors.at(i).s_kx;
    float priorSKY = anchors.at(i).s_ky;

    float dx = bboxesPtr[4 * i + 0];
    float dy = bboxesPtr[4 * i + 1];
    float dw = bboxesPtr[4 * i + 2];
    float dh = bboxesPtr[4 * i + 3];
    // ref: https://github.com/biubug6/Pytorch_Retinaface/blob/master/utils/box_utils.py
    float cx = priorCX + dx * mVariance[0] * priorSKX;
    float cy = priorCY + dy * mVariance[0] * priorSKY;
    float w = priorSKX * std::exp(dw * mVariance[1]);
    float h = priorSKY * std::exp(dh * mVariance[1]); // norm coor (0.,1.)

    faceBboxStruct box;
    box.x1 = (cx - w / 2.f) * aImgWidth;
    box.y1 = (cy - h / 2.f) * aImgHeight;
    box.x2 = (cx + w / 2.f) * aImgWidth;
    box.y2 = (cy + h / 2.f) * aImgHeight;
    box.score = conf;
    for (int i = 0; i < 5; i++) {
      float landmarkX = (priorCX + landmarksPtr[2*i] * mVariance[0] * priorSKX);
      float landmarkY = (priorCY + landmarksPtr[2*i+1] * mVariance[0] * priorSKY);
      box.landmarks[2*i] = landmarkX * aImgWidth;
      box.landmarks[2*i+1] = landmarkY * aImgHeight;
    }
    aBboxCollection.push_back(box);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }
};

void Retinaface::generateAnchors(const int aTargetHeight,
                                 const int aTargetWidth,
                                 std::vector<RetinaAnchor> &aAnchors)
{
  std::vector<std::vector<int>> featureMaps;
  for (auto step: mSteps)
  {
    featureMaps.push_back(
        {
            (int) std::ceil((float) aTargetHeight / (float) step),
            (int) std::ceil((float) aTargetWidth / (float) step)
        } // ceil
    );
  }

  aAnchors.clear();
  const int featureMapsNum = featureMaps.size();

  for (int k = 0; k < featureMapsNum; ++k)
  {
    auto featureMapsTmp = featureMaps.at(k); // e.g [640//8,640//8]
    auto minSizesTmp = mMinSizes.at(k); // e.g [8,16]
    int featureMapsHeight = featureMapsTmp.at(0);
    int featureMapsWidth = featureMapsTmp.at(1);

    for (int i = 0; i < featureMapsHeight; ++i)
    {
      for (int j = 0; j < featureMapsWidth; ++j)
      {
        for (auto min_size: minSizesTmp)
        {
          float skx = (float) min_size / (float) aTargetWidth; // e.g 16/w
          float sky = (float) min_size / (float) aTargetHeight; // e.g 16/h
          float cx = ((float) j + 0.5f) * (float) mSteps.at(k) / (float) aTargetWidth;
          float cy = ((float) i + 0.5f) * (float) mSteps.at(k) / (float) aTargetHeight;
          aAnchors.push_back(RetinaAnchor{cx, cy, skx, sky}); // without clip
        }
      }
    }
  }
};
