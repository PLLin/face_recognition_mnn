#include <math.h>
#include <fstream>
#include <iostream>
#include <memory>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>

#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

using namespace MNN;

#define OUTPUT_NODE_NAME "683"
#define TARGET_LEN 112


void preprocess(const char* inputImageFileName, Tensor* input) {
    int originalWidth;
    int originalHeight;
    int originChannel;
    unsigned char * inputImage = stbi_load(inputImageFileName, &originalWidth, &originalHeight, &originChannel, 4);
    const float means[3] = {127.5f, 127.5f, 127.5f};
    const float norms[3] = {2.0f / 255.0f, 2.0f / 255.0f, 2.0f / 255.0f};
    CV::ImageProcess::Config preProcessConfig;
    ::memcpy(preProcessConfig.mean, means, sizeof(means));
    ::memcpy(preProcessConfig.normal, norms, sizeof(norms));
    preProcessConfig.sourceFormat = CV::RGBA;
    preProcessConfig.destFormat   = CV::RGB;
    preProcessConfig.filterType   = CV::BILINEAR;
    auto pretreat = std::shared_ptr<CV::ImageProcess>(CV::ImageProcess::create(preProcessConfig));
    CV::Matrix trans;
    trans.postScale(1.0 / TARGET_LEN, 1.0 / TARGET_LEN);
    trans.postScale(originalWidth, originalHeight);
    pretreat->setMatrix(trans);
    const auto rgbaPtr = reinterpret_cast<uint8_t*>(inputImage);
    pretreat->convert(rgbaPtr, originalWidth, originalHeight, 0, input);
}


float CalculateSimilarity(std::vector<float> feat1, std::vector<float> feat2) {
    float inner_product = 0.0f;
    float feat_norm1 = 0.0f;
    float feat_norm2 = 0.0f;
    for(int i = 0; i < 512; ++i) {
        inner_product += feat1[i] * feat2[i];
        feat_norm1 += feat1[i] * feat1[i];
        feat_norm2 += feat2[i] * feat2[i];
	  }
	  return inner_product / sqrt(feat_norm1) / sqrt(feat_norm2);
}

std::vector<float> inference(std::shared_ptr<MNN::Interpreter> mnnNet, Session* session) {
    mnnNet->runSession(session);
    auto embeddings = mnnNet->getSessionOutput(session, OUTPUT_NODE_NAME);
    Tensor embeddingsHost(embeddings, Tensor::CAFFE);
    embeddings->copyToHostTensor(&embeddingsHost);
    std::vector<float> feat(512);
    for (int i = 0; i < 512; ++i) {
        feat[i] = embeddingsHost.host<float>()[i];
    }
    return feat;
}

int main(int argc, char* argv[]) {

    const auto frModel = argv[1];
    const char* inputImageFileName1 = argv[2];
    const char* inputImageFileName2 = argv[3];

    std::shared_ptr<MNN::Interpreter> mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(frModel));
    MNN::ScheduleConfig netConfig;
    netConfig.type = MNN_FORWARD_CPU;
    netConfig.numThread = 4;
    Session* session = mnnNet->createSession(netConfig);
    Tensor* input = mnnNet->getSessionInput(session, nullptr);

    if (input->elementSize() <= 4) {
        mnnNet->resizeTensor(input, {1, 3, TARGET_LEN, TARGET_LEN});
        mnnNet->resizeSession(session);
    }

    preprocess(inputImageFileName1, input);
    std::vector<float> feat1 = inference(mnnNet, session);

    preprocess(inputImageFileName2, input);
    std::vector<float> feat2 = inference(mnnNet, session);

    float sim = CalculateSimilarity(feat1, feat2);
    std::cout << "sim:" << sim << std::endl;
    return 0;
}

