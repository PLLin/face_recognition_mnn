#include "arcface.h"
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
  std::string frModelPath = argv[1];
  std::string inputImageFileName1 = argv[2];
  std::string inputImageFileName2 = argv[3];
  
  // Read Image
  cv::Mat image1 = cv::imread(inputImageFileName1);
  cv::Mat image2 = cv::imread(inputImageFileName2);
  cv::Mat processedImage1, processedImage2;
  std::vector<float> embeddings1, embeddings2;
  
  Arcface* frModel= new Arcface;
  bool flag = frModel->loadModel(frModelPath);
  // Get Image1 Embeddings
  frModel->preprocess(image1, processedImage1);
  frModel->inference(processedImage1, embeddings1);
  // Get Image2 Embeddings
  frModel->preprocess(image2, processedImage2);
  frModel->inference(processedImage2, embeddings2);
  // Calculate Similarity
  float similarity = frModel->calculateSimilarity(embeddings1, embeddings2);
  std::cout << "similarity: " << similarity << std::endl;
  return 0;
}
