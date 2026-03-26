#ifndef FAST_FOUNDATION_STEREO_ESTIMATOR_H
#define FAST_FOUNDATION_STEREO_ESTIMATOR_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "stereo_estimator.h"

#include "cuda_utils.h"

// Wrapper for CUDA Memory Management
struct DeviceBuffer {
    void* ptr = nullptr;
    size_t size = 0;  // bytes
    
    DeviceBuffer(size_t s) : size(s) { 
        CHECK_CUDA(cudaMalloc(&ptr, size)); 
    }
    ~DeviceBuffer() { 
        if (ptr) cudaFree(ptr); 
    }
    
    template<typename T> T* as() const { return static_cast<T*>(ptr); }
};

class FastFoundationStereoEstimator : public StereoEstimator {
public:
    // Constructor: loads engine and allocates GPU memory
    FastFoundationStereoEstimator(const std::string& featureEnginePath, const std::string& postEnginePath, int height, int width);
    
    // Destructor: releases resources
    ~FastFoundationStereoEstimator();

    /**
     * @brief Run stereo matching inference
     * @param leftImg Input left image (RGB, CV_8UC3), expected m_inputH x m_inputW input)
     * @param rightImg Input right image (RGB, CV_8UC3), expected m_inputH x m_inputW input)
     * @param outputDisp Output disparity map (Float, CV_32FC1)
     */
    bool inference(const cv::Mat& leftImg, const cv::Mat& rightImg, cv::Mat& outputDisp);

    void printIOTensors();

private:
    // TensorRT components
    nvinfer1::IRuntime* m_runtime = nullptr;

    nvinfer1::ICudaEngine* m_feature_engine = nullptr;
    nvinfer1::IExecutionContext* m_feature_context = nullptr;
    
    nvinfer1::ICudaEngine* m_post_engine = nullptr;
    nvinfer1::IExecutionContext* m_post_context = nullptr;

    cudaStream_t m_stream = nullptr;

    // Dimensions
    int m_inputH;
    int m_inputW;

    // Device (GPU) pointers
    std::unique_ptr<DeviceBuffer> m_dRawRGBLeft; // uint8_t buffer
    std::unique_ptr<DeviceBuffer> m_dRawRGBRight; // uint8_t buffer

    std::unique_ptr<DeviceBuffer> m_dLeft;
    std::unique_ptr<DeviceBuffer> m_dRight;

    std::unique_ptr<DeviceBuffer> m_dFeaturesLeft04;
    std::unique_ptr<DeviceBuffer> m_dFeaturesLeft08;
    std::unique_ptr<DeviceBuffer> m_dFeaturesLeft16;
    std::unique_ptr<DeviceBuffer> m_dFeaturesLeft32;
    std::unique_ptr<DeviceBuffer> m_dFeaturesRight04;
    std::unique_ptr<DeviceBuffer> m_dStem2x;
    std::unique_ptr<DeviceBuffer> m_dGwcVolume;

    std::unique_ptr<DeviceBuffer> m_dFeaturesLeft04Half;
    std::unique_ptr<DeviceBuffer> m_dFeaturesRight04Half;

    std::unique_ptr<DeviceBuffer> m_dDisp;

    // Host (CPU) buffers for flattening/preprocessing
    // Output
    std::vector<float> m_hDisp;

    std::unique_ptr<InputPadder> m_padder;

    // Helper functions
    void loadEngine(const std::string& featureEnginePath, const std::string& postEnginePath);
    
    // void preprocessImage(const cv::Mat& src, std::vector<float>& dstBuffer);
    // void preprocessImage(const cv::Mat& leftImage, const cv::Mat& rightImage);

    bool infer(const cv::Mat& leftImg, const cv::Mat& rightImg, cv::Mat& outputDisp);
};




#endif // FAST_FOUNDATION_STEREO_ESTIMATOR_H