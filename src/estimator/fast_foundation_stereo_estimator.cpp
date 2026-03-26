#include "fast_foundation_stereo_estimator.h"

#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include "gwc_volume_kernel.h"

namespace {

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[FastFoundationStereoEstimator] " << msg << std::endl;
        }
    }
} gLogger;

std::vector<char> _loadEngineData(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.good()) throw std::runtime_error("Error reading engine file: " + enginePath);
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

constexpr int kFeaturesLeft04Size  = 224 * 120 * 160;
constexpr int kFeaturesLeft08Size  = 192 * 60 * 80;
constexpr int kFeaturesLeft16Size  = 320 * 30 * 40;
constexpr int kFeaturesLeft32Size  = 304 * 15 * 20;
constexpr int kFeaturesRight04Size = 224 * 120 * 160;
// constexpr int kStem2xSize          = 16 * 240 * 320;
constexpr int kStem2xSize          = 32 * 240 * 320;
constexpr int kGwcVolumeSize       = 8 * 48 * 120 * 160;

}  // namespace

FastFoundationStereoEstimator::FastFoundationStereoEstimator(const std::string& featureEnginePath, const std::string& postEnginePath, int height, int width) 
    : m_inputH(height), m_inputW(width) {
    
    m_padder = std::make_unique<InputPadder>(m_inputH, m_inputW);

    // 1. Resize Host Buffers (Only what we actually need on CPU)
    m_hDisp.resize(1 * m_inputH * m_inputW);

    // 2. Allocate GPU Memory using RAII containers
    size_t inputBytes = 3 * m_inputH * m_inputW * sizeof(float);
    m_dLeft  = std::make_unique<DeviceBuffer>(inputBytes);
    m_dRight = std::make_unique<DeviceBuffer>(inputBytes);
    m_dDisp  = std::make_unique<DeviceBuffer>(m_inputH * m_inputW * sizeof(float));

    m_dRawRGBLeft = std::make_unique<DeviceBuffer>(m_inputH * m_inputW * 3 * sizeof(uint8_t));
    m_dRawRGBRight = std::make_unique<DeviceBuffer>(m_inputH * m_inputW * 3 * sizeof(uint8_t));
    
    CHECK_CUDA(cudaMemset(m_dDisp->ptr, 0, m_dDisp->size));

    m_dFeaturesLeft04  = std::make_unique<DeviceBuffer>(kFeaturesLeft04Size * sizeof(float));
    m_dFeaturesLeft08  = std::make_unique<DeviceBuffer>(kFeaturesLeft08Size * sizeof(float));
    m_dFeaturesLeft16  = std::make_unique<DeviceBuffer>(kFeaturesLeft16Size * sizeof(float));
    m_dFeaturesLeft32  = std::make_unique<DeviceBuffer>(kFeaturesLeft32Size * sizeof(float));
    m_dFeaturesRight04 = std::make_unique<DeviceBuffer>(kFeaturesRight04Size * sizeof(float));
    m_dStem2x          = std::make_unique<DeviceBuffer>(kStem2xSize * sizeof(float));
    m_dGwcVolume       = std::make_unique<DeviceBuffer>(kGwcVolumeSize * sizeof(half));

    m_dFeaturesLeft04Half  = std::make_unique<DeviceBuffer>(kFeaturesLeft04Size * sizeof(half));
    m_dFeaturesRight04Half = std::make_unique<DeviceBuffer>(kFeaturesRight04Size * sizeof(half));

    CHECK_CUDA(cudaStreamCreate(&m_stream));

    // 3. Load Engine
    loadEngine(featureEnginePath, postEnginePath);
    printIOTensors();
}

FastFoundationStereoEstimator::~FastFoundationStereoEstimator() {
    // Note: All DeviceBuffer unique_ptrs automatically call cudaFree!

    if (m_stream) cudaStreamDestroy(m_stream);

    if (m_feature_context) delete m_feature_context;
    if (m_feature_engine) delete m_feature_engine;
    if (m_post_context) delete m_post_context;
    if (m_post_engine) delete m_post_engine;
    if (m_runtime) delete m_runtime;
}

void FastFoundationStereoEstimator::loadEngine(const std::string& featureEnginePath, const std::string& postEnginePath) {

    m_runtime = nvinfer1::createInferRuntime(gLogger);

    auto setupContext = [&](const std::string& path, nvinfer1::ICudaEngine*& engine, nvinfer1::IExecutionContext*& context) {
        auto buf = _loadEngineData(path);
        engine = m_runtime->deserializeCudaEngine(buf.data(), buf.size());
        context = engine->createExecutionContext();
    };

    // feature runner.
    setupContext(featureEnginePath, m_feature_engine, m_feature_context);
    
    nvinfer1::Dims4 inputDims(1, 3, m_inputH, m_inputW);
    m_feature_context->setInputShape("left", inputDims);
    m_feature_context->setInputShape("right", inputDims);

    m_feature_context->setTensorAddress("left", m_dLeft->ptr);
    m_feature_context->setTensorAddress("right", m_dRight->ptr);

    m_feature_context->setTensorAddress("features_left_04", m_dFeaturesLeft04->ptr);
    m_feature_context->setTensorAddress("features_left_08", m_dFeaturesLeft08->ptr);
    m_feature_context->setTensorAddress("features_left_16", m_dFeaturesLeft16->ptr);
    m_feature_context->setTensorAddress("features_left_32", m_dFeaturesLeft32->ptr);
    m_feature_context->setTensorAddress("features_right_04", m_dFeaturesRight04->ptr);
    m_feature_context->setTensorAddress("stem_2x", m_dStem2x->ptr);  

    // post runner.
    setupContext(postEnginePath, m_post_engine, m_post_context);
    
    m_post_context->setInputShape("features_left_04", nvinfer1::Dims4(1, 224, 120, 160));
    m_post_context->setInputShape("features_left_08", nvinfer1::Dims4(1, 192, 60, 80));
    m_post_context->setInputShape("features_left_16", nvinfer1::Dims4(1, 320, 30, 40));
    m_post_context->setInputShape("features_left_32", nvinfer1::Dims4(1, 304, 15, 20));
    m_post_context->setInputShape("features_right_04", nvinfer1::Dims4(1, 224, 120, 160));
    m_post_context->setInputShape("stem_2x", nvinfer1::Dims4(1, 32, 240, 320));
    m_post_context->setInputShape("gwc_volume", nvinfer1::Dims{5, {1, 8, 48, 120, 160}});

    m_post_context->setTensorAddress("features_left_04", m_dFeaturesLeft04->ptr);
    m_post_context->setTensorAddress("features_left_08", m_dFeaturesLeft08->ptr);
    m_post_context->setTensorAddress("features_left_16", m_dFeaturesLeft16->ptr);
    m_post_context->setTensorAddress("features_left_32", m_dFeaturesLeft32->ptr);
    m_post_context->setTensorAddress("features_right_04", m_dFeaturesRight04->ptr);
    m_post_context->setTensorAddress("stem_2x", m_dStem2x->ptr);
    m_post_context->setTensorAddress("gwc_volume", m_dGwcVolume->ptr);

    m_post_context->setTensorAddress("disp", m_dDisp->ptr);

}

/*
void FastFoundationStereoEstimator::preprocessImage(const cv::Mat& src, std::vector<float>& dstBuffer) {
    std::vector<cv::Mat> bgr_channels(3);
    cv::split(src, bgr_channels); 

    int area = m_inputH * m_inputW;
    cv::Mat r_plane(m_inputH, m_inputW, CV_32FC1, dstBuffer.data());             
    cv::Mat g_plane(m_inputH, m_inputW, CV_32FC1, dstBuffer.data() + area);        
    cv::Mat b_plane(m_inputH, m_inputW, CV_32FC1, dstBuffer.data() + area * 2);    

    bgr_channels[2].convertTo(r_plane, CV_32FC1, 1.0f, 0.0f);
    bgr_channels[1].convertTo(g_plane, CV_32FC1, 1.0f, 0.0f);
    bgr_channels[0].convertTo(b_plane, CV_32FC1, 1.0f, 0.0f);
}
*/

void FastFoundationStereoEstimator::printIOTensors() {
    auto printEngineTensors = [](nvinfer1::ICudaEngine* engine, const char* name) {
        std::cout << "\n=== " << name << " Engine ===" << std::endl;
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* tensorName = engine->getIOTensorName(i);
            auto mode = engine->getTensorIOMode(tensorName);
            auto dims = engine->getTensorShape(tensorName);
            
            std::cout << "Tensor: " << tensorName 
                      << " (I/O: " << (mode == nvinfer1::TensorIOMode::kINPUT ? "IN" : "OUT") << ")"
                      << " dims: ";
            for (int j = 0; j < dims.nbDims; ++j) std::cout << dims.d[j] << " ";
            std::cout << std::endl;
        }
    };
    printEngineTensors(m_feature_engine, "Feature");
    printEngineTensors(m_post_engine, "Post");
}

bool FastFoundationStereoEstimator::inference(const cv::Mat& leftRaw, const cv::Mat& rightRaw, cv::Mat& outputDisp) {    
    if (leftRaw.rows != m_inputH || leftRaw.cols != m_inputW) {
        std::cerr << "Check left_image shape failed! shape: " << leftRaw.rows << "x" << leftRaw.cols << std::endl;
        return false;
    }

    if (rightRaw.rows != m_inputH || rightRaw.cols != m_inputW) {
        std::cerr << "Check right_image shape failed! shape: " << rightRaw.rows << "x" << rightRaw.cols << std::endl;
        return false;
    }

    cv::Mat leftImg = m_padder->pad(leftRaw);
    cv::Mat rightImg = m_padder->pad(rightRaw);

    bool ret = infer(leftImg, rightImg, outputDisp);

    outputDisp = m_padder->unpad(outputDisp);

    return ret;
}

bool FastFoundationStereoEstimator::infer(const cv::Mat& leftImg, const cv::Mat& rightImg, cv::Mat& outputDisp) {
    if (leftImg.empty() || rightImg.empty()) {
        std::cerr << "Input images are empty!" << std::endl;
        return false;
    }

    if (leftImg.rows != m_inputH || leftImg.cols != m_inputW) {
        std::cerr << "Input size mismatch! Expected " << m_inputW << "x" << m_inputH << std::endl;
        return false;
    }

    // 1. Preprocess (CPU -> Host Buffer)
    CHECK_CUDA(cudaMemcpyAsync(m_dRawRGBLeft->ptr, leftImg.data, m_inputH * m_inputW * 3, cudaMemcpyHostToDevice, m_stream));
    CHECK_CUDA(cudaMemcpyAsync(m_dRawRGBRight->ptr, rightImg.data, m_inputH * m_inputW * 3, cudaMemcpyHostToDevice, m_stream));

    // 2. Run CUDA Preprocessing Kernels (Converts BGR HWC -> RGB CHW Float)
    LaunchPreprocessKernel(m_dRawRGBLeft->as<uint8_t>(), m_dLeft->as<float>(), 
                           m_inputW, m_inputH, m_stream);
    LaunchPreprocessKernel(m_dRawRGBRight->as<uint8_t>(), m_dRight->as<float>(), 
                           m_inputW, m_inputH, m_stream);

    // 3. Feature Extraction
    if (!m_feature_context->enqueueV3(m_stream)) {
        std::cerr << "TensorRT execution failed!" << std::endl;
        return false;
    }

    // 4. Build GWC Volume via CUDA
    convertFloatToHalf(m_dFeaturesLeft04->as<float>(), m_dFeaturesLeft04Half->as<half>(), kFeaturesLeft04Size);
    convertFloatToHalf(m_dFeaturesRight04->as<float>(), m_dFeaturesRight04Half->as<half>(), kFeaturesRight04Size);

    int B = 1;
    int C = 224;
    int H = 120;
    int W = 160;

    int max_disp = 192 / 4; // 192 / 4 = 48
    int num_groups = 8;

    LaunchGwcVolumeKernel(
        m_dFeaturesLeft04Half->as<half>(), 
        m_dFeaturesRight04Half->as<half>(), 
        m_dGwcVolume->as<half>(), 
        B, C, H, W, max_disp, num_groups, false, m_stream
    );

    // 5. Post Processing Inference
    if (!m_post_context->enqueueV3(m_stream)) {
        std::cerr << "TensorRT execution failed!" << std::endl;
        return false;
    }

    // 6. Copy results to CPU
    CHECK_CUDA(cudaMemcpyAsync(m_hDisp.data(), m_dDisp->ptr, m_dDisp->size, cudaMemcpyDeviceToHost, m_stream));
    CHECK_CUDA(cudaStreamSynchronize(m_stream));
    
    cv::Mat(m_inputH, m_inputW, CV_32FC1, m_hDisp.data()).copyTo(outputDisp);

    return true;
}

