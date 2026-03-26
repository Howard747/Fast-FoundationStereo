
#include "dnn_stereo_depth_node.h"

#include <chrono>

#include "estimator/fast_foundation_stereo_estimator.h"

using namespace std::chrono_literals; 

namespace {

bool CheckTimestamp(const builtin_interfaces::msg::Time& time1, const builtin_interfaces::msg::Time& time2, double* cur_timestamp) {
    double timestamp1 = static_cast<double>(time1.sec) + static_cast<double>(time1.nanosec) / 1e9;
    double timestamp2 = static_cast<double>(time2.sec) + static_cast<double>(time2.nanosec) / 1e9;

    *cur_timestamp = timestamp1;

    double delta = timestamp1 - timestamp2;

    // 70ms = 0.07s
    if (std::fabs(delta) > 0.07) {
        RCLCPP_ERROR(rclcpp::get_logger("checkTimestamp"), "timestamp1: %f, timestamp2: %f, delta: %f", timestamp1, timestamp2, delta);
        return false;
    }

    return true;
}

void PublishDisparity(const std_msgs::msg::Header& header, const std::string& encoding, const cv::Mat& image_mat, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr& publisher) {
    // 1. Create a CvImage object
    cv_bridge::CvImage out_msg;

    // 2. Populate the header (timestamp and frame_id are crucial)
    out_msg.header = header;

    // 3. Specify the image encoding (must match your cv::Mat type)
    // Common encodings: "bgr8" (for CV_8UC3), "mono8" (for CV_8UC1), "TYPE_32FC1" (for CV_32F)
    //out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1; 
    out_msg.encoding = encoding;

    // 4. Assign the cv::Mat to the CvImage object
    out_msg.image = image_mat;

    // 5. Convert to a sensor_msgs::msg::Image::SharedPtr and publish
    sensor_msgs::msg::Image::SharedPtr msg = out_msg.toImageMsg();
    publisher->publish(*msg);
}

}  // namespace

DnnStereoDepthNode::DnnStereoDepthNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("dnn_stereo_depth_node", options),
      image_qos_profile_(rclcpp::QoS(10)), cam_info_qos_profile_(rclcpp::QoS(10)) {
    // Declare parameters
    this->declare_parameter(
        "image_reliability",
        static_cast<int>(rclcpp::ReliabilityPolicy::BestEffort));
    this->declare_parameter(
        "cam_info_reliability",
        static_cast<int>(rclcpp::ReliabilityPolicy::BestEffort));

    this->declare_parameter("model_input_height", 480);
    this->declare_parameter("model_input_width", 640);
    this->declare_parameter("input_image_height", 480);
    this->declare_parameter("input_image_width", 640);

    this->declare_parameter("engine_file_path", std::vector<std::string>{""});
    this->declare_parameter("model_type", "");

    this->declare_parameter("trigger_on_demands", true);

    onConfigure();
    onActivate();
}

DnnStereoDepthNode::~DnnStereoDepthNode() {
    onDeactivate();
    onShutdown();
}

void DnnStereoDepthNode::onConfigure() {
    RCLCPP_INFO(this->get_logger(), "Configuring...");

    // Get parameters
    image_reliability_ =
        this->get_parameter("image_reliability").as_int();
    cam_info_reliability_ =
        this->get_parameter("cam_info_reliability").as_int();

    model_input_height_ =
        this->get_parameter("model_input_height").as_int();
    model_input_width_ =
        this->get_parameter("model_input_width").as_int(); 
    input_image_height_ =
        this->get_parameter("input_image_height").as_int();
    input_image_width_ =
        this->get_parameter("input_image_width").as_int(); 

    std::vector<std::string> engine_file_path =
        this->get_parameter("engine_file_path").as_string_array();  

    std::string model_type =
        this->get_parameter("model_type").as_string();

    trigger_on_demands_ =
        this->get_parameter("trigger_on_demands").as_bool();


    // Configure QoS profiles
    image_qos_profile_.reliability(static_cast<rclcpp::ReliabilityPolicy>(image_reliability_));
    image_qos_profile_.history(rclcpp::HistoryPolicy::KeepLast);
    image_qos_profile_.durability(rclcpp::DurabilityPolicy::Volatile);

    cam_info_qos_profile_.reliability(static_cast<rclcpp::ReliabilityPolicy>(cam_info_reliability_));
    cam_info_qos_profile_.history(rclcpp::HistoryPolicy::KeepLast);
    cam_info_qos_profile_.durability(rclcpp::DurabilityPolicy::Volatile);


    disparity_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "disparity", 1);


    // Create StereoEstimator
    RCLCPP_INFO(this->get_logger(), "model_type: %s.", model_type.c_str());
    StereoModel stereo_model = StringToStereoModel(model_type);

    switch(stereo_model) {
        case StereoModel::FAST_FOUNDATION_STEREO:
            assert(engine_file_path.size() == 2);
            RCLCPP_INFO(this->get_logger(), "feature_model_file: %s.", engine_file_path[0].c_str());
            RCLCPP_INFO(this->get_logger(), "post_model_file: %s.", engine_file_path[1].c_str());

            estimator_ = std::make_unique<FastFoundationStereoEstimator>(engine_file_path[0], engine_file_path[1], model_input_height_, model_input_width_);
            break;
        default:
            RCLCPP_ERROR(this->get_logger(), "no model fitted, model_type: %s.", model_type.c_str());
            exit(-1);
            break;
    }

    RCLCPP_INFO(this->get_logger(), "Configured.");
    return;
}

void DnnStereoDepthNode::onActivate() {
    RCLCPP_INFO(this->get_logger(), "Activating...");

    // Create subscribers
    left_ir_image_sub_ =
        std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, "left_ir_image", image_qos_profile_.get_rmw_qos_profile());
    

    right_ir_image_sub_ =
        std::make_unique<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            this, "right_ir_image", image_qos_profile_.get_rmw_qos_profile());

    if (trigger_on_demands_) {
        left_ir_image_sub_->unsubscribe();
        right_ir_image_sub_->unsubscribe();
    }

    // Create synchronizer
    sync_ =
        std::make_shared<message_filters::Synchronizer<ApproximateSyncPolicy>>(
            ApproximateSyncPolicy(30));
    sync_->connectInput(*left_ir_image_sub_, *right_ir_image_sub_);

    sync_->setAgePenalty(0.20); // 50 ms.
    sync_->registerCallback(
        std::bind(&DnnStereoDepthNode::onStereoSyncCallback, this, std::placeholders::_1, std::placeholders::_2));

    trigger_service_ = this->create_service<std_srvs::srv::Empty>(
        "trigger_dnn_stereo", std::bind(&DnnStereoDepthNode::onTriggerCallback, this,
                              std::placeholders::_1, std::placeholders::_2,
                              std::placeholders::_3));

    RCLCPP_INFO(this->get_logger(), "Activated");
    return;
}

bool DnnStereoDepthNode::onTriggerCallback(const std::shared_ptr<rmw_request_id_t> request_header,
                      const std::shared_ptr<std_srvs::srv::Empty::Request> req,
                      const std::shared_ptr<std_srvs::srv::Empty::Response> res) {
    onEnableCallback();

    return true;
}

void DnnStereoDepthNode::onEnableCallback() {
    if (enabled_) {
        RCLCPP_WARN(this->get_logger(), "This service has already been enabled.");
        return;
    }

    enabled_ = true;

    left_ir_image_sub_->subscribe();
    right_ir_image_sub_->subscribe();

    auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(3000.0));
    trigger_timer_ = rclcpp::create_timer(
        this, this->get_clock(), period_ms,
        std::bind(&DnnStereoDepthNode::onDisableCallback, this));
}

void DnnStereoDepthNode::onDisableCallback() {
    if (!enabled_) {
        RCLCPP_WARN(this->get_logger(), "This service has already been disabled.");
        return;
    }

    enabled_ = false;

    trigger_timer_->cancel();

    left_ir_image_sub_->unsubscribe();
    right_ir_image_sub_->unsubscribe();
}

void DnnStereoDepthNode::onDeactivate() {
    RCLCPP_INFO(this->get_logger(), "Deactivating...");

    // Reset synchronizer and subscribers
    left_ir_image_sub_.reset();
    right_ir_image_sub_.reset();
    
    sync_.reset();

    RCLCPP_INFO(this->get_logger(), "Deactivated");
    return;
}

void DnnStereoDepthNode::onShutdown() {
    // Reset all resources

    RCLCPP_INFO(this->get_logger(), "Shutting down...");
    return;
}

void DnnStereoDepthNode::onStereoSyncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& left_ir_msg,
    const sensor_msgs::msg::Image::ConstSharedPtr& right_ir_msg) {

    processOnce(left_ir_msg, right_ir_msg);

    return;
}

bool DnnStereoDepthNode::preprocess(
    cv::Mat& left_image, cv::Mat& right_image) {

    if (left_image.size() != right_image.size()) {
        RCLCPP_INFO(this->get_logger(), "left_image.size() != right_image.size()");
       return false;
    }

    if (left_image.rows != input_image_height_ || left_image.cols != input_image_width_) {
        RCLCPP_INFO(this->get_logger(), "left_image.rows != input_image_height_");
        return false;
    }

    // Step 1: resize left_ir_image/right_ir_image to model_input_size
    if (left_image.rows != model_input_height_ || left_image.cols != model_input_width_) {
        RCLCPP_INFO(this->get_logger(), "left_image.rows != model_input_height_");
        return false;
    }

	// Step 2: Gray2RGB
	// input: left/right is 1-channel ir image, which is grayscale style.
    cv::cvtColor(left_image, left_image, cv::COLOR_GRAY2RGB);
    cv::cvtColor(right_image, right_image, cv::COLOR_GRAY2RGB);

    return true;
}

void DnnStereoDepthNode::processOnce(const sensor_msgs::msg::Image::ConstSharedPtr& left_ir_msg, const sensor_msgs::msg::Image::ConstSharedPtr& right_ir_msg) {

    if (left_ir_msg == nullptr || right_ir_msg == nullptr) {
        return;
    }

    double cur_timestamp;
    if (!CheckTimestamp(left_ir_msg->header.stamp, right_ir_msg->header.stamp, &cur_timestamp)) {
        return;
    }

    cv::Mat left_ir_image, right_ir_image;
    try {
        // Convert image
        cv_bridge::CvImagePtr left_ir_cv_ptr = cv_bridge::toCvCopy(left_ir_msg, left_ir_msg->encoding);
        left_ir_image = left_ir_cv_ptr->image;

        cv_bridge::CvImagePtr right_ir_cv_ptr = cv_bridge::toCvCopy(right_ir_msg, right_ir_msg->encoding);
        right_ir_image = right_ir_cv_ptr->image;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Exception in image callback: %s", e.what());
    }

    auto start = std::chrono::high_resolution_clock::now();

    if (!preprocess(left_ir_image, right_ir_image)) {
        RCLCPP_ERROR(get_logger(), "Failed in preprocess.");
        return;
    }

    auto start1 = std::chrono::high_resolution_clock::now();

    cv::Mat disparity;
    estimator_->inference(left_ir_image, right_ir_image, disparity);

    auto end1 = std::chrono::high_resolution_clock::now();


    auto end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();

    RCLCPP_INFO(get_logger(), "Total time cost: %ld ms, InferenceOnce time cost: %ld ms.", total_duration, infer_duration);

    //cv::Mat depth = disparity2depth(disparity, left_ir_cam_info_k_->at(0) /* fx */, baseline_);
    PublishDisparity(left_ir_msg->header, sensor_msgs::image_encodings::TYPE_32FC1, disparity, disparity_image_pub_);

    //cv::Mat disp_vis = disparity2vis(disparity);
    //publishImage(left_ir_msg->header, sensor_msgs::image_encodings::BGR8, disp_vis, disparity_image_vis_pub_);

    return;
}

void signal_handler(int sig) {
    RCLCPP_WARN(rclcpp::get_logger("DnnStereoDepthNode"), "catch sig %d", sig);
    rclcpp::shutdown();
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    std::signal(SIGINT, signal_handler);

    
    rclcpp::spin(std::make_shared<DnnStereoDepthNode>());
    rclcpp::shutdown();
    return 0;
}