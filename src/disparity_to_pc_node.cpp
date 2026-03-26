#include <memory>
#include <string>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include <rclcpp/rclcpp.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <depth_image_proc/depth_traits.hpp>

#include "depth_processor.hpp"

namespace {

template <typename T>
void convertDepthToPointcloud(const cv::Mat& depth_image,
                  sensor_msgs::msg::PointCloud2& cloud_msg,
                  const sensor_msgs::msg::CameraInfo& cam_info,
                  double invalid_depth = 0.0) {
    // Use correct principal point from calibration
    float center_x = cam_info.k[2]; // cx
    float center_y = cam_info.k[5]; // cy

    // Combine unit conversion (if necessary) with scaling by focal length for computing (X,Y)
    double unit_scaling = depth_image_proc::DepthTraits<T>::toMeters(T(1));
    float constant_x = unit_scaling / cam_info.k[0]; // fx
    float constant_y = unit_scaling / cam_info.k[4]; // fy
    float bad_point = std::numeric_limits<float>::quiet_NaN();

    // ensure that the computation only happens in case we have a default depth
    T invalid_depth_cvt = T(0);
    bool use_invalid_depth = invalid_depth != 0.0;
    if (use_invalid_depth) {
        invalid_depth_cvt = depth_image_proc::DepthTraits<T>::fromMeters(invalid_depth);
    }
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

    for (int v = 0; v < depth_image.rows; ++v) {
        for (int u = 0; u < depth_image.cols; ++u, ++iter_x, ++iter_y, ++iter_z) {

            T depth = depth_image.at<T>(v, u);

            // Missing points denoted by NaNs
            if (!depth_image_proc::DepthTraits<T>::valid(depth)) {
                if (use_invalid_depth) {
                    depth = invalid_depth_cvt;
                } else {
                    *iter_x = *iter_y = *iter_z = bad_point;
                    continue;
                }
            }

            // Fill in XYZ
            *iter_x = (u - center_x) * depth * constant_x;
            *iter_y = (v - center_y) * depth * constant_y;
            *iter_z = depth_image_proc::DepthTraits<T>::toMeters(depth);
        }
    }
}

}  // namespace

class DisparityToPointcloudNode : public rclcpp::Node {
   public:
    DisparityToPointcloudNode() : Node("disparity_to_pc_node") {
        // 1. Declare and get parameters
        // Default values set to 0.0 to force user to provide them or to detect unconfigured state
        this->declare_parameter("baseline", 0.095);  // Meters
        baseline_ = this->get_parameter("baseline").as_double();

        this->declare_parameter("zfar", 10.0);  // Meters
        zfar_ = this->get_parameter("zfar").as_double();

        this->declare_parameter("zmin", 0.1);  // Meters
        zmin_ = this->get_parameter("zmin").as_double();

        // 2. Create Publisher for Depth Map (32-bit Float)
        depth_pub_ =
            this->create_publisher<sensor_msgs::msg::Image>("depth_image", 10);
        pointcloud_pub_ =
            this->create_publisher<sensor_msgs::msg::PointCloud2>("points", 10);

        // 3. Create Subscriber for Disparity
        disparity_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "disparity", 10,
            std::bind(&DisparityToPointcloudNode::disparity_callback, this,
                      std::placeholders::_1));

        cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "camera_info", 10,
            std::bind(&DisparityToPointcloudNode::cam_info_callback, this,
                      std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(),
                    "DisparityToPointcloudNode Node Initialized.");
    }

   private:
    void cam_info_callback(
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) {
        if (cam_info_ != nullptr) {
            return;
        }

        cam_info_ = msg;
    }

    void disparity_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        if (cam_info_ == nullptr) {
            return;
        }

        double focal_length = cam_info_->k[0];  // fx

        if (baseline_ <= 0) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Invalid baseline (%.2f) or focal_length "
                                 "(%.2f). Please set parameters.",
                                 baseline_, focal_length);
            return;
        }

        // sensor_msgs::msg::Image::SharedPtr depth_msg;
        cv::Mat depth_mat;

        try {
            // 1. Convert ROS Disparity Image to OpenCV Mat
            // Disparity images are typically 32-bit float (32FC1)
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(
                msg, sensor_msgs::image_encodings::TYPE_32FC1);
            cv::Mat disparity_mat = cv_ptr->image;

            // 2. Calculate Depth
            // Formula: Z = (f * B) / d
            double numerator = focal_length * baseline_;

            // cv::Mat depth_mat;

            // We use cv::divide to handle the matrix operation efficiently.
            // Note: Division by zero (disparity = 0) will result in Inf, which is standard for infinite depth.
            // However, to make it safe for visualization or processing, we often handle 0 disparity.
            cv::divide(numerator, disparity_mat, depth_mat);

            // Optional: Filter out infinite or negative values (depending on use case)
            // Here we replace Inf with 0.0 (unknown depth) for safety
            cv::patchNaNs(depth_mat, 0.0);

            // Mask out values where disparity was <= 0 (invalid)
            // (This step is optional depending on if you want Inf or 0 for sky/far objects)
            cv::Mat invalid_mask = (disparity_mat <= 0);
            depth_mat.setTo(0.0, invalid_mask);

            depth_processor_.filter_zmin_zfar(depth_mat, zmin_, zfar_);
            depth_mat = depth_processor_.filter_edge_flying_pixels(depth_mat, zmin_);

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s",
                         e.what());
        }

        auto depth_msg_future = std::async(std::launch::async, [this, &msg, &depth_mat]() {

            // Publish Depth Image
            cv_bridge::CvImage cv_depth;
            cv_depth.header = msg->header;  // Keep time stamp and frame_id
            cv_depth.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            cv_depth.image = depth_mat;

            depth_pub_->publish(*cv_depth.toImageMsg());

        });

        auto pc_future = std::async(std::launch::async, [this, &msg, &depth_mat]() {

            // Publish Pointcloud
            auto cloud_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
            cloud_msg->header = msg->header;
            cloud_msg->height = msg->height;
            cloud_msg->width = msg->width;
            cloud_msg->is_dense = false;
            cloud_msg->is_bigendian = false;

            sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
            pcd_modifier.setPointCloud2FieldsByString(1, "xyz");

            convertDepthToPointcloud<float>(depth_mat, *cloud_msg, *cam_info_);

            pointcloud_pub_->publish(std::move(cloud_msg));

        });

        depth_msg_future.get();
        pc_future.get();
    }

    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr disparity_sub_;

    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>>
        pointcloud_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> depth_pub_;

    double baseline_;
    double zfar_;
    double zmin_;

    DepthProcessor depth_processor_;

    sensor_msgs::msg::CameraInfo::ConstSharedPtr cam_info_;
};

void signal_handler(int sig) {
    RCLCPP_WARN(rclcpp::get_logger("DisparityToDepthNode"), "catch sig %d",
                sig);
    rclcpp::shutdown();
}

int main(int argc, char** argv) {
    std::signal(SIGINT, signal_handler);

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DisparityToPointcloudNode>());
    rclcpp::shutdown();
    return 0;
}
