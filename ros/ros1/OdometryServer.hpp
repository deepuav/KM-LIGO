// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill Stachniss.
// Modified by Daehan Lee, Hyungtae Lim, and Soohee Han, 2024
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

// GenZ-ICP
#include "genz_icp/pipeline/GenZICP.hpp"
#include "genz_icp/core/Registration.hpp"

// ROS
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <nav_msgs/Odometry.h>

#include <string>
#include <deque>
#include <geometry_msgs/PoseStamped.h>

namespace genz_icp_ros {

// 类型别名定义
using PointCloud = std::vector<Eigen::Vector3d>;

struct PX4Pose {
    Sophus::SE3d pose;
    ros::Time timestamp;
    Eigen::Vector3d linear_velocity;
    Eigen::Vector3d angular_velocity;
};

class OdometryServer {
public:
    /// OdometryServer constructor
    OdometryServer(const ros::NodeHandle &nh, const ros::NodeHandle &pnh);

private:
    /// Register new frame
    void RegisterFrame(const sensor_msgs::PointCloud2::ConstPtr &msg);

    /// Stream the estimated pose to ROS
    void PublishOdometry(const Sophus::SE3d &pose,
                         const ros::Time &stamp,
                         const std::string &cloud_frame_id);

    /// Stream the debugging point clouds for visualization (if required)
    void PublishClouds(const ros::Time &stamp,
                       const std::string &cloud_frame_id,
                       const std::vector<Eigen::Vector3d> &planar_points,
                       const std::vector<Eigen::Vector3d> &non_planar_points);

    /// Utility function to compute transformation using tf tree
    Sophus::SE3d LookupTransform(const std::string &target_frame,
                                 const std::string &source_frame) const;

    /// Callback for PX4 local position odometry
    void PX4PoseCallback(const nav_msgs::Odometry::ConstPtr &msg);

    /// Find nearest poses for interpolation
    std::pair<PX4Pose, PX4Pose> FindNearestPoses(const ros::Time &target_time) const;

    /// Interpolate pose between two poses
    Sophus::SE3d InterpolatePose(const PX4Pose &pose1, const PX4Pose &pose2, const ros::Time &target_time) const;
    
    /// Calculate pose covariance based on GenZ-ICP quality metrics and comparison with PX4 pose
    void CalculateCovariance(const Sophus::SE3d &genz_pose, const Sophus::SE3d &mid_pose, 
                             const Sophus::SE3d &last_mid_pose, double dt,
                             const ros::Time &frame_start_time,
                             Eigen::Matrix<double, 6, 6> &pose_covariance);

    /// Ros node stuff
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    int queue_size_{1};

    /// Tools for broadcasting TFs.
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    tf2_ros::Buffer tf2_buffer_;
    tf2_ros::TransformListener tf2_listener_;
    bool publish_odom_tf_;
    bool publish_debug_clouds_;

    /// Data subscribers.
    ros::Subscriber pointcloud_sub_;
    ros::Subscriber px4_pose_sub_;

    /// Data publishers.
    ros::Publisher odom_publisher_;
    ros::Publisher map_publisher_;
    ros::Publisher traj_publisher_;
    ros::Publisher planar_points_publisher_;
    ros::Publisher non_planar_points_publisher_;
    ros::Publisher odometry_out_publisher_;
    ros::Publisher px4_pose_publisher_;
    nav_msgs::Path path_msg_;

    /// GenZ-ICP
    genz_icp::pipeline::GenZICP odometry_;
    genz_icp::pipeline::GenZConfig config_;

    /// Global/map coordinate frame.
    std::string odom_frame_{"odom"};
    std::string base_frame_{};
    std::string lidar_frame_{"rslidar"};

    /// PX4 pose queue
    std::deque<PX4Pose> px4_pose_queue_;
    static constexpr size_t MAX_PX4_POSE_QUEUE_SIZE = 200;
    static constexpr size_t MIN_PX4_POSE_QUEUE_SIZE = 30;
    
    /// Last mid pose
    Sophus::SE3d last_mid_pose_;
    ros::Time last_mid_time_;
    bool has_last_mid_pose_ = false;

    // 用于记录最后发布的变换
    ros::Time last_transform_time_;
    Sophus::SE3d last_transform_ = Sophus::SE3d();

    // 从点云消息中提取时间戳
    std::vector<double> GetTimestamps(const sensor_msgs::PointCloud2::ConstPtr &msg) const;
    
    // 检查点云帧的有效性
    bool CheckFrameValidity(const PointCloud& frame);
    
    // 评估配准质量
    bool EvaluateRegistrationQuality(const std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>& registration_result);
    
    // 配准质量相关标志和阈值
    bool use_px4_pose_next_frame_ = false;
    double min_inlier_ratio_threshold_ = 0.7;  // 内点比例阈值
    double max_convergence_error_threshold_ = 0.05;  // 收敛误差阈值
    int max_iteration_count_threshold_ = 100;  // 最大迭代次数阈值
};

}  // namespace genz_icp_ros
