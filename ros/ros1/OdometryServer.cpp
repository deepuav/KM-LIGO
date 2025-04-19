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
#include <Eigen/Core>
#include <memory>
#include <utility>
#include <vector>
#include <algorithm>

// GenZ-ICP-ROS
#include "OdometryServer.hpp"
#include "Utils.hpp"

// GenZ-ICP
#include "genz_icp/pipeline/GenZICP.hpp"

// ROS 1 headers
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/init.h>
#include <ros/node_handle.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

namespace genz_icp_ros {

using utils::EigenToPointCloud2;
using utils::GetTimestamps;
using utils::PointCloud2ToEigen;

OdometryServer::OdometryServer(const ros::NodeHandle &nh, const ros::NodeHandle &pnh)
    : nh_(nh), pnh_(pnh), tf2_listener_(tf2_ros::TransformListener(tf2_buffer_)) {
    pnh_.param("base_frame", base_frame_, base_frame_);
    pnh_.param("odom_frame", odom_frame_, odom_frame_);
    pnh_.param("lidar_frame", lidar_frame_, lidar_frame_);
    pnh_.param("publish_odom_tf", publish_odom_tf_, false);
    pnh_.param("visualize", publish_debug_clouds_, publish_debug_clouds_);
    pnh_.param("max_range", config_.max_range, config_.max_range);
    pnh_.param("min_range", config_.min_range, config_.min_range);
    pnh_.param("deskew", config_.deskew, config_.deskew);
    pnh_.param("use_px4_pose_for_deskew", config_.use_px4_pose_for_deskew, config_.use_px4_pose_for_deskew);
    pnh_.param("use_multiple_poses_for_deskew", config_.use_multiple_poses_for_deskew, config_.use_multiple_poses_for_deskew);
    pnh_.param("use_px4_pose_for_init", config_.use_px4_pose_for_init, config_.use_px4_pose_for_init);
    pnh_.param("use_px4_attitude_with_original_position_for_init", config_.use_px4_attitude_with_original_position_for_init, config_.use_px4_attitude_with_original_position_for_init);
    pnh_.param("use_px4_pose_for_map", config_.use_px4_pose_for_map, config_.use_px4_pose_for_map);
    pnh_.param("publish_odom_to_px4", config_.publish_odom_to_px4, config_.publish_odom_to_px4);
    
    // 配准质量评估参数
    pnh_.param("min_inlier_ratio_threshold", min_inlier_ratio_threshold_, 0.7);
    pnh_.param("max_convergence_error_threshold", max_convergence_error_threshold_, 0.05);
    pnh_.param("max_iteration_count_threshold", max_iteration_count_threshold_, 100);
    
    pnh_.param("voxel_size", config_.voxel_size, config_.max_range / 100.0);
    pnh_.param("map_cleanup_radius", config_.map_cleanup_radius, config_.max_range);
    pnh_.param("planarity_threshold", config_.planarity_threshold, config_.planarity_threshold);
    pnh_.param("max_points_per_voxel", config_.max_points_per_voxel, config_.max_points_per_voxel);
    pnh_.param("desired_num_voxelized_points", config_.desired_num_voxelized_points, config_.desired_num_voxelized_points);
    pnh_.param("initial_threshold", config_.initial_threshold, config_.initial_threshold);
    pnh_.param("min_motion_th", config_.min_motion_th, config_.min_motion_th);
    pnh_.param("max_num_iterations", config_.max_num_iterations, config_.max_num_iterations);
    pnh_.param("convergence_criterion", config_.convergence_criterion, config_.convergence_criterion);
    if (config_.max_range < config_.min_range) {
        ROS_WARN("[WARNING] max_range is smaller than min_range, setting min_range to 0.0");
        config_.min_range = 0.0;
    }

    // Construct the main GenZ-ICP odometry node
    odometry_ = genz_icp::pipeline::GenZICP(config_);

    // Initialize subscribers
    pointcloud_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>("pointcloud_topic", queue_size_,
                                                              &OdometryServer::RegisterFrame, this);
    px4_pose_sub_ = nh_.subscribe<nav_msgs::Odometry>("/mavros/odometry/in", queue_size_,
                                                      &OdometryServer::PX4PoseCallback, this);

    // Initialize publishers
    odom_publisher_ = pnh_.advertise<nav_msgs::Odometry>("/genz/odometry", queue_size_);
    traj_publisher_ = pnh_.advertise<nav_msgs::Path>("/genz/trajectory", queue_size_);
    odometry_out_publisher_ = pnh_.advertise<nav_msgs::Odometry>("/mavros/odometry/out", queue_size_);
    px4_pose_publisher_ = pnh_.advertise<geometry_msgs::PoseStamped>("/genz/px4_pose", queue_size_);
    if (publish_debug_clouds_) {
        map_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("/genz/local_map", queue_size_);
        planar_points_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("/genz/planar_points", queue_size_);
        non_planar_points_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("/genz/non_planar_points", queue_size_);
    }
    // Initialize the transform buffer
    tf2_buffer_.setUsingDedicatedThread(true);
    path_msg_.header.frame_id = odom_frame_;

    // publish odometry msg
    ROS_INFO("GenZ-ICP ROS 1 Odometry Node Initialized");
    ROS_INFO("GENZ odom frame: %s", odom_frame_.c_str());
    ROS_INFO("Base frame: %s", base_frame_.c_str());
    ROS_INFO("LiDAR frame: %s", lidar_frame_.c_str());
}

void OdometryServer::PX4PoseCallback(const nav_msgs::Odometry::ConstPtr &msg) {
    PX4Pose px4_pose;
    px4_pose.timestamp = msg->header.stamp;
    
    // 直接使用MAVROS提供的ENU坐标系位姿，不需要转换
    Eigen::Vector3d translation(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    Eigen::Quaterniond rotation(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x,
                               msg->pose.pose.orientation.y, msg->pose.pose.orientation.z);
    px4_pose.pose = Sophus::SE3d(rotation, translation);
    
    // 直接使用MAVROS提供的ENU坐标系速度，不需要转换
    px4_pose.linear_velocity = Eigen::Vector3d(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
    px4_pose.angular_velocity = Eigen::Vector3d(msg->twist.twist.angular.x, msg->twist.twist.angular.y, msg->twist.twist.angular.z);

    // 发布PX4位姿用于调试，进行坐标系转换使其与其他坐标系对齐
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header = msg->header;
    
    // 应用相同的坐标系转换（绕Z轴旋转-90度，与MAVROS发布转换相反）
    Eigen::Matrix3d rot_z;
    rot_z = Eigen::AngleAxisd(-M_PI/2, Eigen::Vector3d::UnitZ());
    Sophus::SE3d aligned_pose;
    aligned_pose.so3() = Sophus::SO3d(rot_z) * px4_pose.pose.so3();
    aligned_pose.translation() = rot_z * px4_pose.pose.translation();
    
    // 转换为geometry_msgs::Pose
    pose_msg.pose.position.x = aligned_pose.translation().x();
    pose_msg.pose.position.y = aligned_pose.translation().y();
    pose_msg.pose.position.z = aligned_pose.translation().z();
    
    Eigen::Quaterniond q(aligned_pose.rotationMatrix());
    pose_msg.pose.orientation.w = q.w();
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    
    px4_pose_publisher_.publish(pose_msg);

    // Add to queue and maintain size limit
    px4_pose_queue_.push_back(px4_pose);
    if (px4_pose_queue_.size() > MAX_PX4_POSE_QUEUE_SIZE) {
        px4_pose_queue_.pop_front();
    }
}

std::pair<PX4Pose, PX4Pose> OdometryServer::FindNearestPoses(const ros::Time &target_time) const {
    if (px4_pose_queue_.size() < 2) {
        ROS_WARN("Not enough poses in queue for interpolation");
        return {px4_pose_queue_.front(), px4_pose_queue_.front()};
    }

    // Binary search for the closest pose
    auto it = std::lower_bound(px4_pose_queue_.begin(), px4_pose_queue_.end(), target_time,
                              [](const PX4Pose &pose, const ros::Time &time) {
                                  return pose.timestamp < time;
                              });

    // Handle edge cases
    if (it == px4_pose_queue_.begin()) {
        return {px4_pose_queue_.front(), *(px4_pose_queue_.begin() + 1)};
    }
    if (it == px4_pose_queue_.end()) {
        return {*(px4_pose_queue_.end() - 2), px4_pose_queue_.back()};
    }

    // Return the pair of poses that bracket the target time
    return {*(it - 1), *it};
}

Sophus::SE3d OdometryServer::InterpolatePose(const PX4Pose &pose1, const PX4Pose &pose2,
                                             const ros::Time &target_time) const {
    double total_duration = (pose2.timestamp - pose1.timestamp).toSec();
    double target_duration = (target_time - pose1.timestamp).toSec();
    double alpha = target_duration / total_duration;

    // Interpolate translation
    Eigen::Vector3d translation = (1 - alpha) * pose1.pose.translation() + alpha * pose2.pose.translation();

    // Interpolate rotation using SLERP
    Eigen::Quaterniond q1 = pose1.pose.so3().unit_quaternion();
    Eigen::Quaterniond q2 = pose2.pose.so3().unit_quaternion();
    Eigen::Quaterniond q_interp = q1.slerp(alpha, q2);

    return Sophus::SE3d(q_interp, translation);
}

std::vector<double> OdometryServer::GetTimestamps(const sensor_msgs::PointCloud2::ConstPtr &msg) const {
    // 直接使用utils中的GetTimestamps函数
    return utils::GetTimestamps(msg);
}

void OdometryServer::CalculateCovariance(const Sophus::SE3d &genz_pose, const Sophus::SE3d &mid_pose, 
                                        const Sophus::SE3d &last_mid_pose, double dt,
                                        const ros::Time &frame_start_time,
                                        Eigen::Matrix<double, 6, 6> &pose_covariance) {
    // 初始化位姿协方差矩阵
    pose_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
    
    // 1. 只对姿态部分使用GenZ-ICP位姿与PX4位姿的比较
    Sophus::SE3d pose_diff = genz_pose.inverse() * mid_pose;
    Eigen::Matrix<double, 3, 1> rotation_diff_vec = pose_diff.so3().log();
    
    // 只设置姿态协方差（索引3-5）- 基于与PX4姿态的比较
    for (int i = 0; i < 3; ++i) {
        pose_covariance(i+3, i+3) = rotation_diff_vec(i) * rotation_diff_vec(i);
        if (pose_covariance(i+3, i+3) < 1e-6) {
            pose_covariance(i+3, i+3) = 1e-6;  // 设置最小协方差
        }
    }
    
    // 2. 计算基于配准精度的协方差
    // 获取配准质量指标
    double convergence_error = odometry_.GetLastConvergenceError();
    int iteration_count = odometry_.GetLastIterationCount();
    double inlier_ratio = odometry_.GetLastInlierRatio();
    
    // 基于配准精度计算位置协方差（索引0-2）
    double position_quality = convergence_error;  // 收敛误差越大，协方差越大
    position_quality += std::max(0.0, 1.0 - inlier_ratio) * 0.1;  // 内点比例越低，协方差越大
    position_quality += std::max(0.0, (double)(iteration_count - 50) / 50.0) * 0.05;  // 迭代次数越多，协方差越大
    
    // 确保最小值
    position_quality = std::max(position_quality, 0.001);
    
    // 设置位置协方差（仅使用配准精度）
    for (int i = 0; i < 3; ++i) {
        pose_covariance(i, i) = position_quality;
    }
    
    // 3. 混合计算姿态协方差（基于PX4姿态比较和配准精度）
    // 计算配准精度影响因子
    double rotation_quality = convergence_error * 2.0;  // 姿态对收敛误差更敏感
    rotation_quality += std::max(0.0, 1.0 - inlier_ratio) * 0.2;  // 内点比例越低，协方差越大
    rotation_quality += std::max(0.0, (double)(iteration_count - 50) / 50.0) * 0.1;  // 迭代次数越多，协方差越大
    
    // 确保最小值
    rotation_quality = std::max(rotation_quality, 0.002);
    
    // 混合姿态协方差（已经设置了基于PX4比较的部分）
    // 添加配准精度影响
    for (int i = 3; i < 6; ++i) {
        pose_covariance(i, i) = pose_covariance(i, i) * 0.6 + rotation_quality * 0.4;
    }
    
    // 在Z轴旋转（偏航）上增加协方差，因为它通常更不确定
    pose_covariance(5, 5) *= 1.5;
}

void OdometryServer::RegisterFrame(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    // Check if we have enough poses in the queue
    if (px4_pose_queue_.size() < MIN_PX4_POSE_QUEUE_SIZE) {
        ROS_WARN_THROTTLE(1.0, "等待足够的PX4位姿（当前：%zu，需要：%zu）",
                         px4_pose_queue_.size(), MIN_PX4_POSE_QUEUE_SIZE);
        return;
    }

    // 使用配置的激光雷达坐标系而不是消息中的坐标系
    const auto cloud_frame_id = lidar_frame_;
    const auto points = PointCloud2ToEigen(msg);
    const auto timestamps = [&]() -> std::vector<double> {
        if (!config_.deskew) return {};
        return GetTimestamps(msg);
    }();

    // 获取点云帧的起始和结束时间
    ros::Time frame_start_time = msg->header.stamp;
    // 使用实际的激光雷达扫描周期，而不是固定值
    double scan_duration = 0.1;  // 默认值，可从参数服务器读取
    pnh_.param("scan_duration", scan_duration, 0.1);
    
    ros::Time frame_end_time = frame_start_time + ros::Duration(scan_duration);
    ros::Time frame_mid_time = frame_start_time + ros::Duration(scan_duration/2.0);
    
    // 提取在帧开始和结束时间之间的所有PX4位姿
    std::vector<Sophus::SE3d> px4_poses;
    std::vector<double> px4_pose_times;
    
    // 确保至少有起始位姿和结束位姿
    auto [start_pose_pair1, start_pose_pair2] = FindNearestPoses(frame_start_time);
    auto [end_pose_pair1, end_pose_pair2] = FindNearestPoses(frame_end_time);
    auto [mid_pose_pair1, mid_pose_pair2] = FindNearestPoses(frame_mid_time);
    
    // 插值计算精确的起始和结束位姿
    Sophus::SE3d start_px4_pose = InterpolatePose(start_pose_pair1, start_pose_pair2, frame_start_time);
    Sophus::SE3d end_px4_pose = InterpolatePose(end_pose_pair1, end_pose_pair2, frame_end_time);
    Sophus::SE3d mid_px4_pose = InterpolatePose(mid_pose_pair1, mid_pose_pair2, frame_mid_time);
    
    // 查找帧时间段内的所有PX4位姿
    for (const auto& px4_pose : px4_pose_queue_) {
        if (px4_pose.timestamp >= frame_start_time && px4_pose.timestamp <= frame_end_time) {
            px4_poses.push_back(px4_pose.pose);
            // 将时间转换为相对于帧起始时间的秒数
            px4_pose_times.push_back((px4_pose.timestamp - frame_start_time).toSec());
        }
    }
    
    // 确保位姿数组中包含起始和结束位姿
    // 如果数组为空或第一个位姿晚于frame_start_time，添加起始位姿
    if (px4_poses.empty() || px4_pose_times.front() > 0.001) { // 给定一个小的阈值以处理浮点误差
        px4_poses.insert(px4_poses.begin(), start_px4_pose);
        px4_pose_times.insert(px4_pose_times.begin(), 0.0);
    }
    
    // 确保数组包含结束位姿
    if (px4_pose_times.back() < scan_duration - 0.001) {
        px4_poses.push_back(end_px4_pose);
        px4_pose_times.push_back(scan_duration);
    }
    
    // ===== 点云去畸变和初始位姿猜测策略 =====
    // 初始位姿选择逻辑：
    // 1. 如果配准质量良好：使用GenZ-ICP内部预测（最优）
    // 2. 如果配准质量低下：
    //    a. 优先使用PX4位姿（如果config_.use_px4_pose_for_init为true）
    //    b. 或者使用PX4姿态+预测位移（如果config_.use_px4_attitude_with_original_position_for_init为true）
    // 注：use_px4_pose_next_frame_标志表示上一帧配准质量低下
    std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> registration_result;
    
    // 首先决定是否进行点云去畸变处理
    const std::vector<Eigen::Vector3d>* points_to_register = &points;
    std::vector<Eigen::Vector3d> corrected_points;
    
    if (config_.use_px4_pose_for_deskew) {
        // 使用PX4位姿进行点云去畸变
        if (config_.use_multiple_poses_for_deskew && px4_poses.size() >= 2) {
            // 使用多个位姿进行更精确的去畸变
            corrected_points = genz_icp::DeSkewScanWithPoses(points, timestamps, px4_poses, px4_pose_times);
        } else {
            // 简单地使用原始点云
            corrected_points = points;
        }
        points_to_register = &corrected_points;
    }
    
    // 根据配准质量和配置选择初始位姿猜测策略
    bool using_external_initial_pose = false;
    if (config_.use_px4_pose_for_init && use_px4_pose_next_frame_) {
        // 策略1：使用PX4位姿作为初始值（在配准质量低下时）
        ROS_INFO("[GenZ-ICP] 使用PX4位姿作为初始值 (由于上一帧配准质量低)");
        registration_result = odometry_.RegisterFrame(*points_to_register, {}, 
                                                    start_px4_pose, end_px4_pose, mid_px4_pose);
        using_external_initial_pose = true;
    } 
    else if (config_.use_px4_attitude_with_original_position_for_init && use_px4_pose_next_frame_) {
        // 策略2：使用PX4姿态 + 预测位移作为初始值（在配准质量低下时）
        ROS_INFO("[GenZ-ICP] 使用PX4姿态与预测位移合成位姿作为初始值 (由于上一帧配准质量低)");
        
        // 获取GenZ-ICP预测的位姿
        const auto prediction = odometry_.GetPredictionModel();
        const auto last_pose = !odometry_.poses().empty() ? odometry_.poses().back() : Sophus::SE3d();
        const auto original_guess = last_pose * prediction;
        
        // 创建混合初始位姿：PX4的旋转 + 预测的平移
        Sophus::SE3d initial_guess = Sophus::SE3d(
            mid_px4_pose.unit_quaternion(),    // PX4姿态
            original_guess.translation()        // 预测位移
        );
        
        // 使用混合初始位姿进行配准
        registration_result = odometry_.RegisterFrame(*points_to_register, {}, 
                                                    initial_guess, initial_guess, initial_guess);
        using_external_initial_pose = true;
    } 
    else {
        // 策略3：使用GenZ-ICP内部位姿预测（默认/最优方法）
        if (!use_px4_pose_next_frame_) {
            ROS_DEBUG("[GenZ-ICP] 使用默认位姿预测方法（配准质量良好）");
        } else {
            ROS_DEBUG("[GenZ-ICP] 使用默认位姿预测方法（未启用外部位姿初始化）");
        }
        
        // 如果需要考虑时间戳
        if (config_.use_px4_pose_for_deskew) {
            registration_result = odometry_.RegisterFrame(*points_to_register, {});
        } else {
            registration_result = odometry_.RegisterFrame(points, timestamps);
        }
    }
    
    auto [registered_frame, registered_timestamps] = registration_result;

    // 获取GenZ-ICP计算的位姿
    const Sophus::SE3d genz_pose = odometry_.poses().back();

    // 计算协方差
    Eigen::Matrix<double, 6, 6> pose_covariance;
    if (has_last_mid_pose_) {
        double dt = (frame_mid_time - last_mid_time_).toSec();
        if (dt > 0) {
            CalculateCovariance(genz_pose, mid_px4_pose, last_mid_pose_, dt, frame_start_time, pose_covariance);
        } else {
            pose_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
        }
    } else {
        pose_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
    }

    // 更新last_mid_pose
    last_mid_pose_ = mid_px4_pose;
    last_mid_time_ = frame_mid_time;
    has_last_mid_pose_ = true;

    // 评估配准质量，设置下一帧是否使用px4位姿初始化
    bool low_quality_registration = EvaluateRegistrationQuality(registration_result);
    
    // 发布到/mavros/odometry/out话题
    // 仅在配准质量良好时向MAVROS发布里程计
    if (config_.publish_odom_to_px4 && !low_quality_registration) {
        nav_msgs::Odometry odometry_out_msg;
        odometry_out_msg.header.stamp = frame_mid_time;
        odometry_out_msg.header.frame_id = odom_frame_;
        odometry_out_msg.child_frame_id = base_frame_.empty() ? cloud_frame_id : base_frame_;
        
        // 进行坐标系转换，修正90度旋转问题
        Sophus::SE3d mavros_pose = genz_pose;
        
        // 如果发布到MAVROS，需要转换坐标系（ROS->NED转换）
        // PX4使用的是NED（North-East-Down）坐标系，而ROS通常使用ENU（East-North-Up）
        // 这需要绕Z轴旋转90度
        Eigen::Matrix3d rot_z;
        rot_z = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d::UnitZ());
        mavros_pose.so3() = Sophus::SO3d(rot_z) * genz_pose.so3();
        mavros_pose.translation() = rot_z * genz_pose.translation();
        
        odometry_out_msg.pose.pose = tf2::sophusToPose(mavros_pose);
        
        // 设置位姿协方差
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                odometry_out_msg.pose.covariance[i * 6 + j] = pose_covariance(i, j);
            }
        }
        
        odometry_out_publisher_.publish(odometry_out_msg);
    } else if (config_.publish_odom_to_px4 && low_quality_registration) {
        ROS_WARN_THROTTLE(1.0, "[GenZ-ICP] 因配准质量低，暂停向MAVROS发布里程计数据");
    }
    
    // 发布其他消息
    PublishOdometry(genz_pose, msg->header.stamp, cloud_frame_id);
    if (publish_debug_clouds_) {
        PublishClouds(msg->header.stamp, cloud_frame_id, registered_frame, registered_timestamps);
    }
    
    // 评估配准质量，设置下一帧是否使用px4位姿初始化
    use_px4_pose_next_frame_ = low_quality_registration;
}

Sophus::SE3d OdometryServer::LookupTransform(const std::string &target_frame,
                                             const std::string &source_frame) const {
    std::string err_msg;
    if (tf2_buffer_._frameExists(source_frame) &&  //
        tf2_buffer_._frameExists(target_frame) &&  //
        tf2_buffer_.canTransform(target_frame, source_frame, ros::Time(0), &err_msg)) {
        try {
            auto tf = tf2_buffer_.lookupTransform(target_frame, source_frame, ros::Time(0));
            return tf2::transformToSophus(tf);
        } catch (tf2::TransformException &ex) {
            ROS_WARN("%s", ex.what());
        }
    }
    ROS_WARN("Failed to find tf between %s and %s. Reason=%s", target_frame.c_str(),
             source_frame.c_str(), err_msg.c_str());
    return {};
}

void OdometryServer::PublishOdometry(const Sophus::SE3d &pose,
                                     const ros::Time &stamp,
                                     const std::string &cloud_frame_id) {
    // Header for point clouds and stuff seen from desired odom_frame

    // Broadcast the tf
    if (publish_odom_tf_) {
        // 仅发布odom到base_link的变换，避免与静态变换冲突
        geometry_msgs::TransformStamped transform_msg;
        transform_msg.header.stamp = stamp;
        transform_msg.header.frame_id = odom_frame_;
        transform_msg.child_frame_id = base_frame_.empty() ? cloud_frame_id : base_frame_;
        transform_msg.transform = tf2::sophusToTransform(pose);
        tf_broadcaster_.sendTransform(transform_msg);
        
        // 记录最新发布的变换时间和值，用于避免TF跳变
        last_transform_time_ = stamp;
        last_transform_ = pose;
    }

    // publish trajectory msg
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = odom_frame_;
    pose_msg.pose = tf2::sophusToPose(pose);
    path_msg_.poses.push_back(pose_msg);
    traj_publisher_.publish(path_msg_);

    // publish odometry msg
    nav_msgs::Odometry odom_msg;
    odom_msg.header.stamp = stamp;
    odom_msg.header.frame_id = odom_frame_;
    odom_msg.child_frame_id = base_frame_.empty() ? cloud_frame_id : base_frame_;
    odom_msg.pose.pose = tf2::sophusToPose(pose);
    odom_publisher_.publish(odom_msg);
}

void OdometryServer::PublishClouds(const ros::Time &stamp,
                                   const std::string &cloud_frame_id,
                                   const std::vector<Eigen::Vector3d> &planar_points,
                                   const std::vector<Eigen::Vector3d> &non_planar_points) {
    std_msgs::Header odom_header;
    odom_header.stamp = stamp;
    odom_header.frame_id = odom_frame_;

    // Publish map
    const auto genz_map = odometry_.LocalMap();

    if (!publish_odom_tf_) {
        // debugging happens in an egocentric world
        std_msgs::Header cloud_header;
        cloud_header.stamp = stamp;
        cloud_header.frame_id = lidar_frame_;

        map_publisher_.publish(*EigenToPointCloud2(genz_map, odom_header));
        planar_points_publisher_.publish(*EigenToPointCloud2(planar_points, cloud_header));
        non_planar_points_publisher_.publish(*EigenToPointCloud2(non_planar_points, cloud_header));

        return;
    }

    // If transmitting to tf tree we know where the clouds are exactly
    const auto cloud2odom = LookupTransform(odom_frame_, lidar_frame_);
    planar_points_publisher_.publish(*EigenToPointCloud2(planar_points, odom_header));
    non_planar_points_publisher_.publish(*EigenToPointCloud2(non_planar_points, odom_header));

    if (!base_frame_.empty()) {
        const Sophus::SE3d cloud2base = LookupTransform(base_frame_, lidar_frame_);
        map_publisher_.publish(*EigenToPointCloud2(genz_map, cloud2base, odom_header));
    } else {
        map_publisher_.publish(*EigenToPointCloud2(genz_map, odom_header));
    }
}

bool OdometryServer::EvaluateRegistrationQuality(
    const std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>& registration_result) {
  const double convergence_error = odometry_.GetLastConvergenceError();
  const int iterations = odometry_.GetLastIterationCount();
  const double inlier_ratio = odometry_.GetLastInlierRatio();
  
  // 使用评分机制而不是单一条件触发
  int quality_score = 0;
  
  // 收敛误差评分（0-3分）
  if (convergence_error > max_convergence_error_threshold_ * 1.5) {
    quality_score += 3;  // 严重超出阈值
  } else if (convergence_error > max_convergence_error_threshold_ * 1.2) {
    quality_score += 2;  // 明显超出阈值
  } else if (convergence_error > max_convergence_error_threshold_) {
    quality_score += 1;  // 轻微超出阈值
  }
  
  // 迭代次数评分（0-3分）
  if (iterations >= max_iteration_count_threshold_ * 1.2) {
    quality_score += 3;  // 严重超出阈值
  } else if (iterations >= max_iteration_count_threshold_ * 1.1) {
    quality_score += 2;  // 明显超出阈值
  } else if (iterations >= max_iteration_count_threshold_) {
    quality_score += 1;  // 轻微超出阈值
  }
  
  // 内点比例评分（0-3分）
  if (inlier_ratio < min_inlier_ratio_threshold_ * 0.7) {
    quality_score += 3;  // 严重低于阈值
  } else if (inlier_ratio < min_inlier_ratio_threshold_ * 0.85) {
    quality_score += 2;  // 明显低于阈值
  } else if (inlier_ratio < min_inlier_ratio_threshold_) {
    quality_score += 1;  // 轻微低于阈值
  }
  
  // 总分大于等于4分才认为是低质量配准
  const bool low_quality_registration = quality_score >= 4;
  
  // 记录当前配准质量详情（仅在分数变化时记录）
  static int last_quality_score = 0;
  if (quality_score != last_quality_score) {
    if (low_quality_registration) {
      ROS_WARN_STREAM("[GenZ-ICP] 配准质量评分：" << quality_score << "/9 (>=4表示低质量), "
                     << "收敛误差=" << convergence_error
                     << ", 迭代次数=" << iterations 
                     << ", 内点比例=" << inlier_ratio);
    } else {
      ROS_DEBUG_STREAM("[GenZ-ICP] 配准质量评分：" << quality_score << "/9 (<4表示高质量), "
                      << "收敛误差=" << convergence_error
                      << ", 迭代次数=" << iterations 
                      << ", 内点比例=" << inlier_ratio);
    }
    last_quality_score = quality_score;
  }
  
  // 如果当前注册质量很低但之前的注册质量是好的
  if (low_quality_registration && !use_px4_pose_next_frame_) {
    // 将标志位设置为true，下一帧将使用PX4位姿初始化
    use_px4_pose_next_frame_ = true;
    
    ROS_WARN_STREAM("[GenZ-ICP] 配准质量下降，检测到低质量配准: "
                   << "评分=" << quality_score << "/9, "
                   << "收敛误差=" << convergence_error
                   << ", 迭代次数=" << iterations 
                   << ", 内点比例=" << inlier_ratio
                   << "，将在下一帧使用PX4位姿初始化");
  } 
  // 如果当前注册质量良好但之前使用了PX4位姿初始化
  else if (!low_quality_registration && use_px4_pose_next_frame_) {
    // 配准质量已恢复，将标志位设置为false，恢复使用正常初始化方法
    use_px4_pose_next_frame_ = false;
    
    ROS_INFO_STREAM("[GenZ-ICP] 配准质量已恢复: "
                   << "评分=" << quality_score << "/9, "
                   << "收敛误差=" << convergence_error
                   << ", 迭代次数=" << iterations 
                   << ", 内点比例=" << inlier_ratio
                   << "，将恢复使用正常位姿初始化");
  }
  
  return low_quality_registration;
}

}  // namespace genz_icp_ros

int main(int argc, char **argv) {
    ros::init(argc, argv, "genz_icp");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    genz_icp_ros::OdometryServer node(nh, nh_private);

    ros::spin();

    return 0;
}

