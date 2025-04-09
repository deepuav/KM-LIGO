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
    pnh_.param("publish_odom_tf", publish_odom_tf_, false);
    pnh_.param("visualize", publish_debug_clouds_, publish_debug_clouds_);
    pnh_.param("max_range", config_.max_range, config_.max_range);
    pnh_.param("min_range", config_.min_range, config_.min_range);
    pnh_.param("deskew", config_.deskew, config_.deskew);
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
    px4_pose_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>("/mavros/local_position/pose", queue_size_,
                                                              &OdometryServer::PX4PoseCallback, this);

    // Initialize publishers
    odom_publisher_ = pnh_.advertise<nav_msgs::Odometry>("/genz/odometry", queue_size_);
    traj_publisher_ = pnh_.advertise<nav_msgs::Path>("/genz/trajectory", queue_size_);
    odometry_out_publisher_ = pnh_.advertise<nav_msgs::Odometry>("/mavros/odometry/out", queue_size_);
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
}

void OdometryServer::PX4PoseCallback(const geometry_msgs::PoseStamped::ConstPtr &msg) {
    PX4Pose px4_pose;
    px4_pose.timestamp = msg->header.stamp;
    
    // Convert ROS Pose to Sophus::SE3d
    Eigen::Vector3d translation(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    Eigen::Quaterniond rotation(msg->pose.orientation.w, msg->pose.orientation.x, 
                               msg->pose.orientation.y, msg->pose.orientation.z);
    px4_pose.pose = Sophus::SE3d(rotation, translation);

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

void OdometryServer::CalculateCovariance(const Sophus::SE3d &genz_pose, const Sophus::SE3d &mid_pose, 
                                        const Sophus::SE3d &last_mid_pose, double dt,
                                        Eigen::Matrix<double, 6, 6> &pose_covariance,
                                        Eigen::Matrix<double, 6, 6> &velocity_covariance) {
    // 计算位姿差异，作为位姿协方差的基础
    Sophus::SE3d pose_diff = genz_pose.inverse() * mid_pose;
    Eigen::Matrix<double, 6, 1> pose_diff_vec;
    pose_diff_vec.block<3, 1>(0, 0) = pose_diff.translation();
    pose_diff_vec.block<3, 1>(3, 0) = pose_diff.so3().log();

    // 计算位姿协方差
    pose_covariance = Eigen::Matrix<double, 6, 6>::Identity();
    for (int i = 0; i < 6; ++i) {
        pose_covariance(i, i) = pose_diff_vec(i) * pose_diff_vec(i);
        if (pose_covariance(i, i) < 1e-6) {
            pose_covariance(i, i) = 1e-6;  // 设置最小协方差
        }
    }

    // 计算GenZ的速度
    Sophus::SE3d genz_vel_pose;
    if (odometry_.poses().size() >= 2) {
        genz_vel_pose = odometry_.poses()[odometry_.poses().size() - 2].inverse() * odometry_.poses().back();
    } else {
        genz_vel_pose = Sophus::SE3d();
    }
    
    Eigen::Matrix<double, 6, 1> genz_vel;
    genz_vel.block<3, 1>(0, 0) = genz_vel_pose.translation() / dt;
    genz_vel.block<3, 1>(3, 0) = genz_vel_pose.so3().log() / dt;

    // 计算PX4的速度
    Sophus::SE3d px4_vel_pose = last_mid_pose.inverse() * mid_pose;
    Eigen::Matrix<double, 6, 1> px4_vel;
    px4_vel.block<3, 1>(0, 0) = px4_vel_pose.translation() / dt;
    px4_vel.block<3, 1>(3, 0) = px4_vel_pose.so3().log() / dt;

    // 计算速度差异，作为速度协方差的基础
    Eigen::Matrix<double, 6, 1> vel_diff = genz_vel - px4_vel;

    // 计算速度协方差
    velocity_covariance = Eigen::Matrix<double, 6, 6>::Identity();
    for (int i = 0; i < 6; ++i) {
        velocity_covariance(i, i) = vel_diff(i) * vel_diff(i);
        if (velocity_covariance(i, i) < 1e-6) {
            velocity_covariance(i, i) = 1e-6;  // 设置最小协方差
        }
    }
}

void OdometryServer::RegisterFrame(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    // Check if we have enough poses in the queue
    if (px4_pose_queue_.size() < MIN_PX4_POSE_QUEUE_SIZE) {
        ROS_WARN_THROTTLE(1.0, "等待足够的PX4位姿（当前：%zu，需要：%zu）",
                         px4_pose_queue_.size(), MIN_PX4_POSE_QUEUE_SIZE);
        return;
    }

    const auto cloud_frame_id = msg->header.frame_id;
    const auto points = PointCloud2ToEigen(msg);
    const auto timestamps = [&]() -> std::vector<double> {
        if (!config_.deskew) return {};
        return GetTimestamps(msg);
    }();

    // 获取点云帧的起始和结束时间
    ros::Time frame_start_time = msg->header.stamp;
    ros::Time frame_end_time = frame_start_time + ros::Duration(0.1);  // 点云帧持续0.1秒
    ros::Time frame_mid_time = frame_start_time + ros::Duration(0.05); // 中间时间点
    
    // 获取对应时间的PX4位姿
    auto [start_pose_pair1, start_pose_pair2] = FindNearestPoses(frame_start_time);
    auto [end_pose_pair1, end_pose_pair2] = FindNearestPoses(frame_end_time);
    auto [mid_pose_pair1, mid_pose_pair2] = FindNearestPoses(frame_mid_time);
    
    // 插值计算精确的位姿
    Sophus::SE3d start_px4_pose = InterpolatePose(start_pose_pair1, start_pose_pair2, frame_start_time);
    Sophus::SE3d end_px4_pose = InterpolatePose(end_pose_pair1, end_pose_pair2, frame_end_time);
    Sophus::SE3d mid_px4_pose = InterpolatePose(mid_pose_pair1, mid_pose_pair2, frame_mid_time);

    // 使用外部位姿进行去畸变和初始位姿估计
    const auto &[planar_points, non_planar_points] = odometry_.RegisterFrame(points, timestamps, start_px4_pose, end_px4_pose);

    // 获取GenZ-ICP计算的位姿
    const Sophus::SE3d genz_pose = odometry_.poses().back();

    // 计算协方差
    Eigen::Matrix<double, 6, 6> pose_covariance, velocity_covariance;
    if (has_last_mid_pose_) {
        double dt = (frame_mid_time - last_mid_time_).toSec();
        if (dt > 0) {
            CalculateCovariance(genz_pose, mid_px4_pose, last_mid_pose_, dt, pose_covariance, velocity_covariance);
        } else {
            pose_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
            velocity_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
        }
    } else {
        pose_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
        velocity_covariance = Eigen::Matrix<double, 6, 6>::Identity() * 1e-6;
    }

    // 更新last_mid_pose
    last_mid_pose_ = mid_px4_pose;
    last_mid_time_ = frame_mid_time;
    has_last_mid_pose_ = true;

    // 发布到/mavros/odometry/out话题
    nav_msgs::Odometry odometry_out_msg;
    odometry_out_msg.header.stamp = frame_mid_time;
    odometry_out_msg.header.frame_id = "odom"; // 使用MAVROS期望的odom帧
    odometry_out_msg.child_frame_id = base_frame_;
    
    // 创建位姿副本
    Sophus::SE3d mavros_pose = genz_pose;
    
    // 只有当odom和base_link之间有已知的90度差异时才进行转换
    // 这里我们仅旋转位置，保持方向不变，因为方向可能已经被正确处理
    Eigen::Vector3d pos = mavros_pose.translation();
    // 应用-90度旋转（顺时针旋转）
    double x_new = pos.y();
    double y_new = -pos.x();
    mavros_pose.translation() = Eigen::Vector3d(x_new, y_new, pos.z());
    
    odometry_out_msg.pose.pose = tf2::sophusToPose(mavros_pose);
    
    // 设置协方差
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            odometry_out_msg.pose.covariance[i * 6 + j] = pose_covariance(i, j);
            odometry_out_msg.twist.covariance[i * 6 + j] = velocity_covariance(i, j);
        }
    }
    
    // 计算并设置速度
    if (odometry_.poses().size() >= 2) {
        Sophus::SE3d vel_pose = odometry_.poses()[odometry_.poses().size() - 2].inverse() * odometry_.poses().back();
        double dt = 0.1; // 假设每帧间隔0.1秒
        
        // 线速度
        odometry_out_msg.twist.twist.linear.x = vel_pose.translation().x() / dt;
        odometry_out_msg.twist.twist.linear.y = vel_pose.translation().y() / dt;
        odometry_out_msg.twist.twist.linear.z = vel_pose.translation().z() / dt;
        
        // 角速度 (从旋转矩阵提取角速度)
        Eigen::Vector3d angular_velocity = vel_pose.so3().log() / dt;
        odometry_out_msg.twist.twist.angular.x = angular_velocity.x();
        odometry_out_msg.twist.twist.angular.y = angular_velocity.y();
        odometry_out_msg.twist.twist.angular.z = angular_velocity.z();
    }
    
    odometry_out_publisher_.publish(odometry_out_msg);

    // 发布其他消息
    PublishOdometry(genz_pose, msg->header.stamp, cloud_frame_id);
    if (publish_debug_clouds_) {
        PublishClouds(msg->header.stamp, cloud_frame_id, planar_points, non_planar_points);
    }
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
        geometry_msgs::TransformStamped transform_msg;
        transform_msg.header.stamp = stamp;
        transform_msg.header.frame_id = odom_frame_;
        transform_msg.child_frame_id = base_frame_;
        transform_msg.transform = tf2::sophusToTransform(pose);
        tf_broadcaster_.sendTransform(transform_msg);
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
        cloud_header.frame_id = cloud_frame_id;

        map_publisher_.publish(*EigenToPointCloud2(genz_map, odom_header));
        planar_points_publisher_.publish(*EigenToPointCloud2(planar_points, cloud_header));
        non_planar_points_publisher_.publish(*EigenToPointCloud2(non_planar_points, cloud_header));

        return;
    }

    // If transmitting to tf tree we know where the clouds are exactly
    const auto cloud2odom = LookupTransform(odom_frame_, cloud_frame_id);
    planar_points_publisher_.publish(*EigenToPointCloud2(planar_points, odom_header));
    non_planar_points_publisher_.publish(*EigenToPointCloud2(non_planar_points, odom_header));

    if (!base_frame_.empty()) {
        const Sophus::SE3d cloud2base = LookupTransform(base_frame_, cloud_frame_id);
        map_publisher_.publish(*EigenToPointCloud2(genz_map, cloud2base, odom_header));
    } else {
        map_publisher_.publish(*EigenToPointCloud2(genz_map, odom_header));
    }
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
