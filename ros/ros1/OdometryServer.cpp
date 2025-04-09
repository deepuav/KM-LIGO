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
#include <map>

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
    odometry_out_publisher_ = nh_.advertise<nav_msgs::Odometry>("/mavros/odometry/out", queue_size_);
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
    px4_pose.pose = tf2::poseToSophus(msg->pose);

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
    // 处理时间戳在两个位姿之前的情况
    if (target_time <= pose1.timestamp) {
        return pose1.pose;
    }
    
    // 处理时间戳在两个位姿之后的情况
    if (target_time >= pose2.timestamp) {
        return pose2.pose;
    }
    
    // 计算插值比例
    double total_duration = (pose2.timestamp - pose1.timestamp).toSec();
    double target_duration = (target_time - pose1.timestamp).toSec();
    double alpha = target_duration / total_duration;

    // 插值计算平移部分
    Eigen::Vector3d translation = (1 - alpha) * pose1.pose.translation() + alpha * pose2.pose.translation();

    // 插值计算旋转部分（使用SLERP）
    Eigen::Quaterniond q1 = pose1.pose.so3().unit_quaternion();
    Eigen::Quaterniond q2 = pose2.pose.so3().unit_quaternion();
    Eigen::Quaterniond q_interp = q1.slerp(alpha, q2);

    return Sophus::SE3d(q_interp, translation);
}

void OdometryServer::RegisterFrame(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    // Check if we have enough poses in the queue
    if (px4_pose_queue_.size() < MIN_PX4_POSE_QUEUE_SIZE) {
        ROS_WARN_THROTTLE(1.0, "Waiting for enough PX4 poses (current: %zu, required: %zu)",
                         px4_pose_queue_.size(), MIN_PX4_POSE_QUEUE_SIZE);
        return;
    }

    const auto cloud_frame_id = msg->header.frame_id;
    const auto points = PointCloud2ToEigen(msg);
    const auto timestamps = [&]() -> std::vector<double> {
        if (!config_.deskew) return {};
        return GetTimestamps(msg);
    }();

    // 获取点云帧的起始时间和结束时间（结束时间 = 起始时间 + 0.1秒）
    ros::Time frame_start_time = msg->header.stamp;
    ros::Time frame_end_time = frame_start_time + ros::Duration(0.1);
    ros::Time frame_mid_time = frame_start_time + ros::Duration(0.05);

    // 查找最近的位姿对
    std::pair<PX4Pose, PX4Pose> start_pose_pair = FindNearestPoses(frame_start_time);
    std::pair<PX4Pose, PX4Pose> end_pose_pair = FindNearestPoses(frame_end_time);
    std::pair<PX4Pose, PX4Pose> mid_pose_pair = FindNearestPoses(frame_mid_time);
    
    // 插值计算精确的位姿
    Sophus::SE3d start_px4_pose = InterpolatePose(start_pose_pair.first, start_pose_pair.second, frame_start_time);
    Sophus::SE3d finish_px4_pose = InterpolatePose(end_pose_pair.first, end_pose_pair.second, frame_end_time);
    Sophus::SE3d mid_pose_px4 = InterpolatePose(mid_pose_pair.first, mid_pose_pair.second, frame_mid_time);

    // 使用起始位姿和结束位姿进行点云去畸变和配准
    const auto &[planar_points, non_planar_points] = odometry_.RegisterFrame(points, timestamps, start_px4_pose, finish_px4_pose);

    // 获取配准后的位姿
    const Sophus::SE3d genz_pose = odometry_.poses().back();

    // 计算位姿差、线速度和角速度
    Sophus::SE3d pose_diff, vision_pose;
    Eigen::Vector3d genz_v = Eigen::Vector3d::Zero(); // 线速度
    Eigen::Vector3d genz_rv = Eigen::Vector3d::Zero(); // 角速度
    Eigen::Vector3d px4_v = Eigen::Vector3d::Zero(); // PX4线速度
    Eigen::Vector3d px4_rv = Eigen::Vector3d::Zero(); // PX4角速度
    
    // 计算位姿和速度
    if (odometry_.poses().size() >= 2) {
        // 计算poses_向量最后一个位姿和倒数第二个位姿的差
        pose_diff = odometry_.poses().back() * odometry_.poses()[odometry_.poses().size() - 2].inverse();
        // 计算GenZ的线速度和角速度
        Eigen::Vector3d trans_diff = pose_diff.translation();
        Eigen::AngleAxisd rot_diff(pose_diff.so3().matrix());
        double dt = 0.1; // 一帧点云的时间是0.1秒
        genz_v = trans_diff / dt;
        genz_rv = rot_diff.axis() * rot_diff.angle() / dt;
        
        // 将这个差值加到中间时间的位姿上
        vision_pose = pose_diff * mid_pose_px4;
    } else {
        // 如果poses_向量中只有一个元素，使用一半的位姿差
        pose_diff = odometry_.poses().back() * start_px4_pose.inverse();
        // 计算差值的一半
        Sophus::SE3d half_diff = Sophus::SE3d::exp(0.5 * pose_diff.log());
        // 将半个差值加到起始位姿上
        vision_pose = half_diff * start_px4_pose;
    }
    
    // 计算PX4的速度（如果有上一帧中间位姿）
    if (has_prev_mid_pose_) {
        Sophus::SE3d px4_pose_diff = mid_pose_px4 * prev_mid_pose_px4_.inverse();
        Eigen::Vector3d px4_trans_diff = px4_pose_diff.translation();
        Eigen::AngleAxisd px4_rot_diff(px4_pose_diff.so3().matrix());
        double px4_dt = (frame_mid_time - prev_mid_time_).toSec();
        if (px4_dt > 0.001) { // 防止除零
            px4_v = px4_trans_diff / px4_dt;
            px4_rv = px4_rot_diff.axis() * px4_rot_diff.angle() / px4_dt;
        }
    }
    
    // 存储当前中间位姿用于下次计算
    prev_mid_pose_px4_ = mid_pose_px4;
    prev_mid_time_ = frame_mid_time;
    has_prev_mid_pose_ = true;
    
    // 计算协方差
    // 位姿协方差：根据genz_pose和mid_pose_px4的差距计算
    Sophus::SE3d pose_error = genz_pose * mid_pose_px4.inverse();
    double trans_error = pose_error.translation().norm();
    double rot_error = Eigen::AngleAxisd(pose_error.so3().matrix()).angle();
    
    // 线速度协方差：根据genz_v和px4_v的差距计算
    double v_error = (genz_v - px4_v).norm();
    
    // 角速度协方差：根据genz_rv和px4_rv的差距计算
    double rv_error = (genz_rv - px4_rv).norm();
    
    // 设置位姿协方差（使用对角阵简化）
    Eigen::Matrix<double, 6, 6> pose_cov = Eigen::Matrix<double, 6, 6>::Identity();
    double pos_cov_factor = 0.01 * (1.0 + trans_error); // 根据位置误差调整协方差
    double rot_cov_factor = 0.01 * (1.0 + rot_error);   // 根据旋转误差调整协方差
    
    // 位置协方差 (x, y, z)
    pose_cov(0, 0) = pos_cov_factor;
    pose_cov(1, 1) = pos_cov_factor;
    pose_cov(2, 2) = pos_cov_factor;
    
    // 旋转协方差 (roll, pitch, yaw)
    pose_cov(3, 3) = rot_cov_factor;
    pose_cov(4, 4) = rot_cov_factor;
    pose_cov(5, 5) = rot_cov_factor;
    
    // 速度协方差
    Eigen::Matrix<double, 6, 6> twist_cov = Eigen::Matrix<double, 6, 6>::Identity();
    double lin_vel_cov_factor = 0.01 * (1.0 + v_error);  // 根据线速度误差调整协方差
    double ang_vel_cov_factor = 0.01 * (1.0 + rv_error); // 根据角速度误差调整协方差
    
    // 线速度协方差 (vx, vy, vz)
    twist_cov(0, 0) = lin_vel_cov_factor;
    twist_cov(1, 1) = lin_vel_cov_factor;
    twist_cov(2, 2) = lin_vel_cov_factor;
    
    // 角速度协方差 (wx, wy, wz)
    twist_cov(3, 3) = ang_vel_cov_factor;
    twist_cov(4, 4) = ang_vel_cov_factor;
    twist_cov(5, 5) = ang_vel_cov_factor;
    
    // 发布odometry消息到/mavros/odometry/out
    nav_msgs::Odometry odometry_out_msg;
    odometry_out_msg.header.stamp = frame_mid_time;
    odometry_out_msg.header.frame_id = odom_frame_;
    odometry_out_msg.child_frame_id = base_frame_.empty() ? cloud_frame_id : base_frame_;
    
    // 设置位姿
    odometry_out_msg.pose.pose = tf2::sophusToPose(genz_pose);
    
    // 复制位姿协方差
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            odometry_out_msg.pose.covariance[i * 6 + j] = pose_cov(i, j);
        }
    }
    
    // 设置线速度和角速度
    odometry_out_msg.twist.twist.linear.x = genz_v.x();
    odometry_out_msg.twist.twist.linear.y = genz_v.y();
    odometry_out_msg.twist.twist.linear.z = genz_v.z();
    odometry_out_msg.twist.twist.angular.x = genz_rv.x();
    odometry_out_msg.twist.twist.angular.y = genz_rv.y();
    odometry_out_msg.twist.twist.angular.z = genz_rv.z();
    
    // 复制速度协方差
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            odometry_out_msg.twist.covariance[i * 6 + j] = twist_cov(i, j);
        }
    }
    
    // 发布odometry消息
    odometry_out_publisher_.publish(odometry_out_msg);

    // Publish other messages as before
    PublishOdometry(genz_pose, msg->header.stamp, cloud_frame_id);
    if (publish_debug_clouds_) {
        PublishClouds(msg->header.stamp, cloud_frame_id, planar_points, non_planar_points);
    }
}

Sophus::SE3d OdometryServer::LookupTransform(const std::string &target_frame,
                                             const std::string &source_frame) const {
    std::string err_msg;
    static std::map<std::string, ros::Time> last_warn_time;
    std::string frame_pair = target_frame + "-" + source_frame;
    
    if (tf2_buffer_._frameExists(source_frame) &&  //
        tf2_buffer_._frameExists(target_frame) &&  //
        tf2_buffer_.canTransform(target_frame, source_frame, ros::Time(0), &err_msg)) {
        try {
            auto tf = tf2_buffer_.lookupTransform(target_frame, source_frame, ros::Time(0));
            return tf2::transformToSophus(tf);
        } catch (tf2::TransformException &ex) {
            // 限制警告频率，每10秒最多输出一次
            ros::Time now = ros::Time::now();
            if (!last_warn_time.count(frame_pair) || 
                (now - last_warn_time[frame_pair]).toSec() > 10.0) {
                ROS_WARN("%s", ex.what());
                last_warn_time[frame_pair] = now;
            }
        }
    } else {
        // 限制警告频率，每10秒最多输出一次
        ros::Time now = ros::Time::now();
        if (!last_warn_time.count(frame_pair) || 
            (now - last_warn_time[frame_pair]).toSec() > 10.0) {
            ROS_WARN("Failed to find tf between %s and %s. Reason=%s", target_frame.c_str(),
                    source_frame.c_str(), err_msg.c_str());
            last_warn_time[frame_pair] = now;
        }
    }
    // 返回单位变换（不进行任何转换）
    return Sophus::SE3d();
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
        transform_msg.child_frame_id = base_frame_.empty() ? cloud_frame_id : base_frame_;
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

    // 如果传输到tf树，我们需要知道点云的确切位置
    const auto cloud2odom = LookupTransform(odom_frame_, cloud_frame_id);
    
    // 检查cloud2odom变换是否为单位变换（无效变换）
    if (cloud2odom.translation().norm() < 1e-6 && 
        cloud2odom.so3().log().norm() < 1e-6) {
        // cloud2odom变换无效，使用cloud_frame_id作为frame_id发布点云
        std_msgs::Header cloud_header;
        cloud_header.stamp = stamp;
        cloud_header.frame_id = cloud_frame_id;
        
        // 使用cloud_header发布平面点和非平面点
        planar_points_publisher_.publish(*EigenToPointCloud2(planar_points, cloud_header));
        non_planar_points_publisher_.publish(*EigenToPointCloud2(non_planar_points, cloud_header));
        
        // 直接使用cloud_header发布地图
        map_publisher_.publish(*EigenToPointCloud2(genz_map, cloud_header));
        
        // 提前返回
        return;
    }
    
    // 变换有效，使用odom_header发布平面点和非平面点
    planar_points_publisher_.publish(*EigenToPointCloud2(planar_points, odom_header));
    non_planar_points_publisher_.publish(*EigenToPointCloud2(non_planar_points, odom_header));

    if (!base_frame_.empty()) {
        const Sophus::SE3d cloud2base = LookupTransform(base_frame_, cloud_frame_id);
        // 检查变换是否为单位变换（无效变换）
        if (cloud2base.translation().norm() < 1e-6 && 
            cloud2base.so3().log().norm() < 1e-6) {
            // 变换无效，直接使用odom_header发布地图
            map_publisher_.publish(*EigenToPointCloud2(genz_map, odom_header));
        } else {
            // 变换有效，使用变换后发布地图
            map_publisher_.publish(*EigenToPointCloud2(genz_map, cloud2base, odom_header));
        }
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
