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
    mavros_odometry_publisher_ = pnh_.advertise<nav_msgs::Odometry>("/mavros/odometry/out", queue_size_);
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

    // 使用中间位姿作为初始猜测值
    const Sophus::SE3d initial_guess = mid_pose_px4;

    // 输出位姿信息
    ROS_INFO("=== 位姿信息 ===");
    ROS_INFO("start_px4_pose:   trans=[%.3f, %.3f, %.3f], rot=[%.3f, %.3f, %.3f, %.3f]",
             start_px4_pose.translation().x(), start_px4_pose.translation().y(), start_px4_pose.translation().z(),
             start_px4_pose.so3().unit_quaternion().w(), start_px4_pose.so3().unit_quaternion().x(),
             start_px4_pose.so3().unit_quaternion().y(), start_px4_pose.so3().unit_quaternion().z());
    
    ROS_INFO("mid_pose_px4:     trans=[%.3f, %.3f, %.3f], rot=[%.3f, %.3f, %.3f, %.3f]",
             mid_pose_px4.translation().x(), mid_pose_px4.translation().y(), mid_pose_px4.translation().z(),
             mid_pose_px4.so3().unit_quaternion().w(), mid_pose_px4.so3().unit_quaternion().x(),
             mid_pose_px4.so3().unit_quaternion().y(), mid_pose_px4.so3().unit_quaternion().z());
    
    ROS_INFO("finish_px4_pose:  trans=[%.3f, %.3f, %.3f], rot=[%.3f, %.3f, %.3f, %.3f]",
             finish_px4_pose.translation().x(), finish_px4_pose.translation().y(), finish_px4_pose.translation().z(),
             finish_px4_pose.so3().unit_quaternion().w(), finish_px4_pose.so3().unit_quaternion().x(), 
             finish_px4_pose.so3().unit_quaternion().y(), finish_px4_pose.so3().unit_quaternion().z());
    
    ROS_INFO("initial_guess:    trans=[%.3f, %.3f, %.3f], rot=[%.3f, %.3f, %.3f, %.3f]",
             initial_guess.translation().x(), initial_guess.translation().y(), initial_guess.translation().z(),
             initial_guess.so3().unit_quaternion().w(), initial_guess.so3().unit_quaternion().x(),
             initial_guess.so3().unit_quaternion().y(), initial_guess.so3().unit_quaternion().z());

    // 使用起始位姿、中间位姿和结束位姿进行点云去畸变和配准
    const auto &[planar_points, non_planar_points] = odometry_.RegisterFrame(points, timestamps, 
                                                                             start_px4_pose, 
                                                                             mid_pose_px4, 
                                                                             finish_px4_pose);

    // 获取配准后的位姿
    const Sophus::SE3d genz_pose = odometry_.poses().back();

    // 输出配准后的位姿
    ROS_INFO("genz_pose:        trans=[%.3f, %.3f, %.3f], rot=[%.3f, %.3f, %.3f, %.3f]",
             genz_pose.translation().x(), genz_pose.translation().y(), genz_pose.translation().z(),
             genz_pose.so3().unit_quaternion().w(), genz_pose.so3().unit_quaternion().x(),
             genz_pose.so3().unit_quaternion().y(), genz_pose.so3().unit_quaternion().z());
    ROS_INFO("=================");

    // 计算并发布mavros odometry
    nav_msgs::Odometry mavros_odom_msg;
    mavros_odom_msg.header.stamp = frame_mid_time;
    mavros_odom_msg.header.frame_id = odom_frame_;
    mavros_odom_msg.child_frame_id = base_frame_.empty() ? cloud_frame_id : base_frame_;
    
    // 设置位姿
    mavros_odom_msg.pose.pose = tf2::sophusToPose(genz_pose);
    
    // 计算位姿协方差（根据genz_pose和mid_pose_px4的差距）
    Sophus::SE3d pose_diff = genz_pose.inverse() * mid_pose_px4;
    double position_diff = pose_diff.translation().norm();
    double rotation_diff = pose_diff.so3().log().norm();
    
    // 位姿协方差（根据位姿差距设置）
    double pos_cov = std::max(0.01, position_diff * 0.1); // 最小协方差为0.01
    double rot_cov = std::max(0.01, rotation_diff * 0.1); // 最小协方差为0.01
    
    // 设置位姿协方差矩阵 (6x6)
    // 顺序是 [x, y, z, rot_x, rot_y, rot_z]
    mavros_odom_msg.pose.covariance[0] = pos_cov;  // x
    mavros_odom_msg.pose.covariance[7] = pos_cov;  // y
    mavros_odom_msg.pose.covariance[14] = pos_cov; // z
    mavros_odom_msg.pose.covariance[21] = rot_cov; // rot_x
    mavros_odom_msg.pose.covariance[28] = rot_cov; // rot_y
    mavros_odom_msg.pose.covariance[35] = rot_cov; // rot_z
    
    // 计算速度
    Eigen::Vector3d genz_v(0, 0, 0);
    Eigen::Vector3d px4_v(0, 0, 0);
    
    if (odometry_.poses().size() >= 2 && has_last_frame_) {
        // 计算GenZ ICP速度（根据最后两个位姿）
        double dt = (frame_mid_time - last_frame_mid_time_).toSec();
        if (dt > 0) {
            const Sophus::SE3d& prev_pose = odometry_.poses()[odometry_.poses().size() - 2];
            Sophus::SE3d pose_change = prev_pose.inverse() * genz_pose;
            genz_v = pose_change.translation() / dt;
            
            // 计算PX4速度（根据前后两帧的中间位姿）
            Sophus::SE3d px4_pose_change = last_mid_pose_px4_.inverse() * mid_pose_px4;
            px4_v = px4_pose_change.translation() / dt;
            
            // 设置线速度
            mavros_odom_msg.twist.twist.linear.x = genz_v.x();
            mavros_odom_msg.twist.twist.linear.y = genz_v.y();
            mavros_odom_msg.twist.twist.linear.z = genz_v.z();
            
            // 计算角速度（从旋转矩阵的差分）
            Eigen::Vector3d angular_velocity = pose_change.so3().log() / dt;
            mavros_odom_msg.twist.twist.angular.x = angular_velocity.x();
            mavros_odom_msg.twist.twist.angular.y = angular_velocity.y();
            mavros_odom_msg.twist.twist.angular.z = angular_velocity.z();
            
            // 根据genz_v和px4_v的差距计算速度协方差
            Eigen::Vector3d vel_diff = genz_v - px4_v;
            double vel_diff_norm = vel_diff.norm();
            double vel_cov = std::max(0.01, vel_diff_norm * 0.1); // 最小协方差为0.01
            
            // 设置速度协方差矩阵 (6x6)
            mavros_odom_msg.twist.covariance[0] = vel_cov;  // vx
            mavros_odom_msg.twist.covariance[7] = vel_cov;  // vy
            mavros_odom_msg.twist.covariance[14] = vel_cov; // vz
            mavros_odom_msg.twist.covariance[21] = rot_cov; // ang_vx
            mavros_odom_msg.twist.covariance[28] = rot_cov; // ang_vy
            mavros_odom_msg.twist.covariance[35] = rot_cov; // ang_vz
        }
    }
    
    // 发布odometry消息
    mavros_odometry_publisher_.publish(mavros_odom_msg);
    
    // 保存当前帧信息用于下一次计算
    last_mid_pose_px4_ = mid_pose_px4;
    last_frame_mid_time_ = frame_mid_time;
    has_last_frame_ = true;

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
