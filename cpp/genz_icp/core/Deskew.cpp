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
#include "Deskew.hpp"

#include <tbb/parallel_for.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

namespace {
constexpr double mid_pose_timestamp{0.5};
}  // namespace

namespace genz_icp {
std::vector<Eigen::Vector3d> DeSkewScan(const std::vector<Eigen::Vector3d> &frame,
                                        const std::vector<double> &timestamps,
                                        const Sophus::SE3d &start_pose,
                                        const Sophus::SE3d &finish_pose) {
    const auto delta_pose = (start_pose.inverse() * finish_pose).log();

    std::vector<Eigen::Vector3d> corrected_frame(frame.size());
    tbb::parallel_for(size_t(0), frame.size(), [&](size_t i) {
        const auto motion = Sophus::SE3d::exp((timestamps[i] - mid_pose_timestamp) * delta_pose); // original
        corrected_frame[i] = motion * frame[i];
    });
    return corrected_frame;
}

std::vector<Eigen::Vector3d> DeSkewScanWithPoses(const std::vector<Eigen::Vector3d> &frame,
                                               const std::vector<double> &timestamps,
                                               const std::vector<Sophus::SE3d> &poses,
                                               const std::vector<double> &pose_times) {
    if (poses.size() < 2 || pose_times.size() != poses.size()) {
        // 位姿数量不足或时间戳数量不匹配，回退到原始实现
        if (poses.size() >= 2) {
            return DeSkewScan(frame, timestamps, poses.front(), poses.back());
        }
        return frame; // 无法去畸变，返回原始点云
    }
    
    // 假设timestamps是归一化的，范围在[0,1]之间
    // 如果不是，我们需要归一化timestamp以匹配pose_times的范围
    double min_timestamp = *std::min_element(timestamps.begin(), timestamps.end());
    double max_timestamp = *std::max_element(timestamps.begin(), timestamps.end());
    double timestamp_range = max_timestamp - min_timestamp;
    
    // 如果时间戳范围太小，不需要去畸变
    if (timestamp_range < 1e-6) {
        return frame;
    }
    
    // 初始化点云坐标
    std::vector<Eigen::Vector3d> corrected_frame(frame.size());
    
    // 使用TBB并行处理每个点
    tbb::parallel_for(size_t(0), frame.size(), [&](size_t i) {
        // 将点的时间戳归一化到pose_times的范围
        double normalized_time = (timestamps[i] - min_timestamp) / timestamp_range;
        double point_time = pose_times.front() + normalized_time * (pose_times.back() - pose_times.front());
        
        // 在pose_times中找到当前点时间戳的位置
        auto it = std::lower_bound(pose_times.begin(), pose_times.end(), point_time);
        
        // 处理边界情况
        if (it == pose_times.begin()) {
            // 点时间早于第一个位姿，使用第一个位姿
            corrected_frame[i] = poses.front() * frame[i];
            return;
        }
        
        if (it == pose_times.end()) {
            // 点时间晚于最后一个位姿，使用最后一个位姿
            corrected_frame[i] = poses.back() * frame[i];
            return;
        }
        
        // 找到时间上最接近的两个位姿进行插值
        size_t idx2 = std::distance(pose_times.begin(), it);
        size_t idx1 = idx2 - 1;
        
        // 计算插值因子
        double t1 = pose_times[idx1];
        double t2 = pose_times[idx2];
        double alpha = (point_time - t1) / (t2 - t1);
        
        // 提取位姿并进行插值
        const Sophus::SE3d& pose1 = poses[idx1];
        const Sophus::SE3d& pose2 = poses[idx2];
        
        // 在位姿之间进行插值
        Eigen::Quaterniond q1 = pose1.so3().unit_quaternion();
        Eigen::Quaterniond q2 = pose2.so3().unit_quaternion();
        Eigen::Quaterniond q_interp = q1.slerp(alpha, q2);
        
        Eigen::Vector3d t_interp = (1 - alpha) * pose1.translation() + alpha * pose2.translation();
        
        // 创建插值位姿并应用于点
        Sophus::SE3d interp_pose(q_interp, t_interp);
        corrected_frame[i] = interp_pose * frame[i];
    });
    
    return corrected_frame;
}
}  // namespace genz_icp
