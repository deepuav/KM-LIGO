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

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <utility>
#include <vector>

#include "VoxelHashMap.hpp"

namespace genz_icp {

/// Crop the frame with max/min ranges
std::vector<Eigen::Vector3d> Preprocess(const std::vector<Eigen::Vector3d> &frame,
                                        double max_range,
                                        double min_range);

/// This function only applies for the KITTI dataset, and should NOT be used by any other dataset,
/// the original idea and part of the implementation is taken from CT-ICP (although IMLS-SLAM
/// originally introduced the calibration factor)
std::vector<Eigen::Vector3d> CorrectKITTIScan(const std::vector<Eigen::Vector3d> &frame);

std::vector<Eigen::Vector3d> VoxelDownsample(const std::vector<Eigen::Vector3d> &frame,
                                             double voxel_size);

std::vector<Eigen::Vector3d> VoxelDownsampleForMap(const std::vector<Eigen::Vector3d> &frame,
                                                   double voxel_size);

std::vector<Eigen::Vector3d> VoxelDownsampleForScan(const std::vector<Eigen::Vector3d> &frame,
                                                    double voxel_size);

double Clamp(double value, double min, double max);

class Registration {
public:
    Registration(int max_num_iteration, double convergence_criterion);

    std::tuple<Sophus::SE3d, std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> RegisterFrame(
        const std::vector<Eigen::Vector3d> &frame,
        const VoxelHashMap &voxel_map,
        const Sophus::SE3d &initial_guess,
        double max_correspondence_distance,
        double kernel);

    // 获取配准质量指标
    double GetLastConvergenceError() const { return last_convergence_error_; }
    int GetLastIterationCount() const { return last_iteration_count_; }
    double GetLastInlierRatio() const { return last_inlier_ratio_; }

private:
    int max_num_iterations_;
    double convergence_criterion_;
    
    // 配准质量指标
    double last_convergence_error_ = 0.0;
    int last_iteration_count_ = 0;
    double last_inlier_ratio_ = 1.0;
};

}  // namespace genz_icp
