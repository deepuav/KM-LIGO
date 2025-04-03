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
//
// NOTE: This implementation is heavily inspired in the original CT-ICP VoxelHashMap implementation,
// although it was heavily modified and drastically simplified, but if you are using this module you
// should at least acknowledge the work from CT-ICP by giving a star on GitHub
#pragma once

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

namespace genz_icp {
struct VoxelHashMap {
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;
    using Vector3dVectorTuple7 = std::tuple<Vector3dVector, Vector3dVector, Vector3dVector, Vector3dVector, Vector3dVector, size_t, size_t>;
    using Voxel = Eigen::Vector3i;
    struct VoxelBlock {
        // Buffer of points with a max limit of max_points
        std::vector<Eigen::Vector3d> points;
        int max_points_; // Renamed for clarity (was num_points_)
        Eigen::Vector3d sum_points = Eigen::Vector3d::Zero();
        Eigen::Matrix3d sum_outer = Eigen::Matrix3d::Zero();

        // Default constructor for empty voxel
        VoxelBlock(int max_points) : max_points_(max_points) {}

        // Constructor for initializing with the first point
        VoxelBlock(const Eigen::Vector3d& initial_point, int max_points)
            : points{initial_point},
              max_points_(max_points),
              sum_points(initial_point),
              sum_outer(initial_point * initial_point.transpose()) {}

        inline void AddPoint(const Eigen::Vector3d &point) {
            if (points.size() < static_cast<size_t>(max_points_)) {
                points.push_back(point);
                sum_points += point;
                sum_outer += point * point.transpose();
            }
        }
    };
    struct VoxelHash {
        size_t operator()(const Voxel &voxel) const {
            const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
            return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
        }
    };

    explicit VoxelHashMap(double voxel_size, double max_distance, double map_cleanup_radius,
                          double planarity_threshold, int max_points_per_voxel)
        : voxel_size_(voxel_size),
          max_distance_(max_distance),
          map_cleanup_radius_(map_cleanup_radius),
          planarity_threshold_(planarity_threshold),
          max_points_per_voxel_(max_points_per_voxel) {}

    Vector3dVectorTuple7 GetCorrespondences(const Vector3dVector &points,
                                           double max_correspondance_distance) const;
    inline void Clear() { map_.clear(); }
    inline bool Empty() const { return map_.empty(); }
    void Update(const std::vector<Eigen::Vector3d> &points, const Eigen::Vector3d &origin);
    void Update(const std::vector<Eigen::Vector3d> &points, const Sophus::SE3d &pose);
    void AddPoints(const std::vector<Eigen::Vector3d> &points);
    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    std::vector<Eigen::Vector3d> Pointcloud() const;
    std::tuple<Eigen::Vector3d, size_t, Eigen::Matrix3d, double> GetClosestNeighbor(const Eigen::Vector3d &query) const;
    std::pair<bool, Eigen::Vector3d> DeterminePlanarity(const Eigen::Matrix3d &covariance) const;

    double voxel_size_;
    double max_distance_;
    double map_cleanup_radius_;
    double planarity_threshold_;
    int max_points_per_voxel_;
    tsl::robin_map<Voxel, VoxelBlock, VoxelHash> map_;
};
}  // namespace genz_icp
