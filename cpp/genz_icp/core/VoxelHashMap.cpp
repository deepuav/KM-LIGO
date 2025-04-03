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
#include "VoxelHashMap.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

namespace {
static const std::array<Eigen::Vector3i, 27> voxel_shifts{
    Eigen::Vector3i(0, 0, 0),   Eigen::Vector3i(1, 0, 0),   Eigen::Vector3i(-1, 0, 0),
    Eigen::Vector3i(0, 1, 0),   Eigen::Vector3i(0, -1, 0),  Eigen::Vector3i(0, 0, 1),
    Eigen::Vector3i(0, 0, -1),  Eigen::Vector3i(1, 1, 0),   Eigen::Vector3i(1, -1, 0),
    Eigen::Vector3i(-1, 1, 0),  Eigen::Vector3i(-1, -1, 0), Eigen::Vector3i(1, 0, 1),
    Eigen::Vector3i(1, 0, -1),  Eigen::Vector3i(-1, 0, 1),  Eigen::Vector3i(-1, 0, -1),
    Eigen::Vector3i(0, 1, 1),   Eigen::Vector3i(0, 1, -1),  Eigen::Vector3i(0, -1, 1),
    Eigen::Vector3i(0, -1, -1), Eigen::Vector3i(1, 1, 1),   Eigen::Vector3i(1, 1, -1),
    Eigen::Vector3i(1, -1, 1),  Eigen::Vector3i(1, -1, -1), Eigen::Vector3i(-1, 1, 1),
    Eigen::Vector3i(-1, 1, -1), Eigen::Vector3i(-1, -1, 1), Eigen::Vector3i(-1, -1, -1)
};
}  // namespace

namespace genz_icp {

std::tuple<Eigen::Vector3d, size_t, Eigen::Matrix3d, double> VoxelHashMap::GetClosestNeighbor(
    const Eigen::Vector3d &query) const {
    Eigen::Vector3d closest_neighbor = Eigen::Vector3d::Zero();
    double closest_squared_distance = std::numeric_limits<double>::max();
    Eigen::Vector3d total_sum_points = Eigen::Vector3d::Zero();
    Eigen::Matrix3d total_sum_outer = Eigen::Matrix3d::Zero();
    size_t total_n = 0;

    auto kx = static_cast<int>(query.x() / voxel_size_);
    auto ky = static_cast<int>(query.y() / voxel_size_);
    auto kz = static_cast<int>(query.z() / voxel_size_);

    for (const auto &voxel_shift : voxel_shifts) {
        Voxel voxel(kx + voxel_shift.x(), ky + voxel_shift.y(), kz + voxel_shift.z());
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            const auto &voxel_block = search->second;
            total_sum_points += voxel_block.sum_points;
            total_sum_outer += voxel_block.sum_outer;
            total_n += voxel_block.points.size();
            for (const auto &neighbor : voxel_block.points) {
                double squared_distance = (neighbor - query).squaredNorm();
                if (squared_distance < closest_squared_distance) {
                    closest_squared_distance = squared_distance;
                    closest_neighbor = neighbor;
                }
            }
        }
    }

    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    if (total_n >= 1) {
        centroid = total_sum_points / static_cast<double>(total_n);
        covariance = (total_sum_outer / static_cast<double>(total_n)) - (centroid * centroid.transpose());
    }
    double closest_distance = (total_n > 0) ? std::sqrt(closest_squared_distance) : std::numeric_limits<double>::max();

    return std::make_tuple(closest_neighbor, total_n, covariance, closest_distance);
}

std::pair<bool, Eigen::Vector3d> VoxelHashMap::DeterminePlanarity(
    const Eigen::Matrix3d &covariance) const {
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance, Eigen::ComputeEigenvectors);
    normal = solver.eigenvectors().col(0).normalized();

    const auto &eigenvalues = solver.eigenvalues();
    double lambda3 = eigenvalues[0];
    double lambda2 = eigenvalues[1];
    double lambda1 = eigenvalues[2];
    bool is_planar = (lambda3 / (lambda1 + lambda2 + lambda3)) < planarity_threshold_;

    return {is_planar, normal};
}

VoxelHashMap::Vector3dVectorTuple7 VoxelHashMap::GetCorrespondences(
    const Vector3dVector &points, double max_correspondance_distance) const {

    struct ResultTuple {
        Vector3dVector source;
        Vector3dVector target;
        Vector3dVector normals;
        Vector3dVector non_planar_source;
        Vector3dVector non_planar_target;
        size_t planar_count = 0;
        size_t non_planar_count = 0;

        ResultTuple() = default;
        ResultTuple(size_t n) {
            source.reserve(n);
            target.reserve(n);
            normals.reserve(n);
            non_planar_source.reserve(n);
            non_planar_target.reserve(n);
        }

        ResultTuple operator+(ResultTuple other) const {
            ResultTuple result(*this);
            result.source.insert(result.source.end(),
                std::make_move_iterator(other.source.begin()), std::make_move_iterator(other.source.end()));
            result.target.insert(result.target.end(),
                std::make_move_iterator(other.target.begin()), std::make_move_iterator(other.target.end()));
            result.normals.insert(result.normals.end(),
                std::make_move_iterator(other.normals.begin()), std::make_move_iterator(other.normals.end()));
            result.non_planar_source.insert(result.non_planar_source.end(),
                std::make_move_iterator(other.non_planar_source.begin()), std::make_move_iterator(other.non_planar_source.end()));
            result.non_planar_target.insert(result.non_planar_target.end(),
                std::make_move_iterator(other.non_planar_target.begin()), std::make_move_iterator(other.non_planar_target.end()));
            result.planar_count += other.planar_count;
            result.non_planar_count += other.non_planar_count;
            return result;
        }
    };

    auto compute = [&](const tbb::blocked_range<size_t> &r, ResultTuple result) -> ResultTuple {
        for (size_t i = r.begin(); i != r.end(); ++i) {
            const Eigen::Vector3d &point = points[i];
            const auto &[closest_neighbor, n_neighbors, covariance, closest_distance] = GetClosestNeighbor(point);
            if (closest_distance > max_correspondance_distance) continue;

            const size_t min_neighbors_for_normal_estimation = 5;
            if (n_neighbors >= min_neighbors_for_normal_estimation) {
                const auto &[is_planar, normal] = DeterminePlanarity(covariance);
                if (is_planar) {
                    result.source.emplace_back(point);
                    result.target.emplace_back(closest_neighbor);
                    result.normals.emplace_back(normal);
                    result.planar_count++;
                } else if (closest_distance < max_correspondance_distance) {
                    result.non_planar_source.emplace_back(point);
                    result.non_planar_target.emplace_back(closest_neighbor);
                    result.non_planar_count++;
                }
            } else if (closest_distance < max_correspondance_distance) {
                result.non_planar_source.emplace_back(point);
                result.non_planar_target.emplace_back(closest_neighbor);
                result.non_planar_count++;
            }
        }
        return result;
    };

    const auto &[source, target, normals, non_planar_source, non_planar_target, planar_count, non_planar_count] = tbb::parallel_reduce(
        tbb::blocked_range<size_t>(0, points.size()),
        ResultTuple(points.size()),
        compute,
        [&](const ResultTuple &a, const ResultTuple &b) { return a + b; });

    return std::make_tuple(source, target, normals, non_planar_source, non_planar_target, planar_count, non_planar_count);
}

std::vector<Eigen::Vector3d> VoxelHashMap::Pointcloud() const {
    std::vector<Eigen::Vector3d> points;
    points.reserve(max_points_per_voxel_ * map_.size());
    for (const auto &[voxel, voxel_block] : map_) {
        (void)voxel;
        for (const auto &point : voxel_block.points) {
            points.emplace_back(point);
        }
    }
    return points;
}

void VoxelHashMap::Update(const Vector3dVector &points, const Eigen::Vector3d &origin) {
    AddPoints(points);
    RemovePointsFarFromLocation(origin);
}

void VoxelHashMap::Update(const Vector3dVector &points, const Sophus::SE3d &pose) {
    Vector3dVector points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) { return pose * point; });
    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, origin);
}

void VoxelHashMap::AddPoints(const std::vector<Eigen::Vector3d> &points) {
    for (const auto &point : points) {
        auto voxel = Voxel((point / voxel_size_).template cast<int>());
        if (map_.count(voxel)) {
            map_.at(voxel).AddPoint(point);
        } else {
            map_.emplace(voxel, VoxelBlock(point, max_points_per_voxel_));
        }
    }
}

void VoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    const auto max_distance2 = map_cleanup_radius_ * map_cleanup_radius_;
    for (auto it = map_.begin(); it != map_.end(); ) {
        const auto &voxel_block = it->second;
        if (voxel_block.points.empty() || (voxel_block.points.front() - origin).squaredNorm() > max_distance2) {
            it = map_.erase(it);
        } else {
            ++it;
        }
    }
}
}  // namespace genz_icp
