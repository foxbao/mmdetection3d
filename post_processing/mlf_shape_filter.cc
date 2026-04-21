/******************************************************************************
 * Copyright 2018 The century Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#include "modules/perception/lidar_tracking/tracker/multi_lidar_fusion/mlf_shape_filter.h"

#include "cyber/common/file.h"
#include "modules/perception/common/util.h"
#include "modules/perception/lidar/common/lidar_object_util.h"
#include "modules/perception/lidar_tracking/tracker/multi_lidar_fusion/proto/multi_lidar_fusion_config.pb.h"

namespace century {
namespace perception {
namespace lidar {

std::array<Eigen::Vector3d, 4> GenerateBottomSurfaceCorners(
  const Eigen::Vector3d& center, 
  const double length, 
  const double width, 
  const double height, 
  const double sinTheta,
  const double cosTheta) {
  
  std::array<Eigen::Vector3d, 4> corners;
  double halfLength = length / 2.0;
  double halfWidth = width / 2.0;
  
  double localCorners[4][3] = {
    { halfLength,  halfWidth, 0.0}, 
    { halfLength, -halfWidth, 0.0},  
    {-halfLength, -halfWidth, 0.0},  
    {-halfLength,  halfWidth, 0.0}   
  };
  
  for (int i = 0; i < corners.size(); ++i) {
    double localX = localCorners[i][0];
    double localY = localCorners[i][1];
    double localZ = localCorners[i][2];
    
    corners[i](0) = localX * cosTheta - localY * sinTheta;
    corners[i](1) = localX * sinTheta + localY * cosTheta;
    corners[i](2) = localZ;  
    
    corners[i](0) += center.x();
    corners[i](1) += center.y();
    corners[i](2) += center.z(); 
  }
  
  return corners;
}

double CalculateAngleDifference(const double& angle1, const double& angle2) {
  double diff = std::fabs(angle1 - angle2);
  if (diff > M_PI) {
      diff = 2 * M_PI - diff;
  }
  return diff;
}

double CalculateAngleDifference(const Eigen::Vector3d& vec1, const Eigen::Vector3d& vec2) {
  double dot_product = vec1.dot(vec2);
  
  double norm1 = vec1.norm();
  double norm2 = vec2.norm();
  
  if (norm1 < 1e-6 || norm2 < 1e-6) {
      return 0.0;
  }
  double cos_angle = dot_product / (norm1 * norm2);
  cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
  return std::acos(cos_angle);
}


void OrientationAnomalyDetector::SetParameters(size_t history_size, double vel_threshold, double yaw_threshold_deg,
    const Eigen::Vector3d last_stable_direction) {
  max_history_size_ = history_size;
  velocity_threshold_ = vel_threshold;
  yaw_change_threshold_ = yaw_threshold_deg * M_PI / 180.0;
  if (last_stable_direction != Eigen::Vector3d::Zero()) {
    stable_angle_ = ExtractYaw(last_stable_direction, false);
  }
}

bool OrientationAnomalyDetector::ProcessNewDetection(const ObjectState& new_detection) {
  if (!new_detection.is_valid) {
    return false;
  }
  yaws_.clear();
  for (const auto& state : history_) {
    yaws_.push_back(ExtractYaw(state.orientation));
  }
  
  bool is_mainly_static = CheckIfMainlyStatic() || true;
  if (is_mainly_static && history_.size() >= 3) {
    bool is_orientation_anomaly = CheckYawAnomaly(new_detection);
    
    if (is_orientation_anomaly) {
      AINFO << "Anomaly yaw, use history" << std::endl;
      return false;
    }
  }
  return true; 
}

ObjectState OrientationAnomalyDetector::GetMostStableResult() {
  if (history_.empty()) {
    throw std::runtime_error("empty history");
  }
  return GetMedianFilteredResult();
}

double OrientationAnomalyDetector::ExtractYaw(const Eigen::Vector3d& orientation, bool norm) const {
    // orientation: (y_component, x_component, ignore)
  if (norm) {
    return NormalizeAngle(std::atan2(orientation[1], orientation[0])); // atan2(y, x)
  } else {
    return std::atan2(orientation[1], orientation[0]); // atan2(y, x)
  }
}

bool OrientationAnomalyDetector::CheckIfMainlyStatic() {
  if (history_.size() < 5) return false; 
  int static_count = 0;
  for (const auto& state : history_) {
    double horizontal_speed = std::sqrt(state.velocity[0] * state.velocity[0] + 
                                      state.velocity[1] * state.velocity[1]);
    if (horizontal_speed < velocity_threshold_) {
      static_count++;
    }
  }
  return static_count >= history_.size() * 0.8;
}

bool OrientationAnomalyDetector::CalculateAngleCluster(int use_ange_size, std::pair<double, int>& max_cluster) {
  std::vector<double> yaws;
  for (const auto& state : history_) {
    yaws.push_back(ExtractYaw(state.orientation, false));
  }
  if (use_ange_size > yaws.size()) {
    return false;
  }
  
  std::vector<std::pair<double, int>> clusters;
  for (int i = 0; i < use_ange_size;i++) {
    double angle = yaws[yaws.size() - i -1];
  
    bool found_cluster = false;
    
    for (auto& cluster : clusters) {
      if (NormalizeAngle(NormalizeAngle(cluster.first) -  NormalizeAngle(angle)) < 3.0 / 180.0 * M_PI) {
          
        double cluster_sin = std::sin(cluster.first) * cluster.second;
        double cluster_cos = std::cos(cluster.first) * cluster.second;
        
        cluster_sin += std::sin(angle);
        cluster_cos += std::cos(angle);
        
        cluster.first = std::atan2(cluster_sin, cluster_cos);
        cluster.second++;
        found_cluster = true;
        break;
      }
    }
    
    if (!found_cluster) {
      clusters.emplace_back(angle, 1);
    }
  }
    
  auto max_iter = std::max_element(clusters.begin(), clusters.end(),
      [](const auto& a, const auto& b) { return a.second < b.second;});
  if (max_iter == clusters.end()) {
    return false;
  }
  max_cluster = *max_iter;
  return true;
}

bool OrientationAnomalyDetector::CheckYawAnomaly(const ObjectState& new_detection) {
  double new_yaw = ExtractYaw(new_detection.orientation);
  
  // double historical_std = CalculateYawStdDev();
  double historical_avg = CalculateAverageYaw();
  std::pair<double, int> historical_angele_cluster;
  bool cluster_flag = CalculateAngleCluster(20, historical_angele_cluster);

  if (cluster_flag) {
    if (historical_angele_cluster.second >= 5) {
      if (NormalizeAngle(historical_angele_cluster.first - historical_avg) > 3.0 / 180.0 * M_PI
          || NormalizeAngle(historical_angele_cluster.first - stable_angle_) > 3.0 / 180.0 * M_PI) {
        int count = 0;
        int diff_num = 0;
        for (auto it = yaws_.rbegin(); it != yaws_.rend() && count < 10; ++it, ++count) {
          if (NormalizeAngle(*it - stable_angle_) > 3.0 / 180.0 * M_PI) {
            diff_num++;
          }
        }
        if (diff_num > 5) {
          update_stable_dir_ = true;
          stable_angle_ = historical_angele_cluster.first;
        }  
        historical_avg = historical_angele_cluster.first;
      } 
    }
  }
  
  double max_historical_change = CalculateMaxHistoricalYawChange();
  double angle_diff = NormalizeAngle(new_yaw - historical_avg);
  bool is_anomaly = (std::abs(angle_diff) > yaw_change_threshold_) || 
                    (std::abs(angle_diff) > max_historical_change * 3.0);
  
  return is_anomaly;
}

double OrientationAnomalyDetector::NormalizeAngle(double angle) const {
  while (angle >= 2 * M_PI) angle -= 2 * M_PI;
  while (angle < 0) angle += 2 * M_PI;          
  return std::min(angle, 2 * M_PI - angle);

}

double OrientationAnomalyDetector::CalculateYawStdDev() {
  if (history_.size() < 2) return 0.0;
  double avg_yaw = CalculateAverageYaw();
  
  double variance = 0.0;
  for (double yaw : yaws_) {
      double diff = NormalizeAngle(yaw - avg_yaw);
      variance += diff * diff;
  }
  variance /= yaws_.size();
  
  return std::sqrt(variance);
}

double OrientationAnomalyDetector::CalculateAverageYaw() {
  if (history_.empty()) return 0.0;
  double sum_sin = 0.0, sum_cos = 0.0;
  for (const auto& state : history_) {
    double yaw = ExtractYaw(state.orientation);
    sum_sin += std::sin(yaw);
    sum_cos += std::cos(yaw);
  }
  return NormalizeAngle(std::atan2(sum_sin, sum_cos));
}

double OrientationAnomalyDetector::CalculateMaxHistoricalYawChange() {
  if (history_.size() < 2) return 0.0;
  
  double max_change = 0.0;
  for (size_t i = 1; i < history_.size(); ++i) {
    double yaw1 = ExtractYaw(history_[i-1].orientation);
    double yaw2 = ExtractYaw(history_[i].orientation);
    double change = std::abs(NormalizeAngle(yaw2 - yaw1));
    max_change = std::max(max_change, change);
  }
  return max_change;
}

ObjectState OrientationAnomalyDetector::GetMedianFilteredResult() {
  if (history_.empty()) {
    throw std::runtime_error("history is empty");
  }
  // sort by yaw stableity score
  std::vector<std::pair<double, ObjectState>> stability_scores;
  
  for (const auto& state : history_) {
    double stability = CalculateFrameStability(state);
    stability_scores.emplace_back(stability, state);
  }
  
  std::sort(stability_scores.begin(), stability_scores.end(),
            [](const auto& a, const auto& b) {
                return a.first > b.first; 
            });
  
  return stability_scores[0].second;
}

double OrientationAnomalyDetector::CalculateFrameStability(const ObjectState& state) {
    double state_yaw = ExtractYaw(state.orientation);
    double avg_yaw = CalculateAverageYaw();
    double yaw_diff = std::abs(NormalizeAngle(state_yaw - avg_yaw));
    double yaw_stability = 1.0 / (1.0 + yaw_diff);  
    double speed = std::sqrt(state.velocity[0] * state.velocity[0] + 
                            state.velocity[1] * state.velocity[1]);
    double speed_stability = 1.0 / (1.0 + speed);   
    return yaw_stability * 0.7 + speed_stability * 0.3; 
}

bool MlfShapeFilter::Init(const MlfFilterInitOptions& options) {
  std::string config_file = "mlf_shape_filter.conf";
  if (!options.config_file.empty()) {
    config_file = options.config_file;
  }
  config_file = GetConfigFile(options.config_path, config_file);
  MlfShapeFilterConfig config;
  ACHECK(century::cyber::common::GetProtoFromFile(config_file, &config));

  bottom_points_ignore_threshold_ = config.bottom_points_ignore_threshold();
  top_points_ignore_threshold_ = config.top_points_ignore_threshold();
  return true;
}

void MlfShapeFilter::UpdateWithObject(const MlfFilterOptions& options,
                                      const MlfTrackDataConstPtr& track_data,
                                      TrackedObjectPtr new_object) {
  // compute tight object polygon
  auto& obj = new_object->object_ptr;
  if (new_object->is_background) {
    hull_.GetConvexHull(obj->lidar_supplement.cloud_world, &obj->polygon);
  } else {
    hull_.GetConvexHullWithoutGroundAndHead(
        obj->lidar_supplement.cloud_world,
        static_cast<float>(bottom_points_ignore_threshold_),
        static_cast<float>(top_points_ignore_threshold_), &obj->polygon);
  }

  // window filter for direction
  auto mersure_direction = new_object->direction;
  if (track_data->age_ > 0) {
    TrackedObjectConstPtr latest_object = track_data->GetLatestObject().second;
    if (new_object->direction.dot(latest_object->direction) < 0) {
      new_object->direction *= -1;
      mersure_direction *= -1;
    }
    
    std::vector<TrackedObjectConstPtr> history_objects;
    track_data->GetLatestKObjects(50, &history_objects);
    if (!history_objects.size()) {
        AWARN << "No history objects find. Unreasonable";
        return;
    }
    std::shared_ptr<OrientationAnomalyDetector> detector = std::make_shared<OrientationAnomalyDetector>();
    detector->SetParameters(20, 0.1, 2.0, track_data->stable_direction_); // history 50, vel 0.3m/s, yaw 2 degree
    for (const auto& hist_obj : history_objects) {
      detector->AddToHistory(ObjectState{
        hist_obj->output_center,
        hist_obj->output_velocity,
        hist_obj->direction,
        hist_obj->timestamp,
        true
      });
    }
    bool accepted = detector->ProcessNewDetection(ObjectState{
      new_object->center,
      new_object->output_velocity,
      new_object->direction,
      new_object->timestamp,
      true
    });

    if (detector->get_update_stable_dir()) {
      new_object->updated_stable_direction = true;
      new_object->stable_direction =  Eigen::Vector3d(cos(detector->get_stable_angle()), sin(detector->get_stable_angle()), 0.0);
      new_object->stable_direction.normalize();
    }
    if (!accepted && track_data->stable_direction_ != Eigen::Vector3d::Zero() ) {
        // use history
        // ObjectState stable_state = detector->GetMostStableResult();
        // new_object->direction = stable_state.orientation;
      mersure_direction = track_data->stable_direction_;
    } else {
      // simple moving average orientation filtering
      // if (CalculateAngleDifference(new_object->direction, latest_object->direction) > M_PI / 6) {
      //     mersure_direction = latest_object->direction;
      // }
      if (track_data->stable_direction_ == Eigen::Vector3d::Zero()) {
        static const double kMovingAverage = 0.6;
         mersure_direction = latest_object->output_direction * (1 - kMovingAverage) +
        new_object->direction * kMovingAverage;
        mersure_direction.normalize();
      } else {
        mersure_direction = track_data->stable_direction_;
      }
  
      // new_object->direction =
      //   latest_object->output_direction * (1 - kMovingAverage) +
      //   new_object->direction * kMovingAverage;
      
    }
  }

  Eigen::Vector3f local_direction = obj->direction;
  Eigen::Vector3d local_center = obj->center;
  Eigen::Vector3f local_size = obj->size;
  obj->direction = mersure_direction.cast<float>();  // sync
  // finally, recompute object shape
  ComputeObjectShapeFromPolygon(obj, true);
  // new_object->center = obj->center;
  new_object->size = obj->size.cast<double>();
  // center and size in object should not changed
  obj->center = local_center;
  obj->size = local_size;
  obj->direction = local_direction;
  if (new_object->type == base::ObjectType::VEHICLE && new_object->output_velocity.norm() < 0.1) {
    new_object->output_direction = mersure_direction;
    auto corners = GenerateBottomSurfaceCorners(
        new_object->center, new_object->size(0), new_object->size(1), new_object->size(2), new_object->output_direction(1), new_object->output_direction(0));
    new_object->polygon_points.clear();
    for (int i = 0; i < 4; ++i) {
      new_object->polygon_points.emplace_back(corners[i][0], corners[i][1], corners[i][2]); 
    }
  }
}

void MlfShapeFilter::UpdateWithoutObject(const MlfFilterOptions& options,
                                         double timestamp,
                                         MlfTrackDataPtr track_data) {
  // TODO(.)
}

PERCEPTION_REGISTER_MLFFILTER(MlfShapeFilter);

}  // namespace lidar
}  // namespace perception
}  // namespace century
