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
#pragma once

#include <string>

#include "modules/perception/common/algorithm/geometry/convex_hull_2d.h"
#include "modules/perception/lidar_tracking/tracker/multi_lidar_fusion/mlf_base_filter.h"

namespace century {
namespace perception {
namespace lidar {

class MlfShapeFilter : public MlfBaseFilter {
 public:
  MlfShapeFilter() = default;
  virtual ~MlfShapeFilter() = default;
  /**
   * @brief Init mlf filter
   *
   * @param options
   * @return true
   * @return false
   */
  bool Init(
      const MlfFilterInitOptions& options = MlfFilterInitOptions()) override;
  /**
   * @brief Updating shape filter with object
   *
   * @param options for updating
   * @param track_data not include new object
   * @param new_object new object for updating
   */
  void UpdateWithObject(const MlfFilterOptions& options,
                        const MlfTrackDataConstPtr& track_data,
                        TrackedObjectPtr new_object) override;
  /**
   * @brief Updating shape filter without object
   *
   * @param options for updating
   * @param timestamp current timestamp
   * @param track_data track data to be updated
   */
  void UpdateWithoutObject(const MlfFilterOptions& options, double timestamp,
                           MlfTrackDataPtr track_data) override;
  /**
   * @brief Get class name
   *
   * @return std::string
   */
  std::string Name() const override { return "MlfShapeFilter"; }

 protected:
  algorithm::ConvexHull2D<base::PointDCloud, base::PolygonDType> hull_;
  double bottom_points_ignore_threshold_ = 0.1;
  double top_points_ignore_threshold_ = 1.6;
};  // class MlfShapeFilter


struct ObjectState {
    Eigen::Vector3d position;      // loc (x, y, z)
    Eigen::Vector3d velocity;      // velocity (vx, vy, vz)
    Eigen::Vector3d orientation;   // orentation  (y_component, x_component, ignore)
    double timestamp;              // time
    bool is_valid;                 
};

class OrientationAnomalyDetector {
 private:
  std::vector<ObjectState> history_;  // history queue
  size_t max_history_size_ = 10;      // max his size
  double velocity_threshold_ = 0.1;   // static velocity threshold
  double yaw_change_threshold_ = 30.0 * M_PI / 180.0; // yaw change threshold (radian)
  std::vector<double> yaws_;
  bool update_stable_dir_ = false;
  double stable_angle_ = -1.0; 
 public:
  void SetParameters(size_t history_size, double vel_threshold, double yaw_threshold_deg,
      const Eigen::Vector3d last_stable_direction);

  bool get_update_stable_dir() const {
    return update_stable_dir_;
  }

  double get_stable_angle() const {
    return stable_angle_;
  }

  void AddToHistory(const ObjectState& new_state) {
    history_.push_back(new_state);
    if (history_.size() > max_history_size_) {
        history_.erase(history_.begin());
    }
  }
  
  bool ProcessNewDetection(const ObjectState& new_detection);
    
  ObjectState GetMostStableResult();

  size_t GetHistorySize() const {
    return history_.size();
  }

  void ClearHistory() {
    history_.clear();
  }

private:

  double ExtractYaw(const Eigen::Vector3d& orientation, bool norm = false) const;

  bool CheckIfMainlyStatic();

  bool CalculateAngleCluster(int use_ange_size, std::pair<double, int>& max_cluster);

  bool CheckYawAnomaly(const ObjectState& new_detection);
  
  double NormalizeAngle(double angle) const;
  
  double CalculateYawStdDev();
  
  double CalculateAverageYaw();
  
  double CalculateMaxHistoricalYawChange();
  
  ObjectState GetMedianFilteredResult();
  
  double CalculateFrameStability(const ObjectState& state);
};


}  // namespace lidar
}  // namespace perception
}  // namespace century
