//
// Created by cx on 24-9-19.
//

#ifndef LIDAR_H
#define LIDAR_H


#include <iostream>
#include <string>
#include <experimental/filesystem>
#include <algorithm>
#include <exception>
#include "utils.hpp"

#include <ros/ros.h>
#include <rosbag/bag.h>

#include <sensor_msgs/Image.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointField.h>

#include <std_msgs/Time.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/image_encodings.h>

using namespace sensor_msgs;
using namespace std;


class lidar {
    void push_back_float32(std::vector<uint8_t>& data, float float_data);
    boost::optional<PointCloud2> readLidarFile(const fs::path& filePath);
};



#endif //LIDAR_H
