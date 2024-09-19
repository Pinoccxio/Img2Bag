//
// Created by cx on 24-9-19.
//


#include <iostream>
#include <string>
#include <experimental/filesystem>
#include <algorithm>
#include <exception>
#include "utils.hpp"

#include <ros/ros.h>
#include <rosbag/bag.h>

#include <sensor_msgs/Image.h>
// #include <opencv4/opencv2/core/core.hpp>
// #include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <std_msgs/Time.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointField.h>


using namespace std;
using namespace sensor_msgs;
namespace fs = experimental::filesystem;


namespace nuscenes2bag{
auto topic_convertion(string topicName) {
    // Turn into Lowercase
    for (char& c : topicName) {
        if (c >= 'A' && c <= 'Z') {
            // Add 32 to convert uppercase to lowercase in ASCII
            c += 32;
        }
    }
    string topic;
    if (topicName == "cam_front"  or topicName == "cam_front_left" or topicName == "cam_front_right" or topicName == "cam_back" or topicName == "cam_back_left" or topicName == "cam_back_right")
    {topic = topicName + "/raw";}
    else if (topicName == "lidar_top")
    {topic = topicName;}
    else{topic = "0";}
    return topic;
}

void readBinFile(const string &bin_file, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
    ifstream file(bin_file, ios::binary);

    if (!file) {
        ROS_ERROR("Could not open BIN file.");
        return;
    }

    // 假设每个点包含 x, y, z 三个浮点数
    vector<float> data((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    size_t num_points = data.size() / 3;

    cloud->width = num_points;
    cloud->height = 1; // 1 for unorganized
    cloud->points.resize(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        cloud->points[i].x = data[i * 3];
        cloud->points[i].y = data[i * 3 + 1];
        cloud->points[i].z = data[i * 3 + 2];
    }
}





inline void fillFieldsForPointcloud(std::vector<PointField>& fields)
{
    PointField field;
    field.datatype = PointField::FLOAT32;
    field.offset = 0;
    field.count = 1;
    field.name = std::string("x");
    fields.push_back(field);

    field.datatype = PointField::FLOAT32;
    field.offset = 4;
    field.count = 1;
    field.name = std::string("y");
    fields.push_back(field);

    field.datatype = PointField::FLOAT32;
    field.offset = 8;
    field.count = 1;
    field.name = std::string("z");
    fields.push_back(field);

    field.datatype = PointField::FLOAT32;
    field.offset = 12;
    field.count = 1;
    field.name = std::string("intensity");
    fields.push_back(field);
}


union
{
    float value;
    uint8_t byte[4];
} floatToBytes;

inline void push_back_float32(std::vector<uint8_t>& data, float float_data)
{
    floatToBytes.value = float_data;
    data.push_back(floatToBytes.byte[0]);
    data.push_back(floatToBytes.byte[1]);
    data.push_back(floatToBytes.byte[2]);
    data.push_back(floatToBytes.byte[3]);
}




inline std::vector<float> readBinaryPcdFile(std::ifstream& fin)
{
    std::vector<float> fileValues;
    uint8_t skipCounter = 0;
    float f;
    while (fin.read(reinterpret_cast<char*>(&f), sizeof(float))) {
        // skip 5th value of each point
        if (skipCounter < 4) {
            fileValues.push_back(f);
            skipCounter++;
        } else {
            skipCounter = 0;
        }
    }

    return fileValues;
}


boost::optional<PointCloud2> readLidarFile(const fs::path& filePath)
{

    PointCloud2 cloud;
    cloud.header.frame_id = std::string("lidar");
    cloud.is_bigendian = false;
    cloud.point_step = sizeof(float) * 4; // Length of each point in bytes
    cloud.height = 1;

    try {
        std::ifstream fin(filePath.string(), std::ios::binary);
        const std::vector<float> fileValues = readBinaryPcdFile(fin);

        if (fileValues.size() % 4 != 0) {
            throw UnableToParseFileException(filePath.string());
        }
        const size_t pointsNumber = fileValues.size() / 4;
        cloud.width = pointsNumber;

        std::vector<uint8_t> data;
        for (auto float_data : fileValues) {
            push_back_float32(data, float_data);
        }

        fillFieldsForPointcloud(cloud.fields);
        cloud.data = data;
        cloud.row_step = data.size(); // Length of row in bytes

    } catch (const std::exception& e) {
        PRINT_EXCEPTION(e);

        return boost::none;
    }

    return boost::optional<PointCloud2>(cloud);
}
}

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "Img2Ros");
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <root>" << std::endl;
        return -1;
    }
    const string rootPath = argv[1];
    std::cout << "rootPath:" << rootPath << std::endl;

    ros::NodeHandle n;
    if (n.ok())
    {
        ros::start();

        if (!fs::is_directory(rootPath)) {
            cerr << "给定的路径不是一个有效的文件夹: " << rootPath << endl;
            return -1;
        }

        rosbag::Bag bag_out("/home/cx/dataset/bag/test_1.bag", rosbag::bagmode::Write);
        ros::Time t = ros::Time::now();
//        ros::Time t_0 = ros::Time::now();
        double freq = 10;
        const float T = 1.0f/freq;
        ros::Duration d(T);


        // Traverse every folder in ROOT
        for (const auto& folder : fs::directory_iterator(rootPath)) {
            if (!fs::is_directory(folder)) {
                cout << folder << "not a dir" << endl;
            }
            else
            {
                string topic = nuscenes2bag::topic_convertion(folder.path().stem());
                vector<fs::path> files;

                if (topic == "0") { continue; }
                for (const auto &file : fs::directory_iterator(folder)) {
                    if (!fs::is_regular_file(file)){
                        cout << file << "isn't a regular file" << endl;
                        continue;
                    }
                    string file_path = file.path().string();
                    files.emplace_back(file_path);
                    sort(files.begin(), files.end(), [](const fs::path &a, const fs::path &b) { return a.filename() < b.filename(); });
                }
                cout << "Reading " << folder << endl;
                cout << "Topic is " << topic << endl;
                ros::Time t_0 = t;
                if (topic == "lidar_top") {
                    for (const auto &file : files) {
                        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
                        nuscenes2bag::readBinFile(file, cloud);   //函数

                        // 转换为 ROS 消息
                        PointCloud2 cloud_msg;
                        pcl::toROSMsg(*cloud, cloud_msg);
                        cloud_msg.header.stamp = t_0;
                        cloud_msg.header.frame_id = "map"; // 根据需要设置框架ID

                        bag_out.write(topic, ros::Time(t_0), cloud_msg);
                        t_0+=d;
                    }
                    cout << topic << " done" << endl;
                    continue;
                }

                for (const auto &file : files){

                    cv::Mat im = cv::imread(file,cv::IMREAD_COLOR);
                    cv_bridge::CvImage cvImage;
                    cvImage.image = im;
                    cvImage.encoding = image_encodings::BGR8;
                    cvImage.header.stamp = t_0;
                    bag_out.write(topic,ros::Time(t_0),cvImage.toImageMsg());
                    t_0+=d;
                }
                cout << topic << " done" << endl;
            }
        }
        bag_out.close();
        ros::shutdown();
    }

    return 0;
}