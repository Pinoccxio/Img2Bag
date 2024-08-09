// #include <ros/ros.h>
// #include <rosbag/bag.h>
// #include <sensor_msgs/PointCloud2.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <sensor_msgs/conversion.h>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <experimental/filesystem>
// using namespace std;
//
// void readBinFile(const string &bin_file, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
//     ifstream file(bin_file, ios::binary);
//
//     if (!file) {
//         ROS_ERROR("Could not open BIN file.");
//         return;
//     }
//
//     // 假设每个点包含 x, y, z 三个浮点数
//     vector<float> data((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
//     size_t num_points = data.size() / 3;
//
//     cloud->width = num_points;
//     cloud->height = 1; // 1 for unorganized
//     cloud->points.resize(num_points);
//
//     for (size_t i = 0; i < num_points; ++i) {
//         cloud->points[i].x = data[i * 3];
//         cloud->points[i].y = data[i * 3 + 1];
//         cloud->points[i].z = data[i * 3 + 2];
//     }
// }
//
// void binToRosbag(const string &bin_file, const string &bag_file) {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
//     readBinFile(bin_file, cloud);
//
//     // 转换为 ROS 消息
//     sensor_msgs::PointCloud2 cloud_msg;
//     pcl::toROSMsg(*cloud, cloud_msg);
//     cloud_msg.header.stamp = ros::Time::now();
//     cloud_msg.header.frame_id = "map"; // 根据需要设置框架ID
//
//     // 写入 bag 文件
//     rosbag::Bag bag;
//     bag.open(bag_file, rosbag::bagmode::Write);
//     bag.write("/point_cloud", cloud_msg.header.stamp, cloud_msg);
//     bag.close();
// }
//
// int main(int argc, char **argv) {
    ros::init(argc, argv, "bin_to_rosbag");
    ros::NodeHandle nh;

    string bin_file = "path/to/your/file.bin"; // 二进制文件路径
    string bag_file = "output.bag"; // 输出 bag 文件路径

    binToRosbag(bin_file, bag_file);
    return 0;
}