#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>  

// 函数用于读取二进制文件并转换为PCL格式  
pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointCloudFromBin(const std::string& filename) {  
    std::ifstream file(filename, std::ios::binary);  
    if (!file.is_open()) {  
        std::cerr << "Could not open the binary file: " << filename << std::endl;  
        return nullptr;  
    }  

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);  

    // 文件流读取每点数据  
    pcl::PointXYZ point;  
    while (file.read(reinterpret_cast<char*>(&point), sizeof(pcl::PointXYZ))) {  
        cloud->points.push_back(point);  
    }  
    cloud->width = cloud->size();  
    cloud->height = 1; // 表示单个点云  
    cloud->is_dense = false;

    file.close();  
    return cloud;  
}  

int main(int argc, char** argv) {  
    // 确保传入二进制文件的正确路径  
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_bin_file>" << std::endl;  
        return -1;  
    }  

    std::string bin_file_path = argv[2];

    // 读取点云数据  
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = loadPointCloudFromBin(bin_file_path);  
    if (!cloud) {  
        return -1;  
    }  

    // 可视化点云数据  
    pcl::visualization::CloudViewer viewer("Cloud Viewer");  
    viewer.showCloud(cloud);  

    while (!viewer.wasStopped()) {  
        // 循环让可视化保持打开状态  
    }  

    return 0;  
}
