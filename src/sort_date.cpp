#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

void organizeFiles(const fs::path& sourceFolder, const fs::path& destinationFolder, const std::string topic) {
    // 创建目标文件夹，如果它不存在
    if (!fs::exists(destinationFolder)) {
        fs::create_directory(destinationFolder);
        std::cout << "create BaseFolder:" << destinationFolder << std::endl;
    }

    // 遍历源文件夹中的所有文件
    for (const auto& entry : fs::directory_iterator(sourceFolder)) {
        std::cout << "filename:" << entry << std::endl;
        if (entry.is_regular_file() && ( entry.path().extension() == ".bin" || entry.path().extension() == ".jpg" )) {
            std::string fileName = entry.path().filename().string();
            // 提取日期字段
            size_t start = fileName.find('-') + 1; // 跳过第一个'-'
            std::cout << "start:" << start << std::endl;
            size_t end = fileName.find('-', start + 2); // 找到第二个'-'
            std::cout << "end:" << end << std::endl;
            std::string dateKey = fileName.substr(start, end - start + 6); // 获取日期部分
            std::cout << "date:" << dateKey << std::endl;
            // 创建以日期命名的子文件夹
            fs::path dateFolder = destinationFolder / dateKey;
            if (!fs::exists(dateFolder)) {
                fs::create_directory(dateFolder);
                std::cout << "create dateFolder:" << dateFolder << std::endl;
            }
            fs::path dstFolder = dateFolder / topic;
            if (!fs::exists(dstFolder)) {
                fs::create_directory(dstFolder);
                std::cout << "create dstFolder:" << dateFolder << std::endl;
            }

            // 移动文件到对应的日期文件夹
            fs::copy(entry.path(), dstFolder / fileName);
        }
    }

    std::cout << topic << "文件已根据日期字段成功分类！" << std::endl;
}

int main(int argc, char** argv) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <rootFolder>" << " <dstFolder>" << std::endl;
        return -1;
    }
    fs::path srcFolder = argv[1];
    fs::path destinationFolder = argv[2];

    std::vector<std::string> topicList;
    topicList.push_back("CAM_BACK");
    topicList.push_back("CAM_BACK_LEFT");
    topicList.push_back("CAM_BACK_RIGHT");
    topicList.push_back("CAM_FRONT");
    topicList.push_back("CAM_FRONT_LEFT");
    topicList.push_back("CAM_FRONT_RIGHT");
    topicList.push_back("LIDAR_TOP");

    for (const auto topic : topicList){
        fs::path sourceFolder = srcFolder / topic;
        std::cout << "Opening folder:" << sourceFolder << std::endl;

        organizeFiles(sourceFolder, destinationFolder, topic);
    }

    return 0;
}