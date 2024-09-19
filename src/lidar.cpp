//
// Created by cx on 24-9-19.
//

#include "lidar.h"


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