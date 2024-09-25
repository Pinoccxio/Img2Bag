//
// Created by cx on 24-9-19.
//

#include "lidar.h"

namespace lidar{

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
                throw nuscenes2bag::UnableToParseFileException(filePath.string());
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