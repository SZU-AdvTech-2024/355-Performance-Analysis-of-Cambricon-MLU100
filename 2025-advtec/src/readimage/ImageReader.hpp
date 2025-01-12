#ifndef IMAGEREADER_HPP
#define IMAGEREADER_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "Shape.hpp"

class ImageReader{
public:
    ImageReader();
    ImageReader(const std::string& image_task_file);

    void readFile(const std::string& image_task_file);

    void convertToMluFormat(
        float* image_data, uint input_n, uint input_c, uint input_h, uint input_w);

    // void sample(cv::Mat processed, const cv::Mat& origin);

//    float* preprocess(const std::vector<std::string>& image_vec, Shape input_shape, int input_data_count);
    float* preprocess1(const std::vector<std::string>& image_vec, Shape input_shape, int input_data_count);
    std::vector<std::string> image_task_path_vector_;
};

#endif //IMAGEREADER_HPP
