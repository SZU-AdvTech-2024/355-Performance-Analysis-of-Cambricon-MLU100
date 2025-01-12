#include "ImageReader.hpp"
#include "../lib/ThreadPool/ThreadPool.h"

ImageReader::ImageReader(){}

/**
 * @brief 构造函数
 * @param image_task_file: 待加载的配置文件名
*/
ImageReader::ImageReader(const std::string& image_task_file){
    readFile(image_task_file);
}


/**
 * @brief 从配置文件中读图片路径
 * @param image_task_file: 需加载的配置文件名
*/
void ImageReader::readFile(const std::string& image_task_file){
    std::ifstream f_stream(image_task_file, std::ios::in);
    std::string line;
    while(std::getline(f_stream, line)){
        image_task_path_vector_.push_back(line);
    }
    f_stream.close();
}

// void ImageReader::sample(cv::Mat processed, const cv::Mat& origin){
//     if(origin.channels == 3)
// }


/**
 * @brief 转换至mlu格式
*/
void ImageReader::convertToMluFormat(
    float* input_data, uint input_n, uint input_c, uint input_h, uint input_w){
    int i = 0;
    int turn = 0;
    for(auto& e : image_task_path_vector_){
        
        // 1. 读取图片
        cv::Mat image = cv::imread(e);
        if(image.empty()){
            std::cerr << "Image not found!" << std::endl;
            exit(-1);
        }

        
        // 2. 颜色转换，根据模型所需的channel来；opencv默认是BGR
        if(image.channels() == 3 && input_c == 1){
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        }else if(image.channels() == 4 && input_c == 1){
            cv::cvtColor(image, image, cv::COLOR_BGRA2GRAY);
        }else if(image.channels() == 4 && input_c == 3){
            cv::cvtColor(image, image, cv::COLOR_BGRA2BGR);
        }else if(image.channels() == 1 && input_c == 3){
            cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        }else if(image.channels() == 3 && input_c == 3){
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        }else{
            // 不需要进行颜色转换
        }

        // 3. 调整尺寸
        cv::resize(image, image, cv::Size(input_w, input_h));

        // 4. 归一化
        image.convertTo(image, CV_32FC3);

        // 5. 将图像数据从HWC转换成NCHW，并填充至input_data数组中
        int offset = turn * input_c * input_h * input_w;
        for(int c = 0; c < input_c; ++c){
            for(int h = 0; h < input_h; ++h){
                for(int w = 0; w < input_w; ++w){
                    int index = offset + c*h*w + h*w + w;
                    input_data[index] = image.at<cv::Vec3f>(h,w)[c];
                }
            }
        }
        // batch++
        turn++;
    }
}

//float* ImageReader::preprocess(
//    const std::vector<std::string>& image_vec,
//    Shape input_shape,
//    int input_data_count){
//    ThreadPool pool(4);
//    std::vector<std::future<std::vector<float>>> results;
//    for(int i = 0; i < image_vec.size(); ++i){
//        std::cout << "i: " << i << std::endl;
//        std::string image_path = image_vec[i];
//        std::cout << "image path: " << image_path << std::endl;
//        results.emplace_back(pool.enqueue([image_path, input_shape, input_data_count]{
//            std::this_thread::sleep_for(std::chrono::seconds(1));
//            std::cout << "i: " << input_data_count << std::endl;
//            uint n = input_shape.n_;
//            uint c = input_shape.c_;
//            uint h = input_shape.h_;
//            uint w = input_shape.w_;
//
//            cv::Mat mat_image = cv::imread(image_path, cv::IMREAD_COLOR);
//            if(mat_image.channels() == 3 && c == 3){
//                cv::cvtColor(mat_image, mat_image, cv::COLOR_BGR2RGB);
//            }
//            if(mat_image.channels() == 3)
//                mat_image.convertTo(mat_image, CV_32FC3);
//            cv::resize(mat_image, mat_image, cv::Size(h, w));
//
//            std::vector<float> res(input_data_count);
//            for(int cc = 0; cc < c; ++cc){
//                for(int hh = 0; hh < h; ++hh){
//                    for(int ww = 0; ww < w; ++ww){
//                        int index = cc * h * w + hh * w + ww;
//                        res[index] = mat_image.at<cv::Vec3f>(hh, ww)[cc];
//                    }
//                }
//            }
//            return res;
//        }));
//    }
//    // float* cpu_ptr;
//    std::vector<float> cpu_ptr_vector(input_data_count * image_vec.size());
//    for(auto& future_res: results){
//        auto processed = future_res.get();
//        cpu_ptr_vector.insert(cpu_ptr_vector.end(), processed.begin(), processed.end());
//    }
//    for(auto& e: cpu_ptr_vector){
//        std::cout << e;
//    }
//    return cpu_ptr_vector.data();
//}

float* ImageReader::preprocess1(
        const std::vector<std::string>& image_vec,
        Shape input_shape,
        int input_data_count
        ){

}