#ifndef MODELMANAGER_HPP
#define MODELMANAGER_HPP

#include <iostream>
#include "cnrt.h"

class ModelManager{
public:
// variant member
    
    // cnrt model
    std::string model_path_;        // 存储模型路径
    cnrtModel_t cnrt_model_t_;      // 获得模型指针
    int model_parallelism_;         // 模型并行度
    
    // cnrt set device
    cnrtDev_t cnrt_dev_t_;
    int device_id_;

    // cnrt function
    cnrtFunction_t cnrt_function_t_;// 从模型中提取函数的指针

    // network
    std::string network_;            // 要提取的网络类型，对于classification来说是"subnet0"

    // cnrt get function I/O's data description
    int input_num_;                                 // 输入描述符数组的大小，通常为1，目前无例外
    int output_num_;                                // 输出描述符数组的大小，通常为1，目前无例外
    cnrtDataDescArray_t cnrt_input_data_desc_array_t_;   // 指向输入描述符数组的指针
    cnrtDataDescArray_t cnrt_output_data_desc_array_t_;  // 指向输出描述符数组的指针

    

// function member
    // constructor;
    ModelManager();
    ModelManager(cnrtModel_t cnrt_model_t, std::string network, int device_id);

    // deconstructor;
    ~ModelManager();

    // cnrt model
    void loadModel();               // 装载模型
    void unloadModel();             // 卸载模型
    int getModelParamllelism();     // 获得模型并行度：即core_num/batch_num >= 1

    // cnrt function
    void createFunction();          // 创建函数提取描述符
    void extractFunction();         // 从模型中提取函数，函数提取描述符

    // cnrt get io data description
    void getIODataDesciption();     // 提取IO数据描述符

    // cnrt set device
    void setDevice();               // 设置设备id
};

#endif