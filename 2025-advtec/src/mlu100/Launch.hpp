#ifndef LAUNCH_HPP
#define LAUNCH_HPP

#include <iostream>
#include "cnrt.h"
#include "ImageReader.hpp"


class Launch{
public:
    Launch(const std::string &model_path, const int dev_id, const std::string &net);
    ~Launch();
    
    void setDevice();
    void loadModel(const std::string& model_path);
    void unloadModel();
    void createFunction();
    void createRuntimeContext();
    void getFunctionDataDesciption();
    void cpuMalloc();
    void mluMalloc();
    void copyDataHostToDevice();
    void prepareInvokeFunctionParam();

    void prepareParam();
    void invokeFunction();
    void copyDataDeviceToHost();

    void freeQueueAndContextAndFunction();
    void freeCpuAndMlu();

    void execute();

    int64_t getModelMemoryUsed();
    int getModelMemorySize();
    
    // 一个task/box带有的参数
    std::string model_path_;        // 指定模型路径
    int dev_id_;                    // 指定模型执行设备号—ordinal
    std::string net_;                // symbol name-->标志符号
    cnrtDataType_t cnrt_data_type_t_;    // 指定数据类型是INT8 还是 FLOAT32
    std::vector<int> task_vector_;  // 指定任务列表大小，每次为执行的任务分配一个batch大小，和待处理图片列表
    int batch_size_;                // 这批任务的批次大小——同dataparallelism?
    int core_number_;               // 这批任务用到的核心数
    int input_data_count_;          // 输入数据个数
    int output_data_count_;         // 输出数据个数
    float duration;

    // 读取图片
    ImageReader image_reader_;

    // 一些想查看的参数
    int64_t model_memory_used_;     // model 实际占用空间
    int model_memory_size_;         // model 模型大小

    // cnrt相关
    cnrtModel_t cnrt_model_t_;      // model pointer
    cnrtDev_t cnrt_dev_t_;          // model device num(u64_t)
    cnrtFunction_t cnrt_function_t_;// model fuction pointer
    cnrtRuntimeContext_t cnrt_runtime_context_t_;        // runtime context pointer
    cnrtQueue_t cnrt_queue_t_;      // cnrt queue pointer
    cnrtDataDescArray_t cnrt_input_data_desc_array_t_;  // input data desc array pointer
    cnrtDataDescArray_t cnrt_output_data_desc_array_t_; // output data desc array pointer
    int cnrt_input_data_desc_array_length_;             // input data desc array length
    int cnrt_output_data_desc_array_length_;            // output data desc array length
    cnrtInitFuncParam_t cnrt_init_func_param_t_;    // init fuction settings
    cnrtInvokeFuncParam_t cnrt_invoke_param_;       // incoke function settings   
    void** cnrt_runtime_param_;                     // runtime param                    
    cnrtNotifier_t cnrt_notifier_start_t_;          // count time notifier start
    cnrtNotifier_t cnrt_notifier_end_t_;            // count time nofifier end

    // data store pointer
    void** cpu_input_data_array_t_;     // cpu input data array pointer
    void** cpu_output_data_array_t_;    // cpu output data array pointer
    void** mlu_input_data_array_t_;     // mlu input data array pointer
    void** mlu_output_data_array_t_;    // mlu output data array pointer
    uint input_n_, input_c_, input_h_, input_w_;    // input data 的 nchw  
    uint output_n_, output_c_, output_h_, output_w_;// output data 的 nchw
};



#endif //LAUNCH_HPP