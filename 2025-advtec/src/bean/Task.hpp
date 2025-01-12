#ifndef TASK_HPP
#define TASK_HPP
#include <iostream>
#include "cnrt.h"
class Task{
public:
    int core_number_;           // 该任务所占核心数
    int batch_size_;            // 该任务的批次大小
    std::string model_name_;    // 任务模型
    int device_id_;             // 该任务分到哪个设备
    std::string net_;           // 图像分类——subnet0
};

#endif // TASK_HPP