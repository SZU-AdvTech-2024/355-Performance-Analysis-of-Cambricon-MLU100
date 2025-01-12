#include <iostream>
#include "cnrt.h"
#include "Launch.hpp"

const std::string SOURCE_MODEL_PATH = 
    "/home/szu/cambricon/tensorflow/models/offline/";
const std::string DEVICE = "_MLU100_";


int main(){
    cnrtInit(0);
    std::string model_name = "vgg16";
    std::string core_number = "32";
    std::string batch_size = "1";
    std::string net = "subnet0";
    
    int dev_id = 0;

    std::string model_path = SOURCE_MODEL_PATH +
                            model_name + "/" + model_name + 
                            DEVICE + core_number + 
                            "core_" + batch_size + "batch.cambricon";

    // std::cout << model_path << std::endl;
    // CNRT_CHECK(cnrtLoadModel())
    // std::cout << "hello world" << std::endl;
    
    Launch* launch = new Launch(model_path, dev_id, net);
    int memory_size = launch->getModelMemorySize();
    int64_t memory_used = launch->getModelMemoryUsed();
    std::cout << "model path: " << launch->model_path_ << std::endl;
    std::cout << "memory size: " << memory_size << std::endl;
    std::cout << "memory used: " << memory_used / (1024.0*1024.0) << "MB" << std::endl;
    std::cout << "cnrt_dev_t: " << launch->cnrt_dev_t_ << std::endl;
    launch->execute();
    delete launch; 
    
    cnrtDestroy();
    return 0;

}