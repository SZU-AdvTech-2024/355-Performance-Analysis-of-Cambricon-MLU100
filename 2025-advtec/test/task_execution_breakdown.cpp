#include <iostream>
#include "../src/inference/ModelManager.hpp"
#include <memory>
#include "../src/inference/ContextManager.hpp"
#include "util/CreateModelUtil.hpp"
int main(){
    CNRT_CHECK(cnrtInit(0));
    std::string model_name = "resnet152_v2";
    std::string device = "_MLU100_";
    std::string core = "32";
    std::string batch = "1";
    std::string model_path = 
        "/home/szu/cambricon/tensorflow/models/offline/"+
        model_name +
        "/" +
        model_name +
        device +
        core +
        "core_" +
        batch +
        "batch.cambricon";
    std::string network = "subnet0";

    std::string image_file_path = "../src/config/inputimage.txt";

    int device_id = 0;
    cnrtModel_t  cnrt_model_t = CreateModelUtil::createModel(model_path);

    ModelManager model_manager(cnrt_model_t, network, device_id);
    // std::shared_ptr<ModelManager> model_manager_ptr = std::make_shared<ModelManager>(model_manager);
    std::cout << model_manager.getModelParamllelism() << std::endl;
    ContextManager context_manager(model_manager, image_file_path, std::stoi(batch));
    // std::unique_ptr<ContextManager> context_manager_ptr = std::make_unique<ContextManager>(context_manager);
    // std::cout << network << std::endl;

    CreateModelUtil::unloadModel(cnrt_model_t);
    cnrtDestroy();
}
