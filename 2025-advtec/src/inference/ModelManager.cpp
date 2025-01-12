#include "ModelManager.hpp"

ModelManager::ModelManager(){}

ModelManager::ModelManager(cnrtModel_t cnrt_model_t, std::string network, int device_id):
        cnrt_model_t_(cnrt_model_t), network_(network), device_id_(device_id){
//    loadModel();
    getModelParamllelism();
    createFunction();
    extractFunction();
    getIODataDesciption();
    setDevice();
}

//ModelManager::ModelManager(std::string model_path, std::string network, int device_id):
//    model_path_(model_path), network_(network), device_id_(device_id){
//    loadModel();
//    getModelParamllelism();
//    createFunction();
//    extractFunction();
//    getIODataDesciption();
//    setDevice();
//}

ModelManager::~ModelManager(){
    std::cout << "ModelManager deconstructor" << std::endl;
    cnrtDestroyFunction(cnrt_function_t_);
//    unloadModel();
}

void ModelManager::loadModel(){
    CNRT_CHECK(cnrtLoadModel(&cnrt_model_t_, model_path_.c_str()));
}

void ModelManager::unloadModel(){
    CNRT_CHECK(cnrtUnloadModel(cnrt_model_t_));
}

int ModelManager::getModelParamllelism(){

    CNRT_CHECK(cnrtQueryModelParallelism(cnrt_model_t_, &model_parallelism_));
    return model_parallelism_;
}

void ModelManager::createFunction(){
    CNRT_CHECK(cnrtCreateFunction(&cnrt_function_t_));
}

void ModelManager::extractFunction(){
    CNRT_CHECK(cnrtExtractFunction(&cnrt_function_t_, cnrt_model_t_, network_.c_str()));
}

void ModelManager::getIODataDesciption(){
    CNRT_CHECK(cnrtGetInputDataDesc(&cnrt_input_data_desc_array_t_, &input_num_, cnrt_function_t_));
    CNRT_CHECK(cnrtGetOutputDataDesc(&cnrt_output_data_desc_array_t_, &output_num_, cnrt_function_t_));
}

void ModelManager::setDevice(){
    CNRT_CHECK(cnrtGetDeviceHandle(&cnrt_dev_t_, device_id_));
    CNRT_CHECK(cnrtSetCurrentDevice(cnrt_dev_t_));
}