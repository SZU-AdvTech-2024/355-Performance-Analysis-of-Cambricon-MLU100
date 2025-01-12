#include "Launch.hpp"
#include "cnrt.h"

/**
 * @brief Launch类构造函数
 * @param model_path: 传入模型路径
 * @param dev_id: 将要使用的设备id
 * @param net: 传入任务网络格式—分类任务是subnet0
*/

Launch::Launch(
    const std::string &model_path, const int dev_id, const std::string& net):
    model_path_(model_path), 
    dev_id_(dev_id),
    net_(net){
    setDevice();
    loadModel(model_path);
}

Launch::~Launch(){
    unloadModel();
    std::cout << "---deconstructor---" << std::endl;
}

int64_t Launch::getModelMemoryUsed(){
    cnrtGetModelMemUsed(cnrt_model_t_, &model_memory_used_);
    return model_memory_used_;
}

int Launch::getModelMemorySize(){
    CNRT_CHECK(cnrtGetModelSize(model_path_.c_str(), &model_memory_size_));
    return model_memory_size_;
}

void Launch::setDevice(){
    unsigned dev_number;
    CNRT_CHECK(cnrtGetDeviceCount(&dev_number));
    std::cout << "number of device: " << dev_number << std::endl;
    CNRT_CHECK(cnrtGetDeviceHandle(&cnrt_dev_t_, dev_id_));
    cnrtSetCurrentDevice(cnrt_dev_t_);
}

/**
 * @brief 加载模型，将模型的结构体指针传给cnrt_model_t_
 * @param model_path 传入模型路径
 * @return void 
*/
void Launch::loadModel(const std::string& model_path){   
        CNRT_CHECK(cnrtLoadModel(&cnrt_model_t_, model_path.c_str()));
    }

/**
 * @brief 创建cnrt模型函数指针，并从模型中提取函数
*/
void Launch::createFunction(){
    CNRT_CHECK(cnrtCreateFunction(&cnrt_function_t_));
    CNRT_CHECK(cnrtExtractFunction(&cnrt_function_t_, cnrt_model_t_, net_.c_str()));
}

void Launch::createRuntimeContext(){
    CNRT_CHECK(cnrtCreateRuntimeContext(&cnrt_runtime_context_t_, cnrt_function_t_, nullptr));
    CNRT_CHECK(cnrtSetRuntimeContextDeviceId(cnrt_runtime_context_t_, dev_id_));
    CNRT_CHECK(cnrtSetRuntimeContextChannel(cnrt_runtime_context_t_, CNRT_CHANNEL_TYPE_NONE));
    CNRT_CHECK(cnrtInitRuntimeContext(cnrt_runtime_context_t_, nullptr));
    CNRT_CHECK(cnrtRuntimeContextCreateQueue(cnrt_runtime_context_t_, &cnrt_queue_t_));
}

void Launch::getFunctionDataDesciption(){
    CNRT_CHECK(cnrtGetInputDataDesc(
        &cnrt_input_data_desc_array_t_,
        &cnrt_input_data_desc_array_length_,
        cnrt_function_t_
    ));
    CNRT_CHECK(cnrtGetOutputDataDesc(
        &cnrt_output_data_desc_array_t_,
        &cnrt_output_data_desc_array_length_,
        cnrt_function_t_
    )); 
}

/**
 * @brief 在cpu端为输入输出分配空间
*/

void Launch::cpuMalloc(){
    cpu_input_data_array_t_ = (void**)malloc(sizeof(void* ) * cnrt_input_data_desc_array_length_);
    cpu_output_data_array_t_ = (void**)malloc(sizeof(void* ) * cnrt_output_data_desc_array_length_);
    for(int i = 0; i < cnrt_input_data_desc_array_length_; ++i){
        // int input_data_count;
        uint w, h, c;
        cnrtDataDesc_t cnrt_input_data_desc_t = cnrt_input_data_desc_array_t_[i];
        CNRT_CHECK(
            cnrtSetHostDataLayout(cnrt_input_data_desc_t, cnrt_data_type_t_, CNRT_NCHW)
        );
        CNRT_CHECK(
            cnrtGetHostDataCount(cnrt_input_data_desc_t, &input_data_count_)
        );
        CNRT_CHECK(
            cnrtGetDataShape(cnrt_input_data_desc_t, &input_n_, &input_c_, &input_h_, &input_w_)
        );
        float* cpu_ptr = (float*)malloc(sizeof(float) * input_data_count_);
        image_reader_.convertToMluFormat(
            cpu_ptr , input_n_, input_c_, input_h_, input_w_);
        cpu_input_data_array_t_[i] = (void*)cpu_ptr;
    }
    for(int i = 0; i < cnrt_output_data_desc_array_length_; ++i){

        // int output_data_count;
        cnrtDataDesc_t cnrt_output_data_desc_t = cnrt_output_data_desc_array_t_[i];
        CNRT_CHECK(
            cnrtSetHostDataLayout(cnrt_output_data_desc_t, cnrt_data_type_t_, CNRT_NCHW)
        );
        CNRT_CHECK(
            cnrtGetHostDataCount(cnrt_output_data_desc_t, &output_data_count_)
        );
        float* cpuPtr = (float*)malloc(sizeof(float) * output_data_count_);
        cpu_output_data_array_t_[i] = (void*)cpuPtr;
    }
}

/**
 * @brief 在mlu端为输入和输出分配空间
*/
void Launch::mluMalloc(){
    mlu_input_data_array_t_ = (void**)malloc(sizeof(void*) * cnrt_input_data_desc_array_length_);
    mlu_output_data_array_t_ = (void**)malloc(sizeof(void*) * cnrt_output_data_desc_array_length_);
    for(int i = 0; i < cnrt_input_data_desc_array_length_; ++i){
        CNRT_CHECK(
            cnrtMallocBatch(&mlu_input_data_array_t_[i], input_data_count_ * sizeof(float), batch_size_)
        );
    }
    for(int i = 0; i < cnrt_output_data_desc_array_length_; ++i){
        CNRT_CHECK(
            cnrtMallocBatch(&mlu_output_data_array_t_[i], output_data_count_ * sizeof(float), batch_size_)
        );     
    }
}

void Launch::copyDataHostToDevice(){
    for(int i = 0; i < cnrt_input_data_desc_array_length_; ++i){
        CNRT_CHECK(
            cnrtMemcpyBatch(mlu_input_data_array_t_[i],
                            cpu_input_data_array_t_[i],
                            input_data_count_ * sizeof(float),
                            batch_size_,
                            CNRT_MEM_TRANS_DIR_HOST2DEV)
        );
    }
}

void Launch::prepareInvokeFunctionParam(){
    bool muta = true;                   
    u32_t affinity = 0x01;
    cnrt_init_func_param_t_.muta = &muta;
    cnrt_init_func_param_t_.affinity = &affinity;
    cnrt_init_func_param_t_.data_parallelism = &batch_size_;
    cnrt_init_func_param_t_.end = CNRT_PARAM_END;
    cnrtInitFunctionMemory_V2(cnrt_function_t_, &cnrt_init_func_param_t_);

    cnrt_invoke_param_.data_parallelism = &batch_size_;
    cnrt_invoke_param_.affinity = &affinity;
    cnrt_invoke_param_.end = CNRT_PARAM_END;
}

void Launch::prepareParam(){
    cnrt_runtime_param_ = 
        (void**)malloc(sizeof(void*) * (cnrt_input_data_desc_array_length_ + cnrt_output_data_desc_array_length_));
    for(int i = 0; i < cnrt_input_data_desc_array_length_; ++i)
        cnrt_runtime_param_[i] = mlu_input_data_array_t_[i];
    for(int i = 0; i < cnrt_output_data_desc_array_length_; ++i)
        cnrt_runtime_param_[i] = mlu_output_data_array_t_[i];
}

void Launch::invokeFunction(){
    CNRT_CHECK(cnrtRuntimeContextCreateNotifier(cnrt_runtime_context_t_, &cnrt_notifier_start_t_));
    CNRT_CHECK(cnrtRuntimeContextCreateNotifier(cnrt_runtime_context_t_, &cnrt_notifier_end_t_));
    CNRT_CHECK(cnrtPlaceNotifier(cnrt_notifier_start_t_, cnrt_queue_t_));
    CNRT_CHECK(
        cnrtInvokeRuntimeContext(
            cnrt_runtime_context_t_,
            cnrt_runtime_param_,
            cnrt_queue_t_,
            (void*)&cnrt_invoke_param_)
    );
    CNRT_CHECK(cnrtPlaceNotifier(cnrt_notifier_end_t_, cnrt_queue_t_));
    CNRT_CHECK(cnrtSyncQueue(cnrt_queue_t_));
    CNRT_CHECK(cnrtNotifierDuration(cnrt_notifier_start_t_, cnrt_notifier_end_t_, &duration));

    CNRT_CHECK(cnrtDestroyNotifier(&cnrt_notifier_start_t_));
    CNRT_CHECK(cnrtDestroyNotifier(&cnrt_notifier_end_t_));
}

/**
 * 
*/
void Launch::copyDataDeviceToHost(){
    for(int i = 0; i < cnrt_output_data_desc_array_length_; ++i){
        CNRT_CHECK(
            cnrtMemcpyBatch(cpu_output_data_array_t_[i],
                            mlu_output_data_array_t_[i],
                            output_data_count_ * sizeof(float),
                            batch_size_,
                            CNRT_MEM_TRANS_DIR_DEV2HOST)
        );
    }
}

/**
 * 释放cnrt同步队列、cnrtFunction、cnrtContext资源
*/

void Launch::freeQueueAndContextAndFunction(){
    CNRT_CHECK(cnrtDestroyQueue(cnrt_queue_t_));
    CNRT_CHECK(cnrtDestroyRuntimeContext(cnrt_runtime_context_t_));
    CNRT_CHECK(cnrtDestroyFunction(cnrt_function_t_));
}

/**
 * @brief 释放cpu和mlu存储数据所申请的内存资源
*/
void Launch::freeCpuAndMlu(){
    for(int i = 0; i < cnrt_input_data_desc_array_length_; ++i){
        free(cpu_input_data_array_t_[i]);
        CNRT_CHECK(cnrtFree(mlu_input_data_array_t_[i]));
    }
    for(int i = 0; i < cnrt_output_data_desc_array_length_; ++i){
        free(cpu_output_data_array_t_[i]);
        CNRT_CHECK(cnrtFree(mlu_output_data_array_t_[i]));
    }
    free(cpu_input_data_array_t_);
    free(cpu_output_data_array_t_);
    free(mlu_input_data_array_t_);
    free(mlu_output_data_array_t_);
    free(cnrt_runtime_param_);
}

/**
 * @brief 卸载模型，销毁指针
 * @return void
*/
void Launch::unloadModel(){
        CNRT_CHECK(cnrtUnloadModel(cnrt_model_t_));
}

void Launch::execute(){
    loadModel(model_path_);
    createFunction();
    std::cout << "hello 1" << std::endl;
    createRuntimeContext();
    std::cout << "hello 2" << std::endl;
    getFunctionDataDesciption();
    std::cout << "hello 3" << std::endl;
    cpuMalloc();
    std::cout << "hello 4" << std::endl;
    mluMalloc();
    std::cout << "hello 5" << std::endl;
    copyDataDeviceToHost();
    prepareInvokeFunctionParam();
    prepareParam();
    invokeFunction();
    copyDataDeviceToHost();
    freeQueueAndContextAndFunction();
    freeCpuAndMlu();
    unloadModel();
}