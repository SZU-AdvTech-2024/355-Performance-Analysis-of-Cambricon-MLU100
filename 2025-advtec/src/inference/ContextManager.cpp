#include "ContextManager.hpp"
#include "../readimage/Shape.hpp"


// constructor
ContextManager::ContextManager(){}

ContextManager::ContextManager(
    ModelManager model_manager, std::string image_file_path, int batch_size):
    model_manager_(model_manager),
    image_file_path_(image_file_path),
    batch_size_(batch_size){

    std::cout << "constructor" << std::endl;
    readImageFromFile();
//    std::cout <<"constructor size: " << image_path_.size() << std::endl;
    prepareCPUInputAndOutput();
    prepareMLUInputAndOutput();
    copyInputCPU2MLU();
    initFunctionMemory();
    prepareRuntimeFunctionMemory();
    prepareInputAndOutputParam();
    runOnMlu();
    copyOutputMLU2CPU();
    
}

// deconstructor
ContextManager::~ContextManager(){
    std::cout << "ContextManager deconstructor" << std::endl;
    releaseResource();
}

float* ContextManager::preprocess(
        const std::vector<std::string>& image_vec,
        Shape input_shape,
        int input_data_count){
    ThreadPool pool(4);
    std::vector<std::future<std::vector<float>>> results;
    for(int i = 0; i < image_vec.size(); ++i){
//        std::cout << "i: " << i << std::endl;
        std::string image_path = image_vec[i];
//        std::cout << "image path: " << image_path << std::endl;
        results.emplace_back(pool.enqueue([image_path, input_shape, input_data_count]{
            std::this_thread::sleep_for(std::chrono::seconds(1));
//            std::cout << "input data count : " << input_data_count << std::endl;
            uint n = input_shape.n_;
            uint c = input_shape.c_;
            uint h = input_shape.h_;
            uint w = input_shape.w_;

            cv::Mat mat_image = cv::imread(image_path, cv::IMREAD_COLOR);
            if(mat_image.channels() == 3 && c == 3){
                cv::cvtColor(mat_image, mat_image, cv::COLOR_BGR2RGB);
            }
            if(mat_image.channels() == 3)
                mat_image.convertTo(mat_image, CV_32FC3);
            cv::resize(mat_image, mat_image, cv::Size(h, w));

            std::vector<float> res(input_data_count);
            for(int cc = 0; cc < c; ++cc){
                for(int hh = 0; hh < h; ++hh){
                    for(int ww = 0; ww < w; ++ww){
                        int index = cc * h * w + hh * w + ww;
                        res[index] = mat_image.at<cv::Vec3f>(hh, ww)[cc];
                    }
                }
            }
            return res;
        }));
    }
    std::cout << "results vector size " << results.size() << std::endl;
    // float* cpu_ptr;
    std::vector<float> cpu_ptr_vector;
    for(auto& future_res: results){
        auto processed = future_res.get();
        cpu_ptr_vector.insert(cpu_ptr_vector.end(), processed.begin(), processed.end());
    }
//    for(auto& e: cpu_ptr_vector){
//        std::cout << e;
//    }
//    std::cout << " cput ptr size: " <<  cpu_ptr_vector.size() << std::endl;
    return cpu_ptr_vector.data();
}

// read image
void ContextManager::readImageFromFile(){
//    image_reader_.image_task_path_vector_.resize(batch_size_);

//    std::cout << "reader size:" << image_path_.size() << std::endl;
    std::ifstream i_f_stream(image_file_path_.c_str());
    std::string line;
    int i = 0;
    while( i < batch_size_ && std::getline(i_f_stream, line)){
        image_path_.push_back(line);
        i++;
    }
    i_f_stream.close();
}

// allocate memory for cpu with threadpool
void ContextManager::prepareCPUInputAndOutput(){
    int input_num = model_manager_.input_num_;    
    int output_num = model_manager_.output_num_;
    cnrtDataDescArray_t input_desc_array_t = model_manager_.cnrt_input_data_desc_array_t_;
    cnrtDataDescArray_t output_desc_array_t = model_manager_.cnrt_output_data_desc_array_t_;
    
    cpu_input_ptr_array_ = (void**)malloc(sizeof(void*) * input_num);
    cpu_output_ptr_array_ = (void**)malloc(sizeof(void*) * output_num);

    //input
    for(int i = 0; i < input_num; ++i){
        int input_data_count;
        cnrtDataDesc_t cnrt_data_desc_t = input_desc_array_t[i];
        CNRT_CHECK(cnrtSetHostDataLayout(cnrt_data_desc_t, CNRT_FLOAT32, CNRT_NCHW));
        CNRT_CHECK(cnrtGetHostDataCount(cnrt_data_desc_t, &input_data_count));
        uint n, c, h, w;
        CNRT_CHECK(cnrtGetDataShape(cnrt_data_desc_t, &n, &c, &h, &w));
        Shape input_shape(n, c, h, w);
//        std::vector<float> cpu_ptr(input_data_count);

        cpu_input_ptr_array_[i] = (void*)malloc(sizeof(float) * input_data_count * n);
        cpu_input_ptr_array_[i] =
                (void*)preprocess(
                image_path_, input_shape, input_data_count);
    }
    // cpu_input_ptr_array_ = input_ptr;
    
    // output
    for(int i = 0; i < output_num; ++i){
        int output_data_count;
        cnrtDataDesc_t cnrt_data_desc_t = output_desc_array_t[i];
        CNRT_CHECK(cnrtSetHostDataLayout(cnrt_data_desc_t, CNRT_FLOAT32, CNRT_NCHW));
        CNRT_CHECK(cnrtGetHostDataCount(cnrt_data_desc_t, &output_data_count));
        uint n,c,h,w;
        CNRT_CHECK(cnrtGetDataShape(cnrt_data_desc_t, &n, &c, &h, &w));
        // if(LOG)
        //     std::cout<<"<"<<output_data_count<<","<<n<<","<<c<<","<<h<<","<<w<<">";
        float* cpu_ptr = (float*) malloc (sizeof(float) * output_data_count);
        
        cpu_output_ptr_array_[i] = (void*)cpu_ptr;
    }
    // cpu_output_ptr_array_ = output_ptr;
    std::cout << "cpu input and output" << std::endl;

}

void ContextManager::prepareMLUInputAndOutput(){
    mlu_input_ptr_array_ = (void**)malloc(sizeof(void*) * model_manager_.input_num_);
    mlu_output_ptr_array_ = (void**)malloc(sizeof(void*) * model_manager_.output_num_);
    for(int i = 0; i < model_manager_.input_num_; ++i)
        CNRT_CHECK(cnrtMallocByDesc(mlu_input_ptr_array_, model_manager_.cnrt_input_data_desc_array_t_[i]));
    for(int i = 0; i < model_manager_.output_num_; ++i)
        CNRT_CHECK(cnrtMallocByDesc(mlu_output_ptr_array_, model_manager_.cnrt_output_data_desc_array_t_[i]));
    std::cout << "mlu input and output" << std::endl;
}

void ContextManager::copyInputCPU2MLU(){
    std::cout << "transform start" << std::endl;
    CNRT_CHECK(cnrtMemcpyByDescArray(
        mlu_input_ptr_array_,
        cpu_input_ptr_array_,
        model_manager_.cnrt_input_data_desc_array_t_,
        model_manager_.input_num_,
        CNRT_MEM_TRANS_DIR_HOST2DEV));
    std::cout << "transform end" << std::endl;
}

void ContextManager::initFunctionMemory(){
    bool muta = false;
    u32_t affinity = 0x01;
    int dp = 1;
    cnrt_init_func_param_t_.muta = &muta;
    cnrt_init_func_param_t_.data_parallelism = &dp;
    cnrt_init_func_param_t_.affinity = &affinity;
    cnrt_init_func_param_t_.end = CNRT_PARAM_END;
    cnrtInitFunctionMemory_V2(model_manager_.cnrt_function_t_, &cnrt_init_func_param_t_);
}

void ContextManager::prepareInputAndOutputParam(){
//    cnrtInitFunctionMemory(model_manager_.cnrt_function_t_, CNRT_FUNC_TYPE_BLOCK0);
    param_ = (void**)malloc(sizeof(void*) * (model_manager_.input_num_ + model_manager_.output_num_));
    for(int i = 0; i < model_manager_.input_num_; ++i)
        param_[i] = mlu_input_ptr_array_[i];
    for(int i = 0; i < model_manager_.output_num_; ++i)
        param_[i+model_manager_.input_num_] = mlu_output_ptr_array_[i];
}

void ContextManager::prepareRuntimeFunctionMemory(){
    u32_t affinity = 0x01;
    int dp = 1;
    cnrt_invoke_func_param_t_.affinity = &affinity;
    cnrt_invoke_func_param_t_.data_parallelism = &dp;
    cnrt_invoke_func_param_t_.end = CNRT_PARAM_END;
}

void ContextManager::runOnMlu(){
//    cnrtRuntimeContext_t cnrt_runtime_context_t_;
    CNRT_CHECK(cnrtCreateRuntimeContext(&cnrt_runtime_context_t_, model_manager_.cnrt_function_t_, nullptr));
    CNRT_CHECK(cnrtSetRuntimeContextDeviceId(cnrt_runtime_context_t_, model_manager_.device_id_));
    CNRT_CHECK(cnrtSetRuntimeContextChannel(cnrt_runtime_context_t_, CNRT_CHANNEL_TYPE_NONE));      // key trick
    CNRT_CHECK(cnrtInitRuntimeContext(cnrt_runtime_context_t_, nullptr));
    CNRT_CHECK(cnrtRuntimeContextCreateQueue(cnrt_runtime_context_t_, &cnrt_queue_t_));

    cnrtNotifier_t start, end;
    CNRT_CHECK(cnrtRuntimeContextCreateNotifier(cnrt_runtime_context_t_, &start));
    CNRT_CHECK(cnrtRuntimeContextCreateNotifier(cnrt_runtime_context_t_, &end));
    CNRT_CHECK(cnrtPlaceNotifier(start, cnrt_queue_t_));
    CNRT_CHECK(cnrtInvokeRuntimeContext(cnrt_runtime_context_t_, param_, cnrt_queue_t_, (void*)&cnrt_invoke_func_param_t_));
    CNRT_CHECK(cnrtPlaceNotifier(end, cnrt_queue_t_));
    CNRT_CHECK(cnrtSyncQueue(cnrt_queue_t_));
    CNRT_CHECK(cnrtNotifierDuration(start, end, &infer_duration_));

    CNRT_CHECK(cnrtDestroyNotifier(&start));
    CNRT_CHECK(cnrtDestroyNotifier(&end));

}

void ContextManager::copyOutputMLU2CPU(){
    CNRT_CHECK(cnrtMemcpyByDescArray(
        cpu_output_ptr_array_,
        mlu_output_ptr_array_,
        model_manager_.cnrt_output_data_desc_array_t_,
        model_manager_.output_num_,
        CNRT_MEM_TRANS_DIR_DEV2HOST
    ));
}

void ContextManager::releaseResource(){
    std::cout << "release resource" << std::endl;
    CNRT_CHECK(cnrtDestroyQueue(this->cnrt_queue_t_));
    CNRT_CHECK(cnrtDestroyRuntimeContext(this->cnrt_runtime_context_t_));
    // cnrtDestroyFunction(cnrt_function_t_);
    for(int i = 0; i < model_manager_.input_num_; ++i){
        free(cpu_input_ptr_array_[i]);
        CNRT_CHECK(cnrtFree(mlu_input_ptr_array_[i]));
    }
    for(int i = 0; i < model_manager_.output_num_; ++i){
        free(cpu_output_ptr_array_[i]);
        CNRT_CHECK(cnrtFree(mlu_output_ptr_array_[i]));
    }
    free(cpu_input_ptr_array_);
    free(cpu_output_ptr_array_);
    free(mlu_input_ptr_array_);
    free(mlu_output_ptr_array_);
    free(param_);
}