#ifndef CONTEXTMANAGER_HPP
#define CONTEXTMANAGER_HPP

#include <iostream>
#include "cnrt.h"
#include "ModelManager.hpp"
//#include "../readimage/ImageReader.hpp"
#include <memory>
#include <vector>
#include "../lib/ThreadPool/ThreadPool.h"
#include "opencv2/opencv.hpp"
#include "Shape.hpp"
#include <fstream>
class ContextManager{
public:
    // default constructor
    ContextManager();

    // constructor 
    ContextManager(ModelManager model_manager, std::string image_file_path, int batch_size);

    // deconstroctor release resource
    ~ContextManager();

    // read image
    void readImageFromFile();

    // prepare input for cpu
    void prepareCPUInputAndOutput();

    // prepare input for mlu , to be done
    void prepareMLUInputAndOutput();

    // copy data from cpu to mlu
    void copyInputCPU2MLU();

    void initFunctionMemory();

    // Initial Runtime Funtcion memory
    void prepareRuntimeFunctionMemory();

    // prepare input and output param
    void prepareInputAndOutputParam();


    // prepare invoke function
//    void prepareRuntimeFunctionMemory();

    // run on mlu with runtime context
    void runOnMlu();

    // copy data from mlu to cpu
    void copyOutputMLU2CPU();

    // release resource
    void releaseResource();

    //
    float* preprocess( const std::vector<std::string>& image_vec,
                       Shape input_shape,
                       int input_data_count);
private:
    std::string image_file_path_;        // input image file path
    ModelManager model_manager_;        // ModelManager member
    int batch_size_;                    // batch size
    cnrtQueue_t cnrt_queue_t_;          // synchronisation queue
    void** cpu_input_ptr_array_;        // allocate input memory for cpu
    void** cpu_output_ptr_array_;       // allocate output memory for cpu
    std::vector<std::string> image_path_;           // read image path string from config file
    void** mlu_input_ptr_array_;        // allocate input memory for mlu
    void** mlu_output_ptr_array_;       // allocate output memory for mlu
    cnrtInitFuncParam_t cnrt_init_func_param_t_;    // init function param
    cnrtInvokeFuncParam_t cnrt_invoke_func_param_t_; // invoke function param
    void** param_;                       // prepare input and output param
    cnrtRuntimeContext_t cnrt_runtime_context_t_;   // create runtime context pointer
    float infer_duration_;               // inference duration
};
#endif