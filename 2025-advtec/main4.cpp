//
// Created by szu on 11/18/24.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include "cnrt.h"
#include <vector>
#include <fstream>
#include "sys/time.h"

#define LOG 0
#define LOG_TIME 1

float getTime(struct timeval start_t, struct timeval end_t){
    return (end_t.tv_sec - start_t.tv_sec) * 1000000 + end_t.tv_usec - start_t.tv_usec;
}

void preprocess(
        std::vector<std::string> image_vec,
        float* cpu_ptr,
        uint n,
        uint c,
        uint h,
        uint w,
        int data_count){
    int chw = c * h * w;
    int offset = 0;

    for(const auto& image: image_vec){
        if(LOG)
            std::cout << "<path," << image << ">";

        // 读取图片，转换成cv::Mat格式
        cv::Mat mat_image = cv::imread(image, cv::IMREAD_COLOR);
        cv::Mat sampled;
        cv::Mat resized;

        if(LOG)
            std::cout << mat_image.channels() << " " << c << std::endl;

        // 将颜色模式从bgr转成rgb
        if(mat_image.channels() == 3 && c == 3){
            if(LOG)
                std::cout << CV_8UC3 <<"--mat_image type: "<< mat_image.type() << std::endl;
            cv::cvtColor(mat_image, mat_image, cv::COLOR_BGR2RGB);
            if(LOG)
                std::cout << "bgr2rgb" << std::endl;
        }else{
            std::cout << "do nothing"<< std::endl;
            // exit(-1);
        }

        // 将图像归一化，预训练好的dnn模型通常要求输入图像满足一定归一化标准；
        if(c == 3)
            mat_image.convertTo(mat_image, CV_32FC3);

        // 将图像resize成模型的h和w；
        cv::resize(mat_image, mat_image, cv::Size(h, w));

        // opencv通常将图像存储为HWC格式，而深度学习处理器是以CHW形式读取图像的；
        // 因此需要将图像从HWC转换为CHW

        // cnrtReshapeNHWCToNCHW();

        for(int cc = 0; cc < c; ++cc){
            for(int hh = 0; hh < h; ++hh){
                for(int ww = 0; ww < w; ++ww){
                    int index = offset + cc * h * w + hh * w + ww;
                    cpu_ptr[index] = mat_image.at<cv::Vec3f>(hh, ww)[cc];
                }
            }
        }
        offset += chw;
        if(LOG)
            std::cout << "<offset: " << offset <<">";
    }
}

void postprocess(
        void** cpu_output_ptr_array,
        int output_num,
        cnrtDataDescArray_t cnrt_output_data_desc_array_t,
        std::vector<std::string>& image_vec){
    for(int i = 0; i < output_num; ++i){
        int output_datacount;
        cnrtDataDesc_t cnrt_data_desc_t = cnrt_output_data_desc_array_t[i];
        cnrtGetHostDataCount(cnrt_data_desc_t, &output_datacount);
        uint n,c,h,w;
        cnrtGetDataShape(cnrt_data_desc_t, &n, &c, &h, &w);
        if(LOG)
            std::cout<<"<"<<output_datacount<<","<<n<<","<<c<<","<<h<<","<<w<<">";
        float* cpu_ptr = (float*)cpu_output_ptr_array[i];

        // std::string file = "./src/config/writetofile.txt";
        // std::ofstream o_f_stream(file);
        // if(!o_f_stream){
        //     std::cout << "invalid file path" << std::endl;
        //     exit(-1);
        // }

        // for(int j = 0; j < output_datacount; ++j){
        //     o_f_stream << cpu_ptr[j] << std::endl;
        // }
        // o_f_stream.close();

        for(int k = 0; k < n; ++k){
            float top = -1;
            int top_index = -1;
            for(int j = 0; j < c; ++j){
                int index = j + k * c;
                if(top < cpu_ptr[index]){
                    top = cpu_ptr[index];
                    top_index = j;
                }
            }
            if(LOG)
                std::cout << "<" <<k+1<<","<<top_index<<">";
        }

    }
}

void prepareCPUInputAndOutput(
        cnrtDataDescArray_t cnrt_input_data_desc_array_t,
        int input_num,
        void*** cpu_input,
        std::vector<std::string> image_vec,
        cnrtDataDescArray_t cnrt_output_data_desc_array_t,
        int output_num,
        void*** cpu_output
){
    void** input_ptr = (void**)malloc(sizeof(void*) * input_num);
    void** output_ptr = (void**)malloc(sizeof(void*) * output_num);

    // 将输入数据转换为mlu能接收的格式
    for(int i = 0; i < input_num; ++i){
        int input_data_count;
        cnrtDataDesc_t cnrt_data_desc_t = cnrt_input_data_desc_array_t[i];
        cnrtSetHostDataLayout(cnrt_data_desc_t, CNRT_FLOAT32, CNRT_NCHW);
        cnrtGetHostDataCount(cnrt_data_desc_t, &input_data_count);
        uint n,c,h,w;
        cnrtGetDataShape(cnrt_data_desc_t, &n, &c, &h, &w);
        if(LOG)
            std::cout<<"<"<<input_data_count<<","<<n<<","<<c<<","<<h<<","<<w<<">";
        float* cpu_ptr = (float*)malloc(sizeof(float) * input_data_count);
        preprocess(image_vec, cpu_ptr, n, c, h, w, input_data_count);
        if(LOG){
            for(int i = 0; i < input_data_count; ++i)
                std::cout << cpu_ptr[i] << std::endl;
        }

        // float* cpu_ptr = (float*)malloc(sizeof(float) * input_data_count);
        // for(int j = 0; j < input_data_count; ++j)
        //     cpu_ptr[j] = 1;
        input_ptr[i] = (void*)cpu_ptr;
    }

    // 为输出创造空间
    for(int i = 0; i < output_num; ++i){
        int output_data_count;
        cnrtDataDesc_t cnrt_data_desc_t = cnrt_output_data_desc_array_t[i];
        cnrtSetHostDataLayout(cnrt_data_desc_t, CNRT_FLOAT32, CNRT_NCHW);
        cnrtGetHostDataCount(cnrt_data_desc_t, &output_data_count);
        uint n,c,h,w;
        cnrtGetDataShape(cnrt_data_desc_t, &n, &c, &h, &w);
        if(LOG)
            std::cout<<"<"<<output_data_count<<","<<n<<","<<c<<","<<h<<","<<w<<">";
        float* cpu_ptr = (float*) malloc (sizeof(float) * output_data_count);

        output_ptr[i] = (void*)cpu_ptr;
    }
    *cpu_input = input_ptr;
    *cpu_output = output_ptr;
}



cnrtRet_t run(
        std::string model_path,
        std::vector<std::string> image_vec,
        std::string network,
        int data_parallelism,
        int device_id,
        std::vector<std::vector<float>> &time_mat,
        int batch_index){
    // 1. 初始化cnrt
    // cnrtInit(0);

    // 2. 装载模型并提取函数
    cnrtModel_t cnrt_model_t;
    cnrtLoadModel(&cnrt_model_t, model_path.c_str());
    cnrtFunction_t cnrt_function_t;
    cnrtCreateFunction(&cnrt_function_t);
    cnrtExtractFunction(&cnrt_function_t, cnrt_model_t, network.c_str());
    int model_parallelism;
    cnrtQueryModelParallelism(cnrt_model_t, &model_parallelism);
    if(LOG)
        std::cout << "model parallelism: " << model_parallelism << std::endl;

    // 3. 获得函数 IO 的数据描述
    int input_num, output_num;
    cnrtDataDescArray_t cnrt_input_data_desc_array_t, cnrt_output_data_desc_array_t;
    cnrtGetInputDataDesc(&cnrt_input_data_desc_array_t, &input_num, cnrt_function_t);
    cnrtGetOutputDataDesc(&cnrt_output_data_desc_array_t, &output_num, cnrt_function_t);
    if(LOG)
        std::cout << "<inputnum,"<<input_num<<">"<<"<outputnum,"<<output_num<<">";

    // 4. 设置运行时device
    cnrtDev_t cnrt_dev_t;
    cnrtGetDeviceHandle(&cnrt_dev_t, device_id);
    cnrtSetCurrentDevice(cnrt_dev_t);

    // 5. 初始化运行时同步队列
    cnrtQueue_t cnrt_queue_t;
    // cnrtCreateQueue(&cnrt_queue_t);          //可以看看和cnrtRuntimeContextCreateQueue的区别


    // 6. 为cpu端分配内存
    void** cpu_input_ptr_array;
    void** cpu_output_ptr_array;

    struct timeval start_t, end_t;
    gettimeofday(&start_t, nullptr);
    prepareCPUInputAndOutput(
            cnrt_input_data_desc_array_t, input_num, &cpu_input_ptr_array,
            image_vec,
            cnrt_output_data_desc_array_t, output_num, &cpu_output_ptr_array
    );
    gettimeofday(&end_t, nullptr);


    if(LOG_TIME){
        time_mat[batch_index][0] += getTime(start_t, end_t);
        std::cout << "<preprocess time: " << getTime(start_t, end_t) << "us>";
    }


    if(LOG){
        std::cout << "cpu io success" << std::endl;
    }

    // 7. 为mlu端分配内存
    void** mlu_input_ptr_array = (void**)malloc(sizeof(void*) * input_num);
    void** mlu_output_ptr_array = (void**)malloc(sizeof(void*) * output_num);
    if(data_parallelism * model_parallelism > 32){
        std::cout << "error: dp * mp > 32." << std::endl;
        exit(0);
    }
    // 7.1 为输入端预留空间
    // cnrtMallocBatchByDescArray(
    //     &mlu_input_ptr_array,
    //     cnrt_input_data_desc_array_t,
    //     input_num,
    //     data_parallelism
    // );
    for(int i = 0; i < input_num; ++i){
        cnrtMallocByDesc(mlu_input_ptr_array, cnrt_input_data_desc_array_t[i]);
    }


    // 7.2 为输出端预留空间
    // cnrtMallocBatchByDescArray(
    //     &mlu_output_ptr_array,
    //     cnrt_output_data_desc_array_t,
    //     output_num,
    //     data_parallelism
    // );
    for(int i = 0; i < output_num; ++i)
        cnrtMallocByDesc(mlu_output_ptr_array, cnrt_output_data_desc_array_t[i]);


    if(LOG){
        std::cout << "mlu io success" << std::endl;
    }

    // 7.3 将输入数据从cpu端复制到mlu端
    // cnrtMemcpyBatchByDescArray(
    //     mlu_input_ptr_array, cpu_input_ptr_array, cnrt_input_data_desc_array_t,
    //     input_num, data_parallelism, CNRT_MEM_TRANS_DIR_HOST2DEV
    // );

    gettimeofday(&start_t, NULL);
    cnrtMemcpyByDescArray(
            mlu_input_ptr_array,
            cpu_input_ptr_array,
            cnrt_input_data_desc_array_t,
            input_num,
            CNRT_MEM_TRANS_DIR_HOST2DEV
    );
    gettimeofday(&end_t, NULL);
    if(LOG_TIME){
        time_mat[batch_index][1] += getTime(start_t, end_t);
        std::cout <<"<h2d, "<< getTime(start_t, end_t)<<"us>";
    }


    if(LOG){
        std::cout << "cpu2mlu success" << std::endl;
    }

    // 8. 初始化函数内存空间，并设置并行度
    cnrtInitFuncParam_t cnrt_init_func_param_t;
    bool muta = false;
    u32_t affinity = 0x01;
    cnrt_init_func_param_t.muta = &muta;
    cnrt_init_func_param_t.data_parallelism = &data_parallelism;
    cnrt_init_func_param_t.affinity = &affinity;
    cnrt_init_func_param_t.end = CNRT_PARAM_END;
    cnrtInitFunctionMemory_V2(cnrt_function_t, &cnrt_init_func_param_t);

    if(LOG){
        std::cout << "init function memory success" << std::endl;
    }


    // 9. 准备输入输出参数
    void** param = (void**)malloc(sizeof(void*) * (input_num + output_num));
    for(int i = 0; i < input_num; ++i)
        param[i] = mlu_input_ptr_array[i];
    for(int i = 0; i < output_num; ++i)
        param[input_num + i] = mlu_output_ptr_array[i];

    // 10. 准备调用运行时参数
    cnrtInvokeFuncParam_t cnrt_invoke_func_param_t;
    cnrt_invoke_func_param_t.affinity = &affinity;
    cnrt_invoke_func_param_t.data_parallelism = &data_parallelism;
    cnrt_invoke_func_param_t.end = CNRT_PARAM_END;

    // 11. 创建运行时上下文指针
    cnrtRuntimeContext_t cnrt_runtime_context_t;
    cnrtCreateRuntimeContext(&cnrt_runtime_context_t, cnrt_function_t, nullptr);
    cnrtSetRuntimeContextDeviceId(cnrt_runtime_context_t, device_id);
    cnrtSetRuntimeContextChannel(cnrt_runtime_context_t, CNRT_CHANNEL_TYPE_NONE); //关键trick
    cnrtInitRuntimeContext(cnrt_runtime_context_t, nullptr);
    cnrtRuntimeContextCreateQueue(cnrt_runtime_context_t, &cnrt_queue_t);

    // 12. run on mlu
    cnrtNotifier_t start, end;
    cnrtRuntimeContextCreateNotifier(cnrt_runtime_context_t, &start);
    cnrtRuntimeContextCreateNotifier(cnrt_runtime_context_t, &end);
    cnrtPlaceNotifier(start, cnrt_queue_t);
    cnrtInvokeRuntimeContext(cnrt_runtime_context_t, param, cnrt_queue_t, (void*)&cnrt_invoke_func_param_t);
    cnrtPlaceNotifier(end, cnrt_queue_t);
    cnrtSyncQueue(cnrt_queue_t);
    float us;
    cnrtNotifierDuration(start, end, &us);
    if(LOG_TIME){
        time_mat[batch_index][2] += us;
        std::cout << "<infer duration, " << us <<"us>";
    }

    // std::cout << model_path << ", duration: " << us << "us" << std::endl;

    cnrtDestroyNotifier(&start);
    cnrtDestroyNotifier(&end);

    // 13. 读取结果
    // cnrtMemcpyBatchByDescArray(
    //     cpu_output_ptr_array,
    //     mlu_output_ptr_array,
    //     cnrt_output_data_desc_array_t,
    //     output_num,
    //     data_parallelism,
    //     CNRT_MEM_TRANS_DIR_DEV2HOST
    // );
    gettimeofday(&start_t, NULL);
    cnrtMemcpyByDescArray(
            cpu_output_ptr_array,
            mlu_output_ptr_array,
            cnrt_output_data_desc_array_t,
            output_num,
            CNRT_MEM_TRANS_DIR_DEV2HOST
    );
    gettimeofday(&end_t, NULL);
    if(LOG_TIME){
        time_mat[batch_index][3] += getTime(start_t, end_t);
        std::cout <<"<d2h, "<< getTime(start_t, end_t)<<"us>";
    }


    gettimeofday(&start_t, NULL);
    postprocess(cpu_output_ptr_array, output_num, cnrt_output_data_desc_array_t, image_vec);
    gettimeofday(&end_t, NULL);
    if(LOG_TIME){
        time_mat[batch_index][4] += getTime(start_t, end_t);
        std::cout << "<postprocess time: "<<getTime(start_t, end_t)<<"us>";
    }


    // 14. 释放资源
    cnrtDestroyQueue(cnrt_queue_t);
    cnrtDestroyRuntimeContext(cnrt_runtime_context_t);
    cnrtDestroyFunction(cnrt_function_t);

    // cnrtFreeArray(mlu_input_ptr_array, input_num);
    // cnrtFreeArray(mlu_input_ptr_array, output_num);

    for(int i = 0; i < input_num; ++i){
        free(cpu_input_ptr_array[i]);
        cnrtFree(mlu_input_ptr_array[i]);
    }
    for(int i = 0; i < output_num; ++i){
        free(cpu_output_ptr_array[i]);
        cnrtFree(mlu_output_ptr_array[i]);
    }
    free(cpu_input_ptr_array);
    free(cpu_output_ptr_array);
    free(mlu_input_ptr_array);
    free(mlu_output_ptr_array);
    free(param);
    cnrtUnloadModel(cnrt_model_t);
    // cnrtDestroy();
}

void writeTimeToFile(
        const std::string& model_name,
        const std::vector<std::string>& batch_vector,
        std::vector<std::vector<float>>& time_mat,
        int TIMES){

    std::string file_path =
            "./result/" +
            model_name +
            "_" + std::to_string(TIMES)
            + "_res.txt";
    // 将结果写入文件
    std::ofstream o_f_stream_time(file_path);
    if(!o_f_stream_time){
        std::cout << "invalid file path" << std::endl;
    }else{
        o_f_stream_time << "---model breakdown(ms)---"<<std::endl;
        o_f_stream_time << "batch" << '\t' << '\t' ;
        o_f_stream_time << "preprocess" << '\t' << '\t';
        o_f_stream_time << "h2d" << '\t' << '\t';
        o_f_stream_time << "infer" << '\t' << '\t';
        o_f_stream_time << "d2h" << '\t' << '\t';
        o_f_stream_time << "postprocess" << '\t' << '\t';
        o_f_stream_time << std::endl;
        for(int i = 0; i < time_mat.size(); ++i){
            o_f_stream_time << batch_vector[i] << '\t' << '\t';
            for(int j = 0; j < time_mat[i].size(); ++j){
                float temp = time_mat[i][j] / 1000 / TIMES;
                o_f_stream_time  << temp << '\t' << '\t';
            }
            o_f_stream_time << std::endl;
        }
        std::cout <<"write success!"<<std::endl;
    }
}

int main(){
    // cv::Mat singleChannelMat(100, 100, CV_32FC1, cv::Scalar(1.5));
    // std::cout << "single channel matrix: " << singleChannelMat.at<float>(0, 0) << std::endl;
    // cv::Mat threeChannelMat(100, 100, CV_32FC3, cv::Scalar(1.0, 2.0, 3.0));
    // std::cout << "Three channel matrix: " << threeChannelMat.at<cv::Vec3f>(0, 0) << std::endl;

    // std::string model_name = "vgg16";
    cnrtInit(0);
    std::vector<std::string> model_vector = {"resnet18_v1","resnet34_v1","resnet50_v2",
                                             "resnet101_v2","resnet152_v2","vgg16","vgg19","alexnet","squeezenet","inception_v1",
                                             "inception_v3","mobileNet"};
    // std::vector<std::string> model_vector = {"resnet152_v2"};

    std::string core = "32";
    // std::vector<std::string> batch_vector = {"1", "2", "4", "8", "16", "32"};
    // std::vector<std::string> batch_vector = {"64"};
    std::vector<std::string> batch_vector = {"4"};


    int data_parallelism = 1;
    std::string network = "subnet0";
    int device_id = 1;

    std::vector<std::vector<float>> time_mat(batch_vector.size(), std::vector<float>(5));

    // std::vector<float> preprocess_vec(batch_vector.size());
    // std::vector<float> h2d_vec(batch_vector.size());
    // std::vector<float> infer_vec(batch_vector.size());
    // std::vector<float> d2h_vec(batch_vector.size());
    // std::vector<float> postprocess_vec(batch_vector.size());

    // 运行k次，取平均值
    int k;
    const int TIMES = 10;
    for(auto&e: model_vector){

        for(k = 0; k < TIMES; ++k){

            int batch_index = 0;
            for(auto &batch: batch_vector){
                int image_vec_size = std::atoi(batch.c_str());
                std::string image_file_path = "./src/config/inputimage.txt";
                std::vector<std::string> image_vec;
                std::ifstream i_f_stream(image_file_path);
                // if(!i_f_stream){
                //     std::cout << "not exist"<< std::endl;
                //     exit(-1);
                // }
                std::string line;
                int i = 0;
                while(i < image_vec_size && std::getline(i_f_stream, line)){
                    image_vec.push_back(line);
                    i++;
                }
                i_f_stream.close();
                std::cout << image_vec.size() << std::endl;
                std::string model_path =
                        "/home/szu/cambricon/tensorflow/models/offline/"+
                        e +
                        "/"+
                        e +
                        "_MLU100_"+
                        core +
                        "core_"+
                        batch +
                        "batch.cambricon";
                std::cout << e << ": ";
                run(model_path, image_vec, network, data_parallelism, device_id, time_mat, batch_index);
                std::cout << std::endl;
                // for(auto &e: model_vector){

                // }
                batch_index++;
            }
        }
        writeTimeToFile(e, batch_vector, time_mat, TIMES);
    }
    cnrtDestroy();
    return 0;
}