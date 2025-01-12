/*Copyright 2018 Cambricon*/
#include <pthread.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <typeinfo>
#include <sstream>
#include <opencv2/core/core.hpp>  // NOLINT
#include <opencv2/highgui/highgui.hpp>  // NOLINT
#include <opencv2/imgproc/imgproc.hpp>  // NOLINT
#include "cnrt.h"  // NOLINT
using namespace cv;  // NOLINT
using namespace std;  // NOLINT

bool file_clear = 0;
int img_processed = 0;
int top1_count = 0;
int top5_count = 0;
int top1_hit = 0;
int top5_hit = 0;
int total_files = 0;
int current_file_id = 0;
double input_std = 1;
vector<string> filename;
vector<string> filename_before;
vector<string> filename_use;
void* dataBuffer;
float* output_cpu;
vector<float> means;

unsigned in_n = 1, in_c = 3, in_h = 224, in_w = 224;
unsigned out_n = 1, out_c = 1000, out_h = 1, out_w = 1;
int count_batch, count_use;
string file_name, names; //NOLINT

#define PARAM_CHECK(rst, err_info) if (rst == 0) {std::cout << \
err_info << std::endl; exit(0);}

string get_string(string names, int index) {
  ifstream files(names.c_str(), std::ios::in);
  string file;
  for (int i = 0; i < index; i++) {
    getline(files, file);
  }
  getline(files, file);
  files.close();
  return file;
}

void* softmax_print(int offset, int expected_label) {
    int outputSize = out_n * out_c * out_h * out_w;
    float* scale_data = (float*)malloc(outputSize * sizeof(float));
    for (int n = 0; n < count_use; n++) {
      double max = output_cpu[n * out_c * out_w * out_h];
      double sum = 0;
      for (int i = 0; i < out_c * out_h * out_w; i++) {
        if (output_cpu[i + n * out_c * out_h * out_w] > max)
          max = output_cpu[i + n * out_c * out_h * out_w];
      }

      for (int i = 0; i < out_c * out_h * out_w; i++) {
        scale_data[i + n * out_c * out_w * out_h] =
            exp(output_cpu[i + n * out_c * out_w * out_h] - max);
        sum += scale_data[i + n * out_c * out_w * out_h];
      }

      int max_index[5] = {0};
      double max_num[5] = {0};

      for (int i = 0; i < static_cast<int>(out_c * out_h * out_w); i++) {
        output_cpu[i + n * out_c * out_w * out_h] =
            scale_data[i + n * out_c * out_w * out_h] / sum;
        double tmp = output_cpu[i + n * out_c * out_w * out_h];
        int tmp_index = i;
        for (int j = 0; j < 5; j++) {
          if (tmp > max_num[j]) {
             tmp_index += max_index[j];
             max_index[j] = tmp_index - max_index[j];
             tmp_index -= max_index[j];
             tmp += max_num[j];
             max_num[j] = tmp-max_num[j];
             tmp -= max_num[j];
          }
        }
      }
    free(scale_data);
    std::cout << "expected label=" << expected_label << std::endl;
    int top1_b = (max_index[0]-offset) == expected_label ? 1 : 0;
    top1_count += top1_b;
    int top5_b = 0;
    for (int i=0; i< 5; i++) {
      std::cout << "max_index" << i << " is " << max_index[i] << std::endl;
      if ((max_index[i]-offset) == expected_label)
        top5_b = 1;
    }
    top5_count += top5_b;

    cout << "------------------------detection for " << filename_use[n] <<
        "---------------------------" << endl;
      for (int i = 0; i < 5; i++) {
        cout << fixed << setprecision(4) << max_num[i] << "  -  "
            << get_string(names, max_index[i] - offset) << endl;
      }
      std::cout << "TOP1_HIT = " << top1_b << std::endl;
      std::cout << "TOP5_HIT = " << top5_b << std::endl;
    }
    return NULL;
}

template <typename T>
void get_img(string color_mode, int* expected_label) {
    string file;
    filename.clear();
    int tmp = img_processed;
    ifstream files(file_name.c_str(), std::ios::in);
    while (tmp > 0 && getline(files, file)) {
       --tmp;
    }
    if (tmp > 0 || (!getline(files, file))) {
        count_batch = 0;
        file_clear = 1;
        count_use = count_batch;
        filename_use = filename;
        return;
    }
    vector<vector<cv::Mat> > input_imgs;
    cv::Mat mean_;
    cv::Scalar channel_mean(means[0], means[1], means[2]);
    auto cv_fc3 = CV_8UC3, cv_fc1 = CV_8UC1;
    if (typeid(T) == typeid(float)) {
      cv_fc3 = CV_32FC3;
      cv_fc1 = CV_32FC1;
    }
    if (in_c == 3)
      mean_ = cv::Mat(cv::Size(in_h, in_w), cv_fc3, channel_mean);
    else
       mean_ = cv::Mat(cv::Size(in_h, in_w), cv_fc1, channel_mean);
    filename.push_back(file);
    Mat img_resized;

    // int pos = 0;
    string image_name;
    string image_label;
    for (int i = 0; i < static_cast<int>(file.size()); i++) {
      if (file[i] == ' ') {
        image_name = string(file.begin(), file.begin()+i);
        image_label = string(file.begin()+i, file.end());
        *expected_label = atoi(image_label.c_str());
      }
    }

    cv::Mat img = imread(image_name, IMREAD_COLOR);
    if (img.empty()) {
        cout << "error ! no such file : " << image_name << endl;
        count_batch = 0;
        file_clear = 1;
        return;
    }
    cv::Mat sample;
    if (img.channels() == 3 && in_c == 1)
      cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && in_c == 1)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && in_c == 3)
      cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && in_c == 3)
      cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else if (color_mode == "rgb")
      cv::cvtColor(img, sample, cv::COLOR_BGR2RGB);
    else
      sample = img;

    T *input_data = (T*)dataBuffer;  // NOLINT
    for (int i = 0; i < static_cast<int>(in_n); i++) {
        input_imgs.push_back(std::vector<cv::Mat> ());
        for (int j = 0; j < static_cast<int>(in_c); j++) {
           cv::Mat channel(in_h, in_w, cv_fc1, input_data);
           input_imgs[i].push_back(channel);
           input_data += in_h * in_w;
        }
    }

    count_batch = 1;
    if (img.rows != static_cast<int>(in_h) || img.cols != static_cast<int>(in_w)
        || img.channels() != static_cast<int>(in_c)) {
      cout << "warining! image " << file << " size is wrong!" << endl;
      cout << "input size should be :" << in_c << " " << in_h << " "
          << in_w << endl;
      cout << "now input size is :" << img.channels() << " " << img.rows
          <<" " << img.cols << endl;
      cout << "img is going to resize!" << endl;
    }

    if (img.rows != static_cast<int>(in_h)
        || img.cols != static_cast<int>(in_w))
      resize(sample, img_resized, cv::Size(in_h, in_w));
    else
      img_resized = sample;

    cv::Mat img_float;
    if (in_c == 3)
       img_resized.convertTo(img_float, cv_fc3);
    else
       img_resized.convertTo(img_float, cv_fc1);
    cv::Mat img_normalized;
    cv::subtract(img_float, mean_, img_normalized);
    // cv::split(img_normalized, (input_imgs)[count_batch-1]);

    std::cout << "input_std is: " << input_std << std::endl;
    cv::Scalar channel_std(input_std, input_std, input_std);
    cv::Mat img_std(cv::Size(in_h, in_w), cv_fc3, channel_std);
    cv::Mat img_div;
    cv::divide(img_normalized, img_std, img_div);
    cv::split(img_div, (input_imgs)[count_batch-1]);

    while (count_batch != static_cast<int>(in_n) && getline(files, file)) {
     cout << "count_batch = " << count_batch <<" in" << __FUNCTION__<< endl;
      Mat img_resized1;
      cv::Mat img1 = imread(file, IMREAD_COLOR);
      cv::Mat sample1;
      if (img.channels() == 3 && in_c == 1)
        cv::cvtColor(img1, sample1, cv::COLOR_BGR2GRAY);
      else if (img1.channels() == 4 && in_c == 1)
        cv::cvtColor(img1, sample1, cv::COLOR_BGRA2GRAY);
      else if (img1.channels() == 4 && in_c == 3)
        cv::cvtColor(img1, sample1, cv::COLOR_BGRA2BGR);
      else if (img1.channels() == 1 && in_c == 3)
        cv::cvtColor(img1, sample1, cv::COLOR_GRAY2BGR);
      else
        sample1 = img1;
      if (img1.empty())  break;
      filename.push_back(file);
      if (img1.rows != static_cast<int>(in_h)
          || img1.cols != static_cast<int>(in_w)
          || img1.channels() != static_cast<int>(in_c)) {
        cout << "warining! image " << file << " size is wrong!" << endl;
        cout << "input size should be :" << in_c << " " << in_h << " "
            << in_w << endl;
        cout << "now input size is :" << img1.channels() << " "
            << img1.rows << " " << img1.cols << endl;
        cout << "img is going to resize!" << endl;
      }

      if (img1.rows != static_cast<int>(in_h)
          || img1.cols != static_cast<int>(in_w))
        resize(sample1, img_resized1, cv::Size(in_h, in_w));
      else
        img_resized1 = sample1;

      count_batch++;
      cv::Mat img_float1;
      img_resized1.convertTo(img_float1, cv_fc3);
      cv::Mat img_normalized1;
      cv::subtract(img_float1, mean_, img_normalized1);
      cv::Mat img_normalized2;
      std::cout << "input_std is: " << input_std << std::endl;
      cv::divide(input_std, img_normalized1, img_normalized2);
      cv::split(img_normalized2, (input_imgs)[count_batch-1]);
    }
    img_processed += count_batch;
    count_use = count_batch;
    filename_use = filename;
}

int main(int argc, char* argv[]) {
  // 0. process arguments
  if (argc != 10) {
    printf("wrong args\n");
    std::cerr << "Usage: " << argv[0]
              << " offline_model.cambricon image_list_file"
              << " labels.txt image_mean_file symbol_name"
              << " color_mode(rgb or bgr) label_offset input_datatype"
              << std::endl;
    exit(0);
  }
  string mef_name = (string)argv[1];  // offline_model.mef
  file_name = (string)(argv[2]);  // image file list
  names = (string)argv[3];  // classification label file
  string input_std_s = (string)argv[5];
  input_std = static_cast<double>(atof(input_std_s.c_str()));
  string symbol_name = (string)argv[6];  // offline model symbol name
  string color_mode = (string)argv[7];
  string offset_str = (string)argv[8];
  int label_offset = std::stoi(offset_str);
  int number_images = 0;
  string input_datatype = (string)argv[9];  // float/uint8
  PARAM_CHECK((color_mode == "bgr" || color_mode == "rgb"),
      ("color mode must be rgb or bgr"));
  PARAM_CHECK((input_datatype == "float" || input_datatype == "uint8"),
      ("input_datatype must be float or uint8"));
  std::cout << "color_mode = " << color_mode << std::endl;
  std::cout << "label_offset = " << label_offset << std::endl;
  std::cout << "input_datatype = " << input_datatype << std::endl;
  std::cout << "input_std = " << input_std << std::endl;

  printf("load file: %s\n", mef_name.c_str());

  string means_data = (string)argv[4];
  char* means_data_c = (char*)means_data.c_str();  // NOLINT
  char* p;
  for (p = strtok(means_data_c, ","); p; p = strtok(NULL, ",")) {  // NOLINT
    float means_val = std::atof(p);
    means.push_back(means_val);
  }
  cnrtInit(0);
  unsigned dev_num;
  cnrtGetDeviceCount(&dev_num);
  if (dev_num == 0) return CNRT_RET_ERR_NODEV;
  cnrtDev_t dev;
  cnrtGetDeviceHandle(&dev, 0);
  cnrtSetCurrentDevice(dev);
  // 2. load model and get function
  cnrtModel_t model;
  cnrtLoadModel(&model, mef_name.c_str());
  cnrtFunction_t function;
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, model, symbol_name.c_str());
  // 3. get function's I/O DataDesc
  int inputNum, outputNum;
  cnrtDataDescArray_t inputDescS, outputDescS;
  cnrtGetInputDataDesc(&inputDescS, &inputNum, function);
  cnrtGetOutputDataDesc(&outputDescS, &outputNum, function);
  std::cout << "inputNum: " << inputNum << std::endl;
  std::cout << "outputNum: " << outputNum << std::endl;
  // 4. allocate I/O data space on CPU memory and prepare Input data
  void** inputCpuPtrS  = (void**) malloc (sizeof(void*) * inputNum);   //NOLINT
  void** outputCpuPtrS = (void**) malloc (sizeof(void*) * outputNum);  //NOLINT

  int input_dataCount, output_dataCount;
  cnrtDataDesc_t input_dataDesc = (inputDescS[0]);
  auto cnrt_datatype = (input_datatype == "float")? CNRT_FLOAT32 : CNRT_UINT8;
  cnrtSetHostDataLayout(input_dataDesc, cnrt_datatype, CNRT_NCHW);
  cnrtGetHostDataCount(input_dataDesc, &input_dataCount);
  cnrtGetDataShape(input_dataDesc, &in_n, &in_c, &in_h, &in_w);
  int element_size = (input_datatype == "float")?
      sizeof(float) : sizeof(uint8_t);
  dataBuffer = (void*) malloc (element_size * input_dataCount);  // NOLINT
  inputCpuPtrS[0] = dataBuffer;

  cnrtDataDesc_t output_dataDesc = (outputDescS[0]);
  cnrtSetHostDataLayout(output_dataDesc, CNRT_FLOAT32, CNRT_NCHW);
  cnrtGetHostDataCount(output_dataDesc, &output_dataCount);
  cnrtGetDataShape(output_dataDesc, &out_n, &out_c, &out_h, &out_w);
  output_cpu = (float*) malloc (sizeof(float) * output_dataCount);  // NOLINT
  outputCpuPtrS[0] = (void*)output_cpu;  // NOLINT

  std::cout << "allocate I/O data space on MLU memory " << std::endl;
  // 5. allocate I/O data space on MLU memory
  void **inputNetPtrS;
  void **outputNetPtrS;
  cnrtMallocByDescArray(&inputNetPtrS ,  inputDescS,  inputNum);
  cnrtMallocByDescArray(&outputNetPtrS, outputDescS, outputNum);

  // 7. create stream and run function
  cnrtStream_t stream;
  CNRT_CHECK(cnrtCreateStream(&stream));
  cnrtInitFunctionMemory(function, CNRT_FUNC_TYPE_BLOCK0);

  int count = 1;
  float time_use;
  struct timeval tpstart, tpend;
  int64_t copyin_times = 0;
  int64_t hardware_times = 0;
  int64_t copyout_times = 0;
  int64_t softmax_print_times = 0;

  while (!file_clear) {
    int expected_label = 0;
    number_images++;
    if (count > 0) {
      if (input_datatype == "float") {
        get_img<float>(color_mode, &expected_label);
      } else {
        get_img<uint8_t>(color_mode, &expected_label);
      }

      if (!file_clear) {
        // 6. copy Input data
        std::cout << "copy input data " << std::endl;
        gettimeofday(&tpstart, NULL);
        cnrtMemcpyByDescArray(inputNetPtrS, inputCpuPtrS, inputDescS,
            inputNum, CNRT_MEM_TRANS_DIR_HOST2DEV);
        gettimeofday(&tpend, NULL);
        std::cout << "copy input data end" << std::endl;
        copyin_times += 1000000 * (tpend.tv_sec - tpstart.tv_sec)
            + tpend.tv_usec - tpstart.tv_usec;
        void *param[2] = {inputNetPtrS[0], outputNetPtrS[0]};
        cnrtDim3_t dim = {1, 1, 1};
        std::cout << "InvokeFunction " << std::endl;
        gettimeofday(&tpstart, NULL);
        cnrtInvokeFunction(function, dim, param, CNRT_FUNC_TYPE_BLOCK0,
            stream, NULL);
        cnrtSyncStream(stream);
        gettimeofday(&tpend, NULL);
        hardware_times += 1000000 * (tpend.tv_sec - tpstart.tv_sec)
            + tpend.tv_usec - tpstart.tv_usec;
        // 8. copy back Output data and write it to file
        std::cout << "copy output data " << std::endl;
        gettimeofday(&tpstart, NULL);
        cnrtMemcpyByDescArray(outputCpuPtrS, outputNetPtrS,
            outputDescS, outputNum, CNRT_MEM_TRANS_DIR_DEV2HOST);
        gettimeofday(&tpend, NULL);
        copyout_times += 1000000 * (tpend.tv_sec - tpstart.tv_sec)
            + tpend.tv_usec - tpstart.tv_usec;

        std::cout << "begin softmax " << std::endl;
        gettimeofday(&tpstart, NULL);
        softmax_print(label_offset, expected_label);
        std::cout << "end softmax" << std::endl;
        gettimeofday(&tpend, NULL);
        softmax_print_times += 1000000 * (tpend.tv_sec - tpstart.tv_sec)
            + tpend.tv_usec - tpstart.tv_usec;
        count++;
      }
    }
  }
  number_images--;
  std::cout << "--------------------time_statistics-------------------"
  << std::endl;
  std::cout << "average copyin time: " <<
  (static_cast<float>(copyin_times)/(count -1))/1000 << "ms" << std::endl;
  std::cout << "average hardware time: "
  << (static_cast<float>(hardware_times)/(count -1))/1000 << "ms" << std::endl;
  std::cout << "average copyout time: "
  << (static_cast<float>(copyout_times)/(count -1))/1000 << "ms" << std::endl;
  std::cout << "average softmax time: " <<
  (static_cast<float>(softmax_print_times)/(count -1))/1000 << "ms" <<
  std::endl;
  std::cout << "a picture execution time: " <<
  (static_cast<float>(copyin_times)/(count -1))/1000 +
  (static_cast<float>(hardware_times)/(count -1))/1000 +
  (static_cast<float>(copyout_times)/(count -1))/1000 +
  (static_cast<float>(softmax_print_times)/(count -1))/1000 << "ms"
  << std::endl;
  std::cout << "--------------------accuracy-------------------" << std::endl;
  std::cout << "image count = " << number_images << std::endl;
  std::cout << "TOP1_COUNT = " << top1_count << ", TOP1_ACCURACY = "
      << static_cast<float>(top1_count)
      / static_cast<float>(number_images) << std::endl;
  std::cout << "TOP5_COUNT = " << top5_count << ", TOP5_ACCURACY = "
      << static_cast<float>(top5_count)
      /static_cast<float>(number_images) << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2)
      << "{\n" 
      << "  \"output\": {\n"
      << "    \"accuracy\": {\n"
      << "      \"top1\": " << static_cast<float>(top1_count) * 100 /
                               static_cast<float>(number_images) << ",\n"
      << "      \"top5\": " << static_cast<float>(top5_count) * 100 /
                               static_cast<float>(number_images) << "\n"
      << "    },\n"
      << "    \"performance\": {\n"
      << "      \"copyintime\": " << static_cast<float>(
                                       copyin_times) / (count - 1) << ",\n"
      << "      \"hardwaretime\": " << static_cast<float>(
                                       hardware_times) / (count - 1) << ",\n"
      << "      \"hardwarefps\": " << static_cast<float>(number_images) * 1000000 / 
                                      static_cast<float>(hardware_times) << ",\n"
      << "      \"copyouttime\": " << static_cast<float>(
                                       copyout_times) / (count - 1) << ",\n"
      << "      \"endtoendtime\": " << static_cast<float>(copyin_times +
                                       hardware_times + copyout_times +
                                       softmax_print_times) / (count - 1) << ",\n"
      << "      \"endtoendfps\": " << static_cast<float>(number_images) * 1000000 /
                                      static_cast<float>(copyin_times +
                                        hardware_times + copyout_times +
                                        softmax_print_times) << "\n"
      << "    }\n"
      << "  }\n"
      << "}";
  std::cout << oss.str() << std::endl;
  char* output_to_json_file = getenv("OUTPUT_TO_JSON_FILE");
  if (output_to_json_file != nullptr) {
    std::string env_str(output_to_json_file);
    std::transform(env_str.begin(), env_str.end(), 
                   env_str.begin(), ::tolower);
    if (env_str == "1" || env_str == "true") {
      std::ofstream json_ofs("output_summary.json", std::ios::binary);
      if (!json_ofs.is_open()) {
        std::cerr << "failed to open json file" << std::endl;
        exit(-1);
      }
      json_ofs << oss.str();
      json_ofs.close();
    }
  }

  // 8. free memory space
  for (int i = 0; i < inputNum; i++) {
      free(inputCpuPtrS[i]);
  }
  for (int i = 0; i < outputNum; i++) {
      free(outputCpuPtrS[i]);
  }
  free(inputCpuPtrS);
  free(outputCpuPtrS);
  cnrtFreeArray(inputNetPtrS, inputNum);
  cnrtFreeArray(outputNetPtrS, outputNum);
  cnrtDestroyStream(stream);
  cnrtDestroyFunction(function);
  cnrtUnloadModel(model);
  cnrtDestroy();
  return 0;
}
