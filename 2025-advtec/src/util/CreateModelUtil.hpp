//
// Created by szu on 11/18/24.
//

#ifndef DAC_MODELUTIL_HPP
#define DAC_MODELUTIL_HPP

#include "cnrt.h"
#include "iostream"
class CreateModelUtil{
public:
    static cnrtModel_t createModel(std::string& modelPath){
        cnrtModel_t model;
        CNRT_CHECK(cnrtLoadModel(&model, modelPath.c_str()));
        return model;
    }
    static void unloadModel(cnrtModel_t& model){
        CNRT_CHECK(cnrtUnloadModel(model));
//        cnrtFunction_t function;
//        cnrtCreateFunction(&function);
//        CNRT_CHECK(cnrtExtractFunction(&function, model, "subnet0"));
    }
    static int getModelParallelism(cnrtModel_t model){
        int modelParallelism;
        CNRT_CHECK(cnrtQueryModelParallelism(model, &modelParallelism));
//        printf("\nmp: %d\n", modelParallelism);
        return modelParallelism;
    }
    static void writeRusultToFile(std::string fileName, void** outputCpuPtrS, int outputNum, int outputDataCount){
        FILE* file = fopen(fileName.c_str(), "w+");
        for (int i = 0; i < outputNum; ++i) {
            float* cpuPtr = (float *)outputCpuPtrS[i];
            for (int j = 0; j < outputDataCount; ++j) {
                fprintf(file, "[%d]: ", j);
                fprintf(file, "%f\n", cpuPtr[j]);
            }
        }
        fclose(file);
    }
};
#endif //DAC_MODELUTIL_HPP
