#include <iostream>
#include <fstream>
#include <iomanip>


int generateImageList(std::string val_filename){
    std::ofstream o_f_stream(val_filename);
    if(!o_f_stream){
        std::cerr << "error" << std::endl;
        return -1;
    }
    for(int i = 1; i <= 100; ++i){
        o_f_stream << "/home/szu/demo_pic/ILSVRC2012_val_";
        o_f_stream << std::setfill('0') << std::setw(8) << i;
        o_f_stream << ".JPEG";
        o_f_stream << std::endl;
    }
    o_f_stream.close();
    return 1;
}

void change1(float* arr){
    arr[0] = 0;
    arr[1] = 1;
    arr[2] = 2;
    arr[3] = 3;
}

int main(){
    
    // float* arr = (float*)malloc(sizeof(float) * 4);
    // change1(arr);
    // for(int i = 0; i < 4; i++)
    //     std::cout << arr[i] << " ";
    // std::cout << std::endl;
    // free(arr);

    /**
     * generate 100 pics
    */
    std::string val_filename = "./src/config/input_image.txt";
    if(generateImageList(val_filename)){
        std::cout << "open file ok" << std::endl;
    }else{
        std::cout << "error" << std::endl;
    }

    /**
     * 
     * */    
    // char* p = nullptr;
    // std::cout << &p << std::endl;

    // std::ifstream f_stream(val_filename.c_str(), std::ios::in);
    // std::string line;
    // while(std::getline(f_stream, line)) label_vector_.push_back(line); 
    // f_stream.close();
}