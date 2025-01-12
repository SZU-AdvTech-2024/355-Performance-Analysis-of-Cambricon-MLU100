#include "LabelReader.hpp"

LabelReader::LabelReader(const std::string& label_file){
    std::ifstream f_stream(label_file.c_str(), std::ios::in);
    std::string line;
    while(std::getline(f_stream, line)) label_vector_.push_back(line); 
    f_stream.close();
}

// int main(){
//     std::string file = "../../src/config/imagetask.txt";
//     LabelReader* label_reader = new LabelReader(file);
//     std::cout << label_reader->label_vector_.size() << std::endl;
//     for(auto &e : label_reader->label_vector_)
//         std::cout << e << std::endl;
//     delete label_reader;
// }