#ifndef LABELREADER_HPP
#define LABELREADER_HPP

#include <iostream>
#include <fstream>
#include <vector>

class LabelReader{
public:
    LabelReader(const std::string& label_file);

    std::vector<std::string> label_vector_;  
};

#endif // LABELREADER_HPP