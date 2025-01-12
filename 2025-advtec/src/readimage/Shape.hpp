#ifndef SHAPE_HPP
#define SHAPE_HPP


#include <iostream>
class Shape{
public:
    Shape(uint n, uint c, uint h, uint w);
    uint count() const;
    uint chw() const;
    uint getSize() const;

    uint n_;
    uint c_;
    uint h_;
    uint w_;
};

#endif //SHAPE_HPP