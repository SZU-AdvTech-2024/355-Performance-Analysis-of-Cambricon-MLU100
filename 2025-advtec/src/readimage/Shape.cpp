#include "Shape.hpp"

Shape::Shape(uint n, uint c, uint h, uint w):n_(n),c_(c),h_(h),w_(w){}
uint Shape::count() const{ return n_ * c_ * h_ * w_;}
uint Shape::chw() const{ return c_ * h_ * w_;}
uint Shape::getSize() const{ return count() * sizeof(float);}