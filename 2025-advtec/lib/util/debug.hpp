//
// Created by szu on 2020/6/16.
//

#ifndef MULTICARD_DEBUG_HPP
#define MULTICARD_DEBUG_HPP
#include <iostream>
#include <sstream>
#define DEBUG 1
#ifdef DEBUG
#define DEBUG_STDERR(x) (std::cerr << (x) << std::endl)
#define DEBUG_STDOUT(x) (std::cout << (x) << std::endl)
#else
#define DEBUG_STDERR(x) do{}while(0)
#define DEBUG_STDOUT(x) do{}while(0)
#endif

struct CnrtException: public std::runtime_error {
    std::string msg;
    CnrtException(const std::string &arg, const char*file, int line): std::runtime_error(arg){
        std::ostringstream o;
        o << file << ":" << line << ": " << arg;
        msg = o.str();
    }
    const char* what() const throw() {
        return msg.c_str();
    }
};
#define throw_cnrt_exception(arg) throw CnrtException(arg, __FILE__, __LINE__)
#define CHECK(statment) \
  do { \
    int ret_code = (statment); \
    if (ret_code != 0) { \
      throw_cnrt_exception("CNRT error, code: " + std::to_string(ret_code)); \
    } \
  } while (false);
#endif //MULTICARD_DEBUG_HPP
