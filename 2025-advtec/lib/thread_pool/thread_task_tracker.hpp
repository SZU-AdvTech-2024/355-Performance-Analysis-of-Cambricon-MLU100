//
// Created by szu on 2020/7/20.
//

#ifndef MULTICARD_THREAD_TASK_TRACKER_HPP
#define MULTICARD_THREAD_TASK_TRACKER_HPP

#include <cstdint>
#include <utility>

template<typename Handler>
struct TaskTracker{
    TaskTracker() {}
    TaskTracker(Handler&& handler, int core_num, uint64_t cost): handler(std::forward<Handler>(handler)), core_num(core_num), cost(cost) {
    }
    uint64_t cost;
    int core_num;
    Handler handler;
};
#endif //MULTICARD_THREAD_TASK_TRACKER_HPP
