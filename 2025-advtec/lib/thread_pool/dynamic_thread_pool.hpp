//
// Created by szu on 2020/9/4.
//

#ifndef MULTICARD_DYNAMIC_THREAD_POOL_HPP
#define MULTICARD_DYNAMIC_THREAD_POOL_HPP
#include <cassert>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <thread_pool/fixed_function.hpp>
#include <iostream>

namespace tp {
    static constexpr size_t WAIT_SECONDS = 30;

    class DynamicThreadPool {
    public:
        using Thread = std::thread;
        using ThreadID = std::thread::id;
        using Task = tp::FixedFunction<void(), 128>;

        explicit DynamicThreadPool(size_t maxThreads = 256):
            running_flag(true),
            current_threads(0),
            idle_threads(0),
            active_num(0),
            max_threads(maxThreads){
        }

        void destroy() {
            {
                std::unique_lock<std::mutex> lock(mu);
                end_cond.wait(lock, [this]() {
                    return active_num == 0;
                });
                if(running_flag) {
                    running_flag = false;
                }
                else {
                    return;
                }
            }

            mu_cond.notify_all();

            for (auto& elem : threads) {
                assert(elem.second.joinable());
                elem.second.join();
            }
            threads.clear();
        }

        ~DynamicThreadPool() {
            destroy();
        }

        template<typename Handler>
        bool post(Handler &&handler) {
            std::lock_guard<std::mutex> guard(mu);
            if(running_flag) {
                tasks.emplace(std::forward<Handler>(handler));
                ++active_num;
                if (idle_threads > 0) {
                    mu_cond.notify_one();
                }
                else if (current_threads < max_threads) {
                    Thread t(&DynamicThreadPool::worker, this);
                    assert(threads.find(t.get_id()) == threads.end());
                    threads[t.get_id()] = std::move(t);
                    ++current_threads;
                }
                return true;
            }
            return false;
        }

        size_t threadsNum() const {
            std::lock_guard<std::mutex> guard(mu);
            return current_threads;
        }

    private:

        // disable the copy operations
        DynamicThreadPool(const DynamicThreadPool&) = delete;
        DynamicThreadPool& operator=(const DynamicThreadPool&) = delete;

        void worker() {
            while (true) {
                Task task;
                {
                    std::unique_lock<std::mutex> lock(mu);
                    ++idle_threads;
                    auto hasTimedout = !mu_cond.wait_for(
                        lock,
                        std::chrono::seconds(WAIT_SECONDS),
                        [this]() {return !running_flag || !tasks.empty();}
                    );
                    --idle_threads;
                    if (tasks.empty()) {
                        if (!running_flag) {
                            --current_threads;
                            return;
                        }
                        if (hasTimedout) {
                            --current_threads;
                            joinFinishedThreads();
                            finishedThreadIDs.emplace(std::this_thread::get_id());
                            return;
                        }
                    }
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                task();
                {
                    std::lock_guard<std::mutex> guard(mu);
                    --active_num;
                    end_cond.notify_one();
                }
            }
        }

        void joinFinishedThreads() {
            while (!finishedThreadIDs.empty()) {
                auto id = std::move(finishedThreadIDs.front());
                finishedThreadIDs.pop();
                auto iter = threads.find(id);

                assert(iter != threads.end());
                assert(iter->second.joinable());

                iter->second.join();
                threads.erase(iter);
            }
        }

        bool running_flag;
        size_t active_num;
        size_t current_threads;
        size_t idle_threads;
        size_t max_threads;

        mutable std::mutex mu;
        std::condition_variable mu_cond;
        std::condition_variable end_cond;
        std::queue<Task> tasks;
        std::queue<ThreadID> finishedThreadIDs;
        std::unordered_map<ThreadID, Thread> threads;
    };


}  // namespace dpool
#endif //MULTICARD_DYNAMIC_THREAD_POOL_HPP
