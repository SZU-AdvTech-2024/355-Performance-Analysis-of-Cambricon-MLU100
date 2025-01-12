include_directories(src/infernce)

set(
    TASK_BREAKDOWN_SOURCE 
    ../src/inference/ModelManager.hpp
    ../src/inference/ModelManager.cpp
    ../src/inference/ContextManager.hpp
    ../src/inference/ContextManager.cpp
    ../src/readimage/ImageReader.cpp 
    ../src/readimage/Shape.cpp
    task_execution_breakdown.cpp)


message(STATUS "${TASK_BREAKDOWN_SOURCE}")

add_executable(infer_breakdown infer_breakdown.cpp)
target_link_libraries(infer_breakdown cnrt ${OpenCV_LIBS})


add_executable(task_execution_breakdown ${TASK_BREAKDOWN_SOURCE})
target_link_libraries(task_execution_breakdown cnrt ${OpenCV_LIBS})
