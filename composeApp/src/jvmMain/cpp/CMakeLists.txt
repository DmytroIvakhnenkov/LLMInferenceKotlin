cmake_minimum_required(VERSION 3.10)
project(llama_jni)

find_package(JNI REQUIRED)

find_library(LLAMA_LIB llama PATHS /home/dmytro/llama.cpp/build/bin/)

add_library(llama_jni SHARED LlamaJni.cpp)

target_include_directories(llama_jni PRIVATE
    /home/dmytro/llama.cpp/include
    /home/dmytro/llama.cpp/ggml/include
    ${JNI_INCLUDE_DIRS}
)
target_link_libraries(llama_jni
    ${LLAMA_LIB}
    ${JNI_LIBRARIES}
)
