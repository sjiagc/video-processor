#pragma once

#include <iostream>


inline void check(int e, int iLine, const char* szFile) {
    if (e < 0) {
        std::cerr << "General error " << e << " at line " << iLine << " in file " << szFile << std::endl;
        throw std::exception();
    }
}

#define ck(call) check(call, __LINE__, __FILE__)


#define NVDEC_API_CALL(cuvidAPI)                                                                                \
    do {                                                                                                        \
        CUresult errorCode = cuvidAPI;                                                                          \
        if (errorCode != CUDA_SUCCESS) {                                                                        \
            std::cerr << "General error " << #cuvidAPI << " returned error " << errorCode                       \
                << " in " << __FUNCTION__ << "(" << __FILE__ << ":" << __LINE__ << ")"                          \
                << std::endl;                                                                                   \
            throw std::exception();                                                                             \
        }                                                                                                       \
    } while (0)

#define CUDA_DRVAPI_CALL(cudaAPI)                                                                               \
    do {                                                                                                        \
        CUresult errorCode = cudaAPI;                                                                           \
        if (errorCode != CUDA_SUCCESS) {                                                                        \
            const char *errName = nullptr;                                                                      \
            cuGetErrorName(errorCode, &errName);                                                                \
            std::cerr << "General error " << #cudaAPI << " returned error " << errName                          \
                << " in " << __FUNCTION__ << "(" << __FILE__ << ":" << __LINE__ << ")"                          \
                << std::endl;                                                                                   \
            throw std::exception();                                                                             \
        }                                                                                                       \
    } while (0)
