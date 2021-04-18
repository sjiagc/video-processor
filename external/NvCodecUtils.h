/*
* Copyright 2017-2020 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

//---------------------------------------------------------------------------
//! \file NvCodecUtils.h
//! \brief Miscellaneous classes and error checking functions.
//!
//! Used by Transcode/Encode samples apps for reading input files, mutithreading, performance measurement or colorspace conversion while decoding.
//---------------------------------------------------------------------------

#pragma once
#include <iomanip>
#include <chrono>
#include <sys/stat.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <ios>
#include <sstream>
#include <thread>
#include <list>
#include <condition_variable>

/**
* @brief Wrapper class around std::thread
*/
class NvThread
{
public:
    NvThread() = default;
    NvThread(const NvThread&) = delete;
    NvThread& operator=(const NvThread& other) = delete;

    NvThread(std::thread&& thread) : t(std::move(thread))
    {

    }

    NvThread(NvThread&& thread) : t(std::move(thread.t))
    {

    }

    NvThread& operator=(NvThread&& other)
    {
        t = std::move(other.t);
        return *this;
    }

    ~NvThread()
    {
        join();
    }

    void join()
    {
        if (t.joinable())
        {
            t.join();
        }
    }
private:
    std::thread t;
};

template <class COLOR32>
void Nv12ToColor32(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix = 0);
