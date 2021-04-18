#include "Utils.hpp"

#include <cuda.h>
#include "Decoder.hpp"

#include <fstream>
#include <iostream>
#include "FramePresenterGLUT.h"
#include "ColorSpace.h"
#include "NvCodecUtils.h"

#define START_TIMER(name) auto start##name = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name) std::cout << #name" " << \
    std::chrono::duration_cast<std::chrono::milliseconds>( \
    std::chrono::high_resolution_clock::now() - start##name).count() \
    << " ms " << std::endl;

void createCudaContext(CUcontext *outContext, int inGpu, CUctx_flags inFlags) {
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, inGpu));
    char szDeviceName[80];
    ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
    std::cout << "GPU in use: " << szDeviceName << std::endl;
    ck(cuCtxCreate(outContext, inFlags, cuDevice));
}

uint8_t buffer[2 * 1024];

int
main() {
    ck(cuInit(0));
    CUcontext cuContext = nullptr;
    createCudaContext(&cuContext, 0, CU_CTX_SCHED_BLOCKING_SYNC);

    std::basic_ifstream<uint8_t, std::char_traits<uint8_t>> inputStream("sample.h264",
        std::ios::binary);
    if (!inputStream) {
        std::cerr << "Open file sample.h264 failed" << std::endl;
        return -1;
    }

    int nWidth = (1920 + 1) & ~1;
    int nPitch = 1920 * 4;

    Decoder decoder(cuContext);
    FramePresenterGLUT gInstance(cuContext, nWidth, 800);
    int& nFrame = gInstance.nFrame;

    CUdeviceptr dpFrame;
    int iMatrix = 0;
    do {
        START_TIMER(read)
        inputStream.read(buffer, sizeof(buffer));
        size_t bytesRead = inputStream.gcount();
        STOP_TIMER(read)
        START_TIMER(decode)
        int frameCount = decoder.decode(buffer, bytesRead);
        STOP_TIMER(decode)

        START_TIMER(render)
        while (frameCount--) {
            CUdeviceptr frame = decoder.getFrame();
            gInstance.GetDeviceFrameBuffer(&dpFrame, &nPitch);
            iMatrix = decoder.GetVideoFormatInfo().video_signal_description.matrix_coefficients;
            // Launch cuda kernels for colorspace conversion from raw video to raw image formats which OpenGL textures can work with
            Nv12ToColor32<BGRA32>((uint8_t*)frame, decoder.GetWidth(), (uint8_t*)dpFrame, nPitch, decoder.GetWidth(), decoder.GetHeight(), iMatrix);

            gInstance.ReleaseDeviceFrameBuffer();
        }
        STOP_TIMER(render)

        if (!inputStream) {
            break;
        }
    } while (true);

    ck(cuCtxDestroy(cuContext));
}
