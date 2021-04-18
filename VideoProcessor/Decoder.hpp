#pragma once

#include <cuda.h>
#include <cuviddec.h>
#include <nvcuvid.h>

#include <cstdint>
#include <istream>
#include <mutex>
#include <vector>

class Decoder {
public:
	Decoder(CUcontext inCuContext);

	~Decoder();

	int decode(uint8_t* inData, size_t inLength);

    CUdeviceptr getFrame();
    CUVIDEOFORMAT GetVideoFormatInfo() { return mVideoFormat; }

    int GetWidth() {
        return (mWidth + 1) & ~1;
    }

    int GetHeight() { return mLumaHeight; }

private:
    struct Rect {
        int l, t, r, b;
    };

    static int CUDAAPI HandleVideoSequenceProc(void* pUserData, CUVIDEOFORMAT* pVideoFormat) { return ((Decoder*)pUserData)->HandleVideoSequence(pVideoFormat); }
    static int CUDAAPI HandlePictureDecodeProc(void* pUserData, CUVIDPICPARAMS* pPicParams) { return ((Decoder*)pUserData)->HandlePictureDecode(pPicParams); }
    static int CUDAAPI HandlePictureDisplayProc(void* pUserData, CUVIDPARSERDISPINFO* pDispInfo) { return ((Decoder*)pUserData)->HandlePictureDisplay(pDispInfo); }
    static int CUDAAPI HandleOperatingPointProc(void* pUserData, CUVIDOPERATINGPOINTINFO* pOPInfo) { return ((Decoder*)pUserData)->GetOperatingPoint(pOPInfo); }

    int GetFrameSize() { return GetWidth() * (mLumaHeight + (mChromaHeight * mNumChromaPlanes)) * mBPP; }

    int HandleVideoSequence(CUVIDEOFORMAT* pVideoFormat);
    int HandlePictureDecode(CUVIDPICPARAMS* pPicParams);
    int HandlePictureDisplay(CUVIDPARSERDISPINFO* pDispInfo);
    int GetOperatingPoint(CUVIDOPERATINGPOINTINFO* pOPInfo);
    int ReconfigureDecoder(CUVIDEOFORMAT* pVideoFormat);

	CUcontext mCuContext;
	CUvideoctxlock mCtxLock;
	CUvideoparser mParser;
	CUvideodecoder mDecoder;
    CUstream mCuvidStream;

    unsigned int mWidth = 0, mLumaHeight = 0, mChromaHeight = 0;
    unsigned int mNumChromaPlanes = 0;
    int mSurfaceHeight = 0;
    int mSurfaceWidth = 0;
    cudaVideoCodec mCodec = cudaVideoCodec_NumCodecs;
    cudaVideoChromaFormat mChromaFormat = cudaVideoChromaFormat_420;
    cudaVideoSurfaceFormat mOutputFormat = cudaVideoSurfaceFormat_NV12;
    int mBitDepthMinus8 = 0;
    int mBPP = 1;
    CUVIDEOFORMAT mVideoFormat = {};
    unsigned int m_nMaxWidth = 0, m_nMaxHeight = 0;
    Rect mDisplayRect = {};

    std::mutex mVPFrameLock;
    std::vector<CUdeviceptr> mVPFrames;
    int mDecodedFrame;
    int mDecodedFrameReturned;
};