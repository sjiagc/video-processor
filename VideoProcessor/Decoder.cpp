#include "Decoder.hpp"

#include "Utils.hpp"

#define START_TIMER auto start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(print_message) std::cout << print_message << \
    std::chrono::duration_cast<std::chrono::milliseconds>( \
    std::chrono::high_resolution_clock::now() - start).count() \
    << " ms " << std::endl;

static const char* GetVideoCodecString(cudaVideoCodec eCodec) {
    static struct {
        cudaVideoCodec eCodec;
        const char* name;
    } aCodecName[] = {
        { cudaVideoCodec_MPEG1,     "MPEG-1"       },
        { cudaVideoCodec_MPEG2,     "MPEG-2"       },
        { cudaVideoCodec_MPEG4,     "MPEG-4 (ASP)" },
        { cudaVideoCodec_VC1,       "VC-1/WMV"     },
        { cudaVideoCodec_H264,      "AVC/H.264"    },
        { cudaVideoCodec_JPEG,      "M-JPEG"       },
        { cudaVideoCodec_H264_SVC,  "H.264/SVC"    },
        { cudaVideoCodec_H264_MVC,  "H.264/MVC"    },
        { cudaVideoCodec_HEVC,      "H.265/HEVC"   },
        { cudaVideoCodec_VP8,       "VP8"          },
        { cudaVideoCodec_VP9,       "VP9"          },
        { cudaVideoCodec_AV1,       "AV1"          },
        { cudaVideoCodec_NumCodecs, "Invalid"      },
        { cudaVideoCodec_YUV420,    "YUV  4:2:0"   },
        { cudaVideoCodec_YV12,      "YV12 4:2:0"   },
        { cudaVideoCodec_NV12,      "NV12 4:2:0"   },
        { cudaVideoCodec_YUYV,      "YUYV 4:2:2"   },
        { cudaVideoCodec_UYVY,      "UYVY 4:2:2"   },
    };

    if (eCodec >= 0 && eCodec <= cudaVideoCodec_NumCodecs) {
        return aCodecName[eCodec].name;
    }
    for (int i = cudaVideoCodec_NumCodecs + 1; i < sizeof(aCodecName) / sizeof(aCodecName[0]); i++) {
        if (eCodec == aCodecName[i].eCodec) {
            return aCodecName[eCodec].name;
        }
    }
    return "Unknown";
}

static const char* GetVideoChromaFormatString(cudaVideoChromaFormat eChromaFormat) {
    static struct {
        cudaVideoChromaFormat eChromaFormat;
        const char* name;
    } aChromaFormatName[] = {
        { cudaVideoChromaFormat_Monochrome, "YUV 400 (Monochrome)" },
        { cudaVideoChromaFormat_420,        "YUV 420"              },
        { cudaVideoChromaFormat_422,        "YUV 422"              },
        { cudaVideoChromaFormat_444,        "YUV 444"              },
    };

    if (eChromaFormat >= 0 && eChromaFormat < sizeof(aChromaFormatName) / sizeof(aChromaFormatName[0])) {
        return aChromaFormatName[eChromaFormat].name;
    }
    return "Unknown";
}


Decoder::Decoder(CUcontext inCuContext)
	: mCuContext(inCuContext)
    , mCtxLock(nullptr)
    , mParser(nullptr)
    , mDecoder(nullptr)
    , mCuvidStream(0)
    , mDecodedFrame(0)
    , mDecodedFrameReturned(0)
{
    NVDEC_API_CALL(cuvidCtxLockCreate(&mCtxLock, mCuContext));

    CUVIDPARSERPARAMS videoParserParameters = {};
    videoParserParameters.CodecType = cudaVideoCodec_H264;
    videoParserParameters.ulMaxNumDecodeSurfaces = 1;
    videoParserParameters.ulClockRate = 0;
    videoParserParameters.ulMaxDisplayDelay = 1;
    videoParserParameters.pUserData = this;
    videoParserParameters.pfnSequenceCallback = HandleVideoSequenceProc;
    videoParserParameters.pfnDecodePicture = HandlePictureDecodeProc;
    videoParserParameters.pfnDisplayPicture = HandlePictureDisplayProc;
    videoParserParameters.pfnGetOperatingPoint = HandleOperatingPointProc;
    NVDEC_API_CALL(cuvidCreateVideoParser(&mParser, &videoParserParameters));
}

Decoder::~Decoder()
{
    if (mParser) {
        cuvidDestroyVideoParser(mParser);
    }
    cuCtxPushCurrent(mCuContext);
    if (mDecoder) {
        cuvidDestroyDecoder(mDecoder);
    }

    {
        std::lock_guard<std::mutex> lock(mVPFrameLock);
        for (CUdeviceptr pFrame : mVPFrames) {
            cuMemFree(pFrame);
        }
    }
    cuCtxPopCurrent(NULL);

    cuvidCtxLockDestroy(mCtxLock);
}

int
Decoder::decode(uint8_t* inData, size_t inLength)
{
    mDecodedFrame = 0;
    mDecodedFrameReturned = 0;
    CUVIDSOURCEDATAPACKET packet = { 0 };
    packet.payload = inData;
    packet.payload_size = static_cast<unsigned long>(inLength);
    packet.flags = 0 | CUVID_PKT_TIMESTAMP;
    packet.timestamp = 0;
    if (!inData || inLength == 0) {
        packet.flags |= CUVID_PKT_ENDOFSTREAM;
    }
    NVDEC_API_CALL(cuvidParseVideoData(mParser, &packet));
    mCuvidStream = 0;

    return mDecodedFrame;
}

CUdeviceptr
Decoder::getFrame()
{
    if (mDecodedFrame > 0) {
        std::lock_guard<std::mutex> lock(mVPFrameLock);
        mDecodedFrame--;
        return mVPFrames[mDecodedFrameReturned++];
    }
    return 0;
}


int
Decoder::HandleVideoSequence(CUVIDEOFORMAT* pVideoFormat)
{
    START_TIMER
    std::cout << "Video Input Information" << std::endl
        << "\tCodec        : " << GetVideoCodecString(pVideoFormat->codec) << std::endl
        << "\tFrame rate   : " << pVideoFormat->frame_rate.numerator << "/" << pVideoFormat->frame_rate.denominator
        << " = " << 1.0 * pVideoFormat->frame_rate.numerator / pVideoFormat->frame_rate.denominator << " fps" << std::endl
        << "\tSequence     : " << (pVideoFormat->progressive_sequence ? "Progressive" : "Interlaced") << std::endl
        << "\tCoded size   : [" << pVideoFormat->coded_width << ", " << pVideoFormat->coded_height << "]" << std::endl
        << "\tDisplay area : [" << pVideoFormat->display_area.left << ", " << pVideoFormat->display_area.top << ", "
        << pVideoFormat->display_area.right << ", " << pVideoFormat->display_area.bottom << "]" << std::endl
        << "\tChroma       : " << GetVideoChromaFormatString(pVideoFormat->chroma_format) << std::endl
        << "\tBit depth    : " << pVideoFormat->bit_depth_luma_minus8 + 8
        << std::endl;

    int decodeSurface = pVideoFormat->min_num_decode_surfaces;

    CUVIDDECODECAPS decodecaps;
    memset(&decodecaps, 0, sizeof(decodecaps));

    decodecaps.eCodecType = pVideoFormat->codec;
    decodecaps.eChromaFormat = pVideoFormat->chroma_format;
    decodecaps.nBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCuContext));
    NVDEC_API_CALL(cuvidGetDecoderCaps(&decodecaps));
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));

    if (!decodecaps.bIsSupported) {
        std::cerr << "Codec not supported on this GPU" << std::endl;
        return decodeSurface;
    }

    if ((pVideoFormat->coded_width > decodecaps.nMaxWidth) ||
        (pVideoFormat->coded_height > decodecaps.nMaxHeight)) {
        std::ostringstream errorString;
        errorString << std::endl
            << "Resolution          : " << pVideoFormat->coded_width << "x" << pVideoFormat->coded_height << std::endl
            << "Max Supported (wxh) : " << decodecaps.nMaxWidth << "x" << decodecaps.nMaxHeight << std::endl
            << "Resolution not supported on this GPU";

        const std::string cErr = errorString.str();
        std::cerr << cErr << std::endl;
        return decodeSurface;
    }

    if ((pVideoFormat->coded_width >> 4) * (pVideoFormat->coded_height >> 4) > decodecaps.nMaxMBCount) {
        std::ostringstream errorString;
        errorString << std::endl
            << "MBCount             : " << (pVideoFormat->coded_width >> 4) * (pVideoFormat->coded_height >> 4) << std::endl
            << "Max Supported mbcnt : " << decodecaps.nMaxMBCount << std::endl
            << "MBCount not supported on this GPU";

        const std::string cErr = errorString.str();
        std::cerr << cErr << std::endl;
        return decodeSurface;
    }

    if (mWidth && mLumaHeight && mChromaHeight) {
        // cuvidCreateDecoder() has been called before, and now there's possible config change
        return ReconfigureDecoder(pVideoFormat);
    }

    // eCodec has been set in the constructor (for parser). Here it's set again for potential correction
    mCodec = pVideoFormat->codec;
    mChromaFormat = pVideoFormat->chroma_format;
    mBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
    mBPP = mBitDepthMinus8 > 0 ? 2 : 1;
    // Check if output format supported. If not, check falback options
    if (!(decodecaps.nOutputFormatMask & (1 << mOutputFormat))) {
        std::cerr << "No supported output format found" << std::endl;
        throw std::exception();
    }

    mVideoFormat = *pVideoFormat;

    CUVIDDECODECREATEINFO videoDecodeCreateInfo = { 0 };
    videoDecodeCreateInfo.CodecType = pVideoFormat->codec;
    videoDecodeCreateInfo.ChromaFormat = pVideoFormat->chroma_format;
    videoDecodeCreateInfo.OutputFormat = mOutputFormat;
    videoDecodeCreateInfo.bitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
    if (pVideoFormat->progressive_sequence)
        videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    else
        videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
    videoDecodeCreateInfo.ulNumOutputSurfaces = 2;
    // With PreferCUVID, JPEG is still decoded by CUDA while video is decoded by NVDEC hardware
    videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    videoDecodeCreateInfo.ulNumDecodeSurfaces = decodeSurface;
    videoDecodeCreateInfo.vidLock = mCtxLock;
    videoDecodeCreateInfo.ulWidth = pVideoFormat->coded_width;
    videoDecodeCreateInfo.ulHeight = pVideoFormat->coded_height;
    if (m_nMaxWidth < (int)pVideoFormat->coded_width)
        m_nMaxWidth = pVideoFormat->coded_width;
    if (m_nMaxHeight < (int)pVideoFormat->coded_height)
        m_nMaxHeight = pVideoFormat->coded_height;
    videoDecodeCreateInfo.ulMaxWidth = m_nMaxWidth;
    videoDecodeCreateInfo.ulMaxHeight = m_nMaxHeight;

    mWidth = pVideoFormat->display_area.right - pVideoFormat->display_area.left;
    mLumaHeight = pVideoFormat->display_area.bottom - pVideoFormat->display_area.top;
    videoDecodeCreateInfo.ulTargetWidth = pVideoFormat->coded_width;
    videoDecodeCreateInfo.ulTargetHeight = pVideoFormat->coded_height;

    videoDecodeCreateInfo.ulTargetWidth = mWidth;
    videoDecodeCreateInfo.ulTargetHeight = mLumaHeight;

    mChromaHeight = (int)(ceil(mLumaHeight * 0.5));
    mNumChromaPlanes = 1;
    mSurfaceHeight = videoDecodeCreateInfo.ulTargetHeight;
    mSurfaceWidth = videoDecodeCreateInfo.ulTargetWidth;
    mDisplayRect.b = videoDecodeCreateInfo.display_area.bottom;
    mDisplayRect.t = videoDecodeCreateInfo.display_area.top;
    mDisplayRect.l = videoDecodeCreateInfo.display_area.left;
    mDisplayRect.r = videoDecodeCreateInfo.display_area.right;

    std::cout << "Video Decoding Params:" << std::endl
        << "\tNum Surfaces : " << videoDecodeCreateInfo.ulNumDecodeSurfaces << std::endl
        << "\tCrop         : [" << videoDecodeCreateInfo.display_area.left << ", " << videoDecodeCreateInfo.display_area.top << ", "
        << videoDecodeCreateInfo.display_area.right << ", " << videoDecodeCreateInfo.display_area.bottom << "]" << std::endl
        << "\tResize       : " << videoDecodeCreateInfo.ulTargetWidth << "x" << videoDecodeCreateInfo.ulTargetHeight << std::endl
        << "\tDeinterlace  : " << std::vector<const char*>{"Weave", "Bob", "Adaptive"} [videoDecodeCreateInfo.DeinterlaceMode]
        << std::endl;

    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCuContext));
    NVDEC_API_CALL(cuvidCreateDecoder(&mDecoder, &videoDecodeCreateInfo));
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(nullptr));
    STOP_TIMER("Session Initialization Time: ");
    return decodeSurface;
}

int
Decoder::HandlePictureDecode(CUVIDPICPARAMS* pPicParams)
{
    if (!mDecoder)
    {
        std::cerr << "Decoder not initialized." << std::endl;
        throw std::exception();
        return 0;
    }
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCuContext));
    NVDEC_API_CALL(cuvidDecodePicture(mDecoder, pPicParams));
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(nullptr));
    return 1;
}

int
Decoder::HandlePictureDisplay(CUVIDPARSERDISPINFO* pDispInfo)
{
    CUVIDPROCPARAMS videoProcessingParameters = {};
    videoProcessingParameters.progressive_frame = pDispInfo->progressive_frame;
    videoProcessingParameters.second_field = pDispInfo->repeat_first_field + 1;
    videoProcessingParameters.top_field_first = pDispInfo->top_field_first;
    videoProcessingParameters.unpaired_field = pDispInfo->repeat_first_field < 0;
    videoProcessingParameters.output_stream = mCuvidStream;

    CUdeviceptr dpSrcFrame = 0;
    unsigned int nSrcPitch = 0;
    CUDA_DRVAPI_CALL(cuCtxPushCurrent(mCuContext));
    NVDEC_API_CALL(cuvidMapVideoFrame(mDecoder, pDispInfo->picture_index, &dpSrcFrame, &nSrcPitch, &videoProcessingParameters));

    CUVIDGETDECODESTATUS DecodeStatus;
    memset(&DecodeStatus, 0, sizeof(DecodeStatus));
    CUresult result = cuvidGetDecodeStatus(mDecoder, pDispInfo->picture_index, &DecodeStatus);
    if (result == CUDA_SUCCESS && (DecodeStatus.decodeStatus == cuvidDecodeStatus_Error
        || DecodeStatus.decodeStatus == cuvidDecodeStatus_Error_Concealed)) {
        std::cerr << "Decode Error occurred for picture" << std::endl;
    }

    CUdeviceptr pDecodedFrame = 0;
    {
        std::lock_guard<std::mutex> lock(mVPFrameLock);
        if ((unsigned)++mDecodedFrame > mVPFrames.size()) {
            // Not enough frames in stock
            CUdeviceptr pFrame = 0;
            CUDA_DRVAPI_CALL(cuMemAlloc((CUdeviceptr*)&pFrame, GetFrameSize()));
            mVPFrames.push_back(pFrame);
        }
        pDecodedFrame = mVPFrames[mDecodedFrame - 1];
    }

    // Copy luma plane
    CUDA_MEMCPY2D m = { 0 };
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = dpSrcFrame;
    m.srcPitch = nSrcPitch;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstDevice = pDecodedFrame;
    m.dstPitch = GetWidth() * mBPP;
    m.WidthInBytes = GetWidth() * mBPP;
    m.Height = mLumaHeight;
    CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&m, mCuvidStream));

    // Copy chroma plane
    // NVDEC output has luma height aligned by 2. Adjust chroma offset by aligning height
    m.srcDevice = (CUdeviceptr)((uint8_t*)dpSrcFrame + m.srcPitch * ((mSurfaceHeight + 1) & ~1));
    m.dstDevice = pDecodedFrame + m.dstPitch * mLumaHeight;
    m.Height = mChromaHeight;
    CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&m, mCuvidStream));

    CUDA_DRVAPI_CALL(cuStreamSynchronize(mCuvidStream));
    CUDA_DRVAPI_CALL(cuCtxPopCurrent(NULL));

    NVDEC_API_CALL(cuvidUnmapVideoFrame(mDecoder, dpSrcFrame));
    return 1;
}

int
Decoder::GetOperatingPoint(CUVIDOPERATINGPOINTINFO* pOPInfo)
{
    return -1;
}

int
Decoder::ReconfigureDecoder(CUVIDEOFORMAT* pVideoFormat)
{
    return -1;
}
