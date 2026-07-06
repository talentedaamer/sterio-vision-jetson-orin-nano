/*
 * Custom NvDsInfer bbox parser for Ultralytics-exported YOLOv8 TensorRT
 * engines (model.export(format='engine'), no `nms=True`/`end2end`).
 *
 * Output tensor layout: [1, 4 + num_classes, num_anchors]
 * (e.g. 1x84x8400 for COCO-80 @ 640x640 input). Per anchor column:
 *   [0..3]            = box in cx,cy,w,h, already decoded to input-pixel
 *                        (network) coordinate space by the exported graph
 *   [4..4+num_classes) = per-class confidence, sigmoid already applied
 *
 * This function only decodes boxes and applies the confidence threshold.
 * NMS is left to gst-nvinfer's own clustering (cluster-mode=2 in the PGIE
 * config) -- do not double-NMS here.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

// Manual IEEE-754 half -> float conversion. Avoids a hard dependency on
// cuda_fp16.h purely for a scalar conversion used only if the engine's
// output layer happens to be exported as FP16 rather than FP32.
float halfToFloat(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h & 0x7C00u) >> 10;
    uint32_t mant = (h & 0x03FFu);
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

inline float readVal(const void *base, NvDsInferDataType dtype, size_t idx) {
    switch (dtype) {
        case FLOAT:
            return static_cast<const float *>(base)[idx];
        case HALF:
            return halfToFloat(static_cast<const uint16_t *>(base)[idx]);
        default:
            return 0.0f;
    }
}

}  // namespace

extern "C" bool NvDsInferParseCustomYoloV8(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList) {
    if (outputLayersInfo.empty()) {
        return false;
    }

    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    if (layer.inferDims.numDims < 2) {
        return false;
    }

    // Batch dim is already stripped by nvinfer: dims are [channels, anchors].
    const int numChannels = layer.inferDims.d[0];
    const int numAnchors = layer.inferDims.d[1];
    const int numClasses = numChannels - 4;
    if (numClasses <= 0 || numAnchors <= 0) {
        return false;
    }

    const void *outputData = layer.buffer;
    const float defaultThresh =
        detectionParams.perClassThreshold.empty() ? 0.25f : detectionParams.perClassThreshold[0];

    objectList.reserve(static_cast<size_t>(numAnchors) / 8);

    for (int a = 0; a < numAnchors; ++a) {
        float bestScore = 0.0f;
        int bestClass = -1;

        for (int c = 0; c < numClasses; ++c) {
            const size_t idx = static_cast<size_t>(4 + c) * numAnchors + a;
            const float score = readVal(outputData, layer.dataType, idx);
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }

        if (bestClass < 0) {
            continue;
        }

        const float threshold = static_cast<size_t>(bestClass) < detectionParams.perClassThreshold.size()
                                     ? detectionParams.perClassThreshold[bestClass]
                                     : defaultThresh;
        if (bestScore < threshold) {
            continue;
        }

        const float cx = readVal(outputData, layer.dataType, static_cast<size_t>(0) * numAnchors + a);
        const float cy = readVal(outputData, layer.dataType, static_cast<size_t>(1) * numAnchors + a);
        const float w = readVal(outputData, layer.dataType, static_cast<size_t>(2) * numAnchors + a);
        const float h = readVal(outputData, layer.dataType, static_cast<size_t>(3) * numAnchors + a);

        const float netW = static_cast<float>(networkInfo.width);
        const float netH = static_cast<float>(networkInfo.height);

        float left = std::max(0.0f, std::min(cx - w * 0.5f, netW));
        float top = std::max(0.0f, std::min(cy - h * 0.5f, netH));
        float width = std::min(w, netW - left);
        float height = std::min(h, netH - top);

        if (width <= 0.0f || height <= 0.0f) {
            continue;
        }

        NvDsInferObjectDetectionInfo obj;
        obj.classId = static_cast<unsigned int>(bestClass);
        obj.detectionConfidence = bestScore;
        obj.left = left;
        obj.top = top;
        obj.width = width;
        obj.height = height;
        objectList.push_back(obj);
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV8);
