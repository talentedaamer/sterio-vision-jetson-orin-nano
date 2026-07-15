/*
 * Custom NvDsInfer bbox parser for Ultralytics-exported YOLO26 TensorRT
 * engines (model.export(format='engine'); end2end=True is YOLO26's
 * default, no extra export flags needed).
 *
 * YOLO26 is natively NMS-free (one-to-one head): the exported engine's
 * output is already final, post-NMS detections -- there is nothing left
 * to decode or cluster, unlike YOLOv8's raw [84, 8400] head (see the
 * predecessor of this file, nvdsinfer_custom_impl_yolov8/, removed when
 * this project switched models).
 *
 * Output tensor layout: [numDetections, 6] (batch dim already stripped by
 * nvinfer), typically [300, 6] -- each row is one final detection:
 *   [x1, y1, x2, y2, confidence, class_id]
 * in network-input-pixel coordinate space (same convention nvinfer's own
 * letterbox-unscaling expects, matching the predecessor parser). This
 * function only applies a confidence threshold -- no decode, no NMS
 * (cluster-mode=4 / "None" in the PGIE config; do not re-cluster
 * already-final detections).
 *
 * NOTE: the exact row-major layout and per-row field order above are per
 * Ultralytics' documented end-to-end export format, not yet confirmed
 * against a real exported engine on this device -- verify on first run
 * (e.g. log networkInfo/layer dims) before trusting silently.
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

extern "C" bool NvDsInferParseCustomYolo26(
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

    // Batch dim already stripped by nvinfer: dims are [numDetections, 6].
    const int numDetections = layer.inferDims.d[0];
    const int numAttributes = layer.inferDims.d[1];
    if (numAttributes != 6 || numDetections <= 0) {
        return false;
    }

    const void *outputData = layer.buffer;
    const float defaultThresh =
        detectionParams.perClassThreshold.empty() ? 0.25f : detectionParams.perClassThreshold[0];

    objectList.reserve(static_cast<size_t>(numDetections));

    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);

    for (int i = 0; i < numDetections; ++i) {
        const size_t base = static_cast<size_t>(i) * numAttributes;
        const float x1 = readVal(outputData, layer.dataType, base + 0);
        const float y1 = readVal(outputData, layer.dataType, base + 1);
        const float x2 = readVal(outputData, layer.dataType, base + 2);
        const float y2 = readVal(outputData, layer.dataType, base + 3);
        const float confidence = readVal(outputData, layer.dataType, base + 4);
        const int classId = static_cast<int>(readVal(outputData, layer.dataType, base + 5) + 0.5f);

        const float threshold = static_cast<size_t>(classId) < detectionParams.perClassThreshold.size()
                                     ? detectionParams.perClassThreshold[classId]
                                     : defaultThresh;
        if (confidence < threshold) {
            continue;
        }

        const float left = std::max(0.0f, std::min(x1, netW));
        const float top = std::max(0.0f, std::min(y1, netH));
        const float width = std::min(x2, netW) - left;
        const float height = std::min(y2, netH) - top;

        if (width <= 0.0f || height <= 0.0f) {
            continue;
        }

        NvDsInferObjectDetectionInfo obj;
        obj.classId = static_cast<unsigned int>(classId);
        obj.detectionConfidence = confidence;
        obj.left = left;
        obj.top = top;
        obj.width = width;
        obj.height = height;
        objectList.push_back(obj);
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYolo26);
