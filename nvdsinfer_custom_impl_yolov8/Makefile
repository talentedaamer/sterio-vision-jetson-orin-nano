# Builds libnvdsinfer_custom_impl_Yolo.so -- the custom bbox parser gst-nvinfer
# dlopen()s for the YOLOv8n engine (see ../configs/pgie_yolov8n_config.txt).
#
# Build ON THE JETSON (aarch64), not on the dev machine:
#   cd nvdsinfer_custom_impl_yolov8 && make
#
# Override NVDS_VERSION/CUDA_VER if your install paths differ, e.g.:
#   make NVDS_VERSION=7.1 CUDA_VER=12.6

NVDS_VERSION ?= 7.1
CUDA_VER ?= 12.6
DS_ROOT ?= /opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)

CXX := g++
SRCS := nvdsinfer_yolov8_parser.cpp
TARGET_LIB := libnvdsinfer_custom_impl_Yolo.so

CXXFLAGS := -Wall -std=c++14 -shared -fPIC -O2
CXXFLAGS += -I$(DS_ROOT)/sources/includes
CXXFLAGS += -I/usr/local/cuda-$(CUDA_VER)/include

all: $(TARGET_LIB)

$(TARGET_LIB): $(SRCS)
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm -f $(TARGET_LIB)

.PHONY: all clean
