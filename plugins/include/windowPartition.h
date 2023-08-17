#ifndef _WINDOW_PARTITION_H_
#define _WINDOW_PARTITION_H_

#include "NvInferPlugin.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
// #include <cmath>
#include "params.h"

using namespace std;

namespace nvinfer1
{
    class WindowPartitionPlugin: public nvinfer1::IPluginV2DynamicExt
    {
        public:
            WindowPartitionPlugin() = delete;
            WindowPartitionPlugin(int sparse_shape_x, int sparse_shape_y, int sparse_shape_z, 
                                            int win_shape_x, int win_shape_y, int win_shape_z, int shift_x, int shift_y, int shift_z,
                                           int max_win_num, int max_voxel_num_per_win);
            WindowPartitionPlugin(const void* data, size_t length);
    
            // ~WindowPartitionPlugin() override;

            // IPluginV2DynamicExt Methods
            nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
            nvinfer1::DimsExprs getOutputDimensions(int outputIndex, 
                const nvinfer1::DimsExprs* inputs, int nbInputs,
                nvinfer1::IExprBuilder& exprBuilder) noexcept override;
            bool supportsFormatCombination(
                int pos, const nvinfer1::PluginTensorDesc* inOut, 
                int nbInputs, int nbOutputs) noexcept override;
            void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
            size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
            int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, 
                const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs, 
                void* workspace, cudaStream_t stream) noexcept override;
            // IPluginV2Ext Methods
            nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, 
                int nbInputs) const noexcept override;
            // IPluginV2 Methods
            const char* getPluginType() const noexcept override;
            const char* getPluginVersion() const noexcept override;
            int getNbOutputs() const noexcept override;
            int initialize() noexcept override;
            void terminate() noexcept override;
            size_t getSerializationSize() const noexcept override;
            void serialize(void* buffer) const noexcept override;
            void destroy() noexcept override;
            void setPluginNamespace(const char* pluginNamespace) noexcept override;
            const char* getPluginNamespace() const noexcept override;
        private:
            std::string mNamespace;
            int sparse_shape_x_;
            int sparse_shape_y_;
            int sparse_shape_z_;
            int win_shape_x_;
            int win_shape_y_;
            int win_shape_z_;
            int shift_x_;
            int shift_y_;
            int shift_z_;
            int max_win_num_;
            int max_voxel_num_per_win_;
};

class WindowPartitionPluginCreator : public nvinfer1::IPluginCreator
{
    public:
        WindowPartitionPluginCreator();
        ~WindowPartitionPluginCreator() override;
        const char* getPluginName() const noexcept override;
        const char* getPluginVersion() const noexcept override;
        const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
        nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
        nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
        void setPluginNamespace(const char* pluginNamespace) noexcept override;
        const char* getPluginNamespace() const noexcept override;
    private:
        static nvinfer1::PluginFieldCollection mFC;
        static std::vector<nvinfer1::PluginField> mPluginAttributes;
        std::string mNamespace;
        // int *max_num_points_per_voxel_ptr = nullptr;
        // int *max_voxels_ptr = nullptr;
        // float *voxel_size_ptr = nullptr;
        // float *point_cloud_range_ptr = nullptr;
        // int *voxel_feature_num_ptr = nullptr;
};
REGISTER_TENSORRT_PLUGIN(WindowPartitionPluginCreator);
};

#endif 
