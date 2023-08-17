#ifndef _MAP_SET_FEATURE_2_VOXEL_H_
#define _MAP_SET_FEATURE_2_VOXEL_H_

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
    class MapSetFeature2VoxelPlugin: public nvinfer1::IPluginV2DynamicExt
    {
        public:
            MapSetFeature2VoxelPlugin() = delete;
            MapSetFeature2VoxelPlugin(int voxel_num_set, int max_win_num, int channel_num,int max_pillars_num,int axis_id);
            MapSetFeature2VoxelPlugin(const void* data, size_t length);
    
            // ~MapSetFeature2VoxelPlugin() override;

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
             int voxel_num_set_;  //36
            int max_win_num_;   //3200
            int channel_num_;  //192
            int max_pillars_num_; // 7000
            int axis_id_;  // 0  y-axis   1   x-axis
};

class MapSetFeature2VoxelPluginCreator : public nvinfer1::IPluginCreator
{
    public:
        MapSetFeature2VoxelPluginCreator();
        ~MapSetFeature2VoxelPluginCreator() override;
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
REGISTER_TENSORRT_PLUGIN(MapSetFeature2VoxelPluginCreator);
};

#endif 
