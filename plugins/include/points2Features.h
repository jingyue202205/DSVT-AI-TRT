#ifndef _POINTS_2_FEATURES_H_
#define _POINTS_2_FEATURES_H_

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
    class Points2FeaturesPlugin: public nvinfer1::IPluginV2DynamicExt
    {
        public:
            Points2FeaturesPlugin() = delete;
            Points2FeaturesPlugin(int max_points_num,int max_points_num_voxel_filter,int max_pillars_num,  int point_feature_num, int feature_num, int max_num_points_per_voxel, float x_min,
              float x_max, float y_min, float y_max, float z_min, float z_max,
              float voxel_x, float voxel_y, float voxel_z, int grid_x, int grid_y, int grid_z);
            Points2FeaturesPlugin(const void* data, size_t length);
    
            // ~Points2FeaturesPlugin() override;

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
              // Shape Num for *input*
            int  max_points_num_; 
            int max_points_num_voxel_filter_;
            int max_pillars_num_;
            int point_feature_num_;  // 4
            int feature_num_;  // 10
            int max_num_points_per_voxel_; // 32
            float min_x_range_;
            float max_x_range_;
            float min_y_range_;
            float max_y_range_;
            float min_z_range_;
            float max_z_range_;
            float voxel_x_size_;  // 0.32
            float voxel_y_size_;  // 0.32
            float voxel_z_size_;  // 0.8
            int grid_size_x_;
            int grid_size_y_;
            int grid_size_z_;
};

class Points2FeaturesPluginCreator : public nvinfer1::IPluginCreator
{
    public:
        Points2FeaturesPluginCreator();
        ~Points2FeaturesPluginCreator() override;
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
REGISTER_TENSORRT_PLUGIN(Points2FeaturesPluginCreator);
};

#endif 
