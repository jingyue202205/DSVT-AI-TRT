#ifndef _GELU_H_
#define _GELU_H_

#include "NvInferPlugin.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include "params.h"

using namespace std;

namespace nvinfer1
{
    class GeluPlugin: public nvinfer1::IPluginV2DynamicExt
    {
        public:
            GeluPlugin() = delete;
            // GeluPlugin(int in_channel, int out_channel,int max_voxels, int feature_num, int out_shape_z,int out_shape_y,int out_shape_x,
            //                         int spatial_shape_z, int spatial_shape_y, int spatial_shape_x,int ksize,
            //                         int stride, int padding, int dilation, int out_padding, const std::vector<float> & weights);
            GeluPlugin(int max_pillars_num, int channel_num);
            GeluPlugin(const void* data, size_t length);
    
            ~GeluPlugin() override;

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
              
            int channel_num_;  //192
            int max_pillars_num_; // 7000
};

class GeluPluginCreator : public nvinfer1::IPluginCreator
{
    public:
        GeluPluginCreator();
        ~GeluPluginCreator() override;
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
        // int *out_shape = nullptr;
        // int *spatial_shape = nullptr;
        // int *ksize = nullptr;
        // int *stride = nullptr;
        // int *padding = nullptr;
        // int *dilation = nullptr;
        // int *out_padding = nullptr;
        // float *d_weights = nullptr;  //device gpu
        // std::vector<float> h_weights; // host

};
REGISTER_TENSORRT_PLUGIN(GeluPluginCreator);
};

#endif 
