#include "gelu.h"


#include <fstream>
#include <iostream>
#include <iomanip> //设置输出格式
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "params.h"
#include <time.h>
#include <chrono>
#include <cmath>
#include <string>
#include <string.h>

using namespace std::chrono;
using std::string;

using namespace nvinfer1;
using nvinfer1::GeluPlugin;
using nvinfer1::GeluPluginCreator;
using namespace std;

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)



#define CUDA_MEM_ALIGN 256

static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"GeluPlugin"};

// Static class fields initialization
PluginFieldCollection GeluPluginCreator::mFC{};
std::vector<PluginField> GeluPluginCreator::mPluginAttributes;

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

// Mimic np.round as in voxel generator in spconv implementation
int np_round(float x) {
  // half way round to nearest-even
  int x2 = int(x * 2.0f);
  if(x != int(x) && x2 == x * 2.0f) {
    return int(x / 2.0f + 0.5f) * 2;
  }
  return int(x + 0.5f);
}

// ALIGNPTR
int8_t* alignPtr(int8_t* ptr, uintptr_t to)
{
    uintptr_t addr = (uintptr_t) ptr;
    if (addr % to)
    {
        addr += to - addr % to;
    }
    return (int8_t*) addr;
}

// NEXTWORKSPACEPTR
int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t) ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t*) addr, CUDA_MEM_ALIGN);
}

// CALCULATE TOTAL WORKSPACE SIZE
size_t calculateTotalWorkspaceSize(size_t* workspaces, int count)
{
    size_t total = 0;
    for (int i = 0; i < count; i++)
    {
        total += workspaces[i];
        if (workspaces[i] % CUDA_MEM_ALIGN)
        {
            total += CUDA_MEM_ALIGN - (workspaces[i] % CUDA_MEM_ALIGN);
        }
    }
    return total;
}

GeluPlugin::GeluPlugin(int max_pillars_num,int channel_num) :
                                        max_pillars_num_(max_pillars_num),channel_num_(channel_num)

{

}

GeluPlugin::GeluPlugin(const void* data, size_t length)
{   
    const char* d = reinterpret_cast<const char*>(data);
    max_pillars_num_ = readFromBuffer<int>(d);
    channel_num_ = readFromBuffer<int>(d);
}

IPluginV2DynamicExt* GeluPlugin::clone() const noexcept
{
    // std::cout << "clone    start" << std::endl;
    auto* plugin = new GeluPlugin(max_pillars_num_,channel_num_);
    plugin->setPluginNamespace(mNamespace.c_str());
    // std::cout << "clone   end" << std::endl;
    return plugin;
}

nvinfer1::DimsExprs GeluPlugin::getOutputDimensions(
int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    auto batch_size = inputs[0].d[0];
    // auto batch_size = 1;
    // std::cout  << inputs[0].nbDims << " " << inputs[0].d[0]->getConstantValue() << " " << inputs[0].d[1]->getConstantValue() << " " << inputs[0].d[2]->getConstantValue() << std::endl;
    // std::cout  << inputs[1].nbDims << " " << inputs[1].d[0]->getConstantValue() << std::endl;
    if (outputIndex == 0)
    {
        nvinfer1::DimsExprs dim0{};
        dim0.nbDims = 3;
        dim0.d[0] = batch_size;
        dim0.d[1] = exprBuilder.constant(max_pillars_num_);
        dim0.d[2] = exprBuilder.constant(channel_num_);
        return dim0; // voxel_featues 1 7000 192
    }
   
}

bool GeluPlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       // voxelfeatures ---dim: 1 7000 192
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)       // valid voxel_num   
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)       // out_voxel_feature, dim: 1 x 7000 x 192
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    
    return false;
}

void GeluPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t GeluPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return  0;
}


__global__ void Gelu_kernel(
          float* input_voxel_features, 
            unsigned int* voxel_num, float* output_voxel_features,
           int channel_num)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= (*voxel_num)*channel_num)  return;
    float x=  *(input_voxel_features+idx);
     *(output_voxel_features+idx) = (GELU_A + GELU_A * tanh(x*(GELU_C*x*x+GELU_B))) * x;
}
   
  
cudaError_t GeluKernel_launch(float* input_voxel_features, 
            unsigned int* voxel_num, float* output_voxel_features, int channel_num,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_PILLARS_NUM*SET_ATTENTION_0_0_GELU_OUT_CHANNEL+threadNum-1)/threadNum);
  dim3 threads(threadNum);
            Gelu_kernel<<<blocks, threads, 0, stream>>>(input_voxel_features,voxel_num,output_voxel_features,channel_num
        );
  cudaError_t err = cudaGetLastError();
  return err;
}

int GeluPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // std::cout << "enqueue start" << std::endl;
    int batchSize = inputDesc[0].dims.d[0];
    // int maxNumPoints = inputDesc[0].dims.d[1];
    //TRT-input
    // std::cout << "batch_size: " << batchSize << std::endl;
    float *input_voxel_features = const_cast<float *>((const float*)inputs[0]);  // 1   7000 192
    unsigned int * voxel_num = const_cast<unsigned int *>((const unsigned int *)inputs[1]);  // valid voxel_num
    //TRT-output
    float *output_voxel_features = (float *)(outputs[0]);  // 1 * 7000 * 192
    
    // FOR tensorrt output
    unsigned int output_voxel_features_size = batchSize * max_pillars_num_  * channel_num_ * sizeof(float);
    // unsigned int output_coords_size = batchSize * max_voxels_ * 4 * sizeof(unsigned int);
    // unsigned int voxel_num_data_size = batchSize * sizeof(unsigned int);
    // unsigned int params_data_size = batchSize * sizeof(unsigned int);
    checkCudaErrors(cudaMemsetAsync(output_voxel_features, 0, output_voxel_features_size, stream));
   
    checkCudaErrors(GeluKernel_launch( input_voxel_features,voxel_num,output_voxel_features,
                                                    channel_num_,  stream));
    return 0;
}

nvinfer1::DataType GeluPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // if(index == 0)
    return inputTypes[0];
    // return inputTypes[1];
}

const char* GeluPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* GeluPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int GeluPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int GeluPlugin::initialize() noexcept
{   
    return 0;
}

void GeluPlugin::terminate() noexcept
{

}

GeluPlugin::~GeluPlugin()
{
    terminate();
}

size_t GeluPlugin::getSerializationSize() const noexcept
{
    return  2 * sizeof(int);
}

void GeluPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);

    writeToBuffer<int>(d, max_pillars_num_);
    writeToBuffer<int>(d, channel_num_);
}

void GeluPlugin::destroy() noexcept
{
    delete this;
}

void GeluPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* GeluPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// __device__ float Logist(float data){ return 1.0f / (1.0f + expf(-data)); };

GeluPluginCreator::GeluPluginCreator()
{
    mPluginAttributes.clear();

    mPluginAttributes.emplace_back(PluginField("max_pillars_num", nullptr, PluginFieldType::kINT32, 1));  //7000
    mPluginAttributes.emplace_back(PluginField("channel_num", nullptr, PluginFieldType::kINT32, 1));  //192
    
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GeluPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* GeluPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* GeluPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GeluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    // std::cout << "createplugin start <<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    // std::cout << nbFields << std::endl;

    int max_pillars_num = 0;
    int channel_num = 0;
  
    // std::cout << fields[0].name << std::endl;
    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name =  fields[i].name;
        // std::cout << " createplugn <<" << attr_name << std::endl;
        if (!strcmp(attr_name, "max_pillars_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_pillars_num = d[0];
            // std::cout << "in_channel <<<<" << in_channel << std::endl;
        }
        else if (!strcmp(attr_name, "channel_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            channel_num = d[0];
            // std::cout << "out_channel <<<<" << out_channel << std::endl;
        }
    }
    // std::cout << max_voxels << " " << in_channel << " " <<out_channel << " " << feature_num << " " << out_shape_x << " "
    // << out_shape_y << " "<< out_shape_z << " " << spatial_shape_x << " " << spatial_shape_y << " " << spatial_shape_z << " "
    // << ksize << " " << stride << " " << padding << " " << dilation << " " << out_padding << " " << std::endl;
    
    IPluginV2DynamicExt* plugin = new GeluPlugin(max_pillars_num,channel_num);
    return plugin;
}

IPluginV2* GeluPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new GeluPlugin(serialData, serialLength);
}

void GeluPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* GeluPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

GeluPluginCreator::~GeluPluginCreator()
{

}