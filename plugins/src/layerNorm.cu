#include "layerNorm.h"


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
using nvinfer1::LayerNormPlugin;
using nvinfer1::LayerNormPluginCreator;
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
static const char* PLUGIN_NAME{"LayerNormPlugin"};

// Static class fields initialization
PluginFieldCollection LayerNormPluginCreator::mFC{};
std::vector<PluginField> LayerNormPluginCreator::mPluginAttributes;

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

LayerNormPlugin::LayerNormPlugin(int max_pillars_num,int channel_num, int weights_size, float eps, nvinfer1::Weights const& weights, nvinfer1::Weights const& bias) :
                                        max_pillars_num_(max_pillars_num),channel_num_(channel_num),weights_size_(weights_size),eps_(eps)

{
    // std::cout << "构造函数 start" << std::endl;
    weights_data_ = (float*)malloc(weights_size_*sizeof(float));
    const float* temp_weight_values = (const float*)weights.values;

    bias_data_ = (float*)malloc(weights_size_*sizeof(float));
    const float* temp_bias_values = (const float*)bias.values;

    for(int i = 0;i < weights_size_;i++)
    {
        weights_data_[i] = temp_weight_values[i];
        bias_data_[i] = temp_bias_values[i];
    }
    weights_.count = weights.count;
    weights_.values = weights_data_;

    bias_.count = bias.count;
    bias_.values = bias_data_;

   
    //  cudamalloc for conv
    checkCudaErrors(cudaMalloc(&weights_dev_,sizeof(float)*weights_size_));
    checkCudaErrors(cudaMalloc(&bias_dev_,sizeof(float)*weights_size_));

    // copy to gpu
    checkCudaErrors(cudaMemcpy(weights_dev_,weights_data_,sizeof(float)*weights_size_,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(bias_dev_,bias_data_,sizeof(float)*weights_size_,cudaMemcpyHostToDevice));

   
}

LayerNormPlugin::LayerNormPlugin(const void* data, size_t length)
{   
    const char* d = reinterpret_cast<const char*>(data);
    max_pillars_num_ = readFromBuffer<int>(d);
    channel_num_ = readFromBuffer<int>(d);
    weights_size_ = readFromBuffer<int>(d);
    eps_ =  readFromBuffer<float>(d);

    weights_.count = weights_size_;
    weights_data_ = (float *)malloc(weights_size_*sizeof(float));
    for(int i=0;i < weights_size_; i++) 
    {
        weights_data_[i] = readFromBuffer<float>(d);
    }
    weights_.values = weights_data_;

    bias_.count = weights_size_;
    bias_data_ = (float *)malloc(weights_size_*sizeof(float));
     for(int i=0;i < weights_size_; i++) 
    {
        bias_data_[i] = readFromBuffer<float>(d);
    }
    bias_.values = bias_data_;
    // std::cout << "构造函数2 end" << std::endl;
    // std::cout << in_channel_ << "," << out_channel_ << "," << max_voxels_ << "," << feature_num_ << "," << out_shape_x_ << "," << 
    //             out_shape_y_ << "," << out_shape_z_ << "," << spatial_shape_x_ << "," << spatial_shape_y_ << "," << spatial_shape_z_ << 
    //             "," << ksize_ << "," << stride_ << "," << padding_ << "," << dilation_ << "," << out_padding_ << "," << weights_size_ << std::endl;
}

IPluginV2DynamicExt* LayerNormPlugin::clone() const noexcept
{
    // std::cout << "clone    start" << std::endl;
    auto* plugin = new LayerNormPlugin(max_pillars_num_,channel_num_, weights_size_,eps_, weights_,bias_);
    plugin->setPluginNamespace(mNamespace.c_str());
    // std::cout << "clone   end" << std::endl;
    return plugin;
}

nvinfer1::DimsExprs LayerNormPlugin::getOutputDimensions(
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

bool LayerNormPlugin::supportsFormatCombination(
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

void LayerNormPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t LayerNormPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
     int batchSize = inputs[0].dims.d[0];
   // workspace buffer
    size_t mean_value_size = batchSize * max_pillars_num_  * sizeof(float);   // 
    size_t var_value_size = batchSize * max_pillars_num_  * sizeof(float);   // 

    
    size_t workspaces[2];
    workspaces[0] = mean_value_size;
    workspaces[1] = var_value_size;
    size_t total_workspace = calculateTotalWorkspaceSize(workspaces, 2);

    return  total_workspace;
}


__global__ void layerNorm_kernel(
          float* input_voxel_features, 
            unsigned int* voxel_num, float* output_voxel_features,
            float* mean_value_data, float* var_value_data,
            float * weights, float *bias, int channel_num,float eps)
{
    unsigned int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxel_idx >= *voxel_num)  return;
    float mean = *(mean_value_data+voxel_idx);
    float var = *(var_value_data + voxel_idx);
    float temp = 0.0;
    for(int j=0;j<channel_num;j++)
    {
        temp = ((*(input_voxel_features+voxel_idx*channel_num+j)) - mean) / sqrt(var+eps);
        temp *= weights[j];
        temp += bias[j];
        *(output_voxel_features+voxel_idx*channel_num+j) = temp;
    }
}
  
cudaError_t layerNormKernel_launch(float* input_voxel_features, 
            unsigned int* voxel_num, float* output_voxel_features,float* mean_value_data,float *var_value_data, float *weights, float *bias, int channel_num,float eps,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_PILLARS_NUM+threadNum-1)/threadNum);
  dim3 threads(threadNum);
            layerNorm_kernel<<<blocks, threads, 0, stream>>>(input_voxel_features,voxel_num,output_voxel_features,mean_value_data,
            var_value_data, weights,bias,channel_num, eps
        );
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void getMean_kernel(
          float* input_voxel_features, 
            unsigned int* voxel_num, float* mean_value_data, int channel_num)
{
    unsigned int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxel_idx >= *voxel_num)  return;
    float avg = 0.0;
    for(int j=0;j<channel_num;j++)
    {
        avg += *(input_voxel_features+voxel_idx*channel_num+j);
    }
    *(mean_value_data+voxel_idx) = avg/channel_num;
}

cudaError_t getMean_launch(float* input_voxel_features, 
            unsigned int* voxel_num, float* mean_value_data, int channel_num,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_PILLARS_NUM+threadNum-1)/threadNum);
  dim3 threads(threadNum);
            getMean_kernel<<<blocks, threads, 0, stream>>>(input_voxel_features,voxel_num,mean_value_data,channel_num
        );
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void getVar_kernel(
          float* input_voxel_features, 
            unsigned int* voxel_num, float* mean_value_data, float* var_value_data,  int channel_num)
{
    unsigned int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxel_idx >= *voxel_num)  return;
    float var = 0.0;
    for(int j=0;j<channel_num;j++)
    {
        var += (*(input_voxel_features+voxel_idx*channel_num+j) - *(mean_value_data+voxel_idx)) * (*(input_voxel_features+voxel_idx*channel_num+j) - *(mean_value_data+voxel_idx));
    }
    *(var_value_data+voxel_idx) = var/channel_num;
}

cudaError_t getVar_launch(float* input_voxel_features, 
            unsigned int* voxel_num, float* mean_value_data, float* var_value_data, int channel_num,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_PILLARS_NUM+threadNum-1)/threadNum);
  dim3 threads(threadNum);
            getVar_kernel<<<blocks, threads, 0, stream>>>(input_voxel_features,voxel_num,mean_value_data,var_value_data, channel_num
        );
  cudaError_t err = cudaGetLastError();
  return err;
}




int LayerNormPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
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
  


   // workspace buffer
    size_t mean_value_size = batchSize * max_pillars_num_  * sizeof(float);   // 
    size_t var_value_size = batchSize * max_pillars_num_  * sizeof(float);   // 

    size_t workspaces[2];
    workspaces[0] = mean_value_size;
    workspaces[1] = var_value_size;
    size_t total_workspace = calculateTotalWorkspaceSize(workspaces, 2);

    float* mean_value_data = static_cast<float *>(workspace);
    float *var_value_data = reinterpret_cast<float*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(mean_value_data), mean_value_size));
    
    // // Initialize workspace memory
    checkCudaErrors(cudaMemsetAsync(mean_value_data, 0, total_workspace, stream)); // total_workspace
     
    
    // FOR tensorrt output
    unsigned int output_voxel_features_size = batchSize * max_pillars_num_  * channel_num_ * sizeof(float);
    // unsigned int output_coords_size = batchSize * max_voxels_ * 4 * sizeof(unsigned int);
    // unsigned int voxel_num_data_size = batchSize * sizeof(unsigned int);
    // unsigned int params_data_size = batchSize * sizeof(unsigned int);
    checkCudaErrors(cudaMemsetAsync(output_voxel_features, 0, output_voxel_features_size, stream));

    checkCudaErrors(getMean_launch( input_voxel_features,voxel_num,mean_value_data,channel_num_, stream));
    checkCudaErrors(getVar_launch( input_voxel_features,voxel_num,mean_value_data,var_value_data,channel_num_, stream));
    checkCudaErrors(layerNormKernel_launch( input_voxel_features,voxel_num,output_voxel_features,mean_value_data,var_value_data,
                                                    weights_dev_,bias_dev_,channel_num_, eps_, stream));
    return 0;
}

nvinfer1::DataType LayerNormPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // if(index == 0)
    return inputTypes[0];
    // return inputTypes[1];
}

const char* LayerNormPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* LayerNormPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int LayerNormPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int LayerNormPlugin::initialize() noexcept
{   
    return 0;
}

void LayerNormPlugin::terminate() noexcept
{
    // // cudafree
    cudaFree(weights_dev_);
    cudaFree(bias_dev_);
    // //c free
    // free(weights_data_);
}

LayerNormPlugin::~LayerNormPlugin()
{
    terminate();
}

size_t LayerNormPlugin::getSerializationSize() const noexcept
{
    return  (channel_num_*2 +1)* sizeof(float) + 3 * sizeof(int);
}

void LayerNormPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);

    writeToBuffer<int>(d, max_pillars_num_);
    writeToBuffer<int>(d, channel_num_);
    writeToBuffer<int>(d, weights_size_);
    writeToBuffer<float>(d, eps_);
  
    const float * weights_data = (const float*)weights_.values;
    for(int i=0; i < weights_size_; i++)
    {
        writeToBuffer<float>(d,weights_data[i]);
    }
    
    const float* bias_data = (const float*)bias_.values;
    for(int i=0; i < weights_size_; i++)
    {
        writeToBuffer<float>(d,bias_data[i]);
    }
}

void LayerNormPlugin::destroy() noexcept
{
    delete this;
}

void LayerNormPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* LayerNormPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// __device__ float Logist(float data){ return 1.0f / (1.0f + expf(-data)); };

LayerNormPluginCreator::LayerNormPluginCreator()
{
    mPluginAttributes.clear();

    mPluginAttributes.emplace_back(PluginField("max_pillars_num", nullptr, PluginFieldType::kINT32, 1));  //7000
    mPluginAttributes.emplace_back(PluginField("channel_num", nullptr, PluginFieldType::kINT32, 1));  //192
    mPluginAttributes.emplace_back(PluginField("weights_size", nullptr, PluginFieldType::kINT32, 1));  //
    mPluginAttributes.emplace_back(PluginField("pes", nullptr, PluginFieldType::kFLOAT32, 1));  //

    mPluginAttributes.emplace_back(PluginField("weights", nullptr, PluginFieldType::kFLOAT32, 1));  // 192
    mPluginAttributes.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));  // 192



    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* LayerNormPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* LayerNormPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* LayerNormPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* LayerNormPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    // std::cout << "createplugin start <<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    // std::cout << nbFields << std::endl;

    int max_pillars_num = 0;
    int channel_num = 0;
    int weights_size = 0;
    float eps = 0.0;
    const float *weights;
    const float*bias;
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
        else if (!strcmp(attr_name, "weights_size"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            weights_size = d[0];
        }
          else if (!strcmp(attr_name, "eps"))
        {
            const float* d = static_cast<const float*>(fields[i].data);
            eps = d[0];
        }
        else if (!strcmp(attr_name, "weights"))
        {
            const float * d = static_cast<const float*>(fields[i].data);
            weights = d;
        }
            else if (!strcmp(attr_name, "bias"))
        {
            const float * d = static_cast<const float*>(fields[i].data);
            bias = d;
        }
    }
    nvinfer1::Weights wt{DataType::kFLOAT, nullptr, 0};
    wt.count = weights_size;
    wt.values = weights;

      nvinfer1::Weights bis{DataType::kFLOAT, nullptr, 0};
    bis.count = weights_size;
    bis.values = bias;
    // std::cout << max_voxels << " " << in_channel << " " <<out_channel << " " << feature_num << " " << out_shape_x << " "
    // << out_shape_y << " "<< out_shape_z << " " << spatial_shape_x << " " << spatial_shape_y << " " << spatial_shape_z << " "
    // << ksize << " " << stride << " " << padding << " " << dilation << " " << out_padding << " " << std::endl;
    
    IPluginV2DynamicExt* plugin = new LayerNormPlugin(max_pillars_num,channel_num,weights_size,eps, wt,bis);
    return plugin;
}

IPluginV2* LayerNormPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new LayerNormPlugin(serialData, serialLength);
}

void LayerNormPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* LayerNormPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

LayerNormPluginCreator::~LayerNormPluginCreator()
{

}