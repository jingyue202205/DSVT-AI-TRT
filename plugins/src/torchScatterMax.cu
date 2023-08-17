#include "torchScatterMax.h"

using namespace nvinfer1;
using nvinfer1::TorchScatterMaxPlugin;
using nvinfer1::TorchScatterMaxPluginCreator;
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
static const char* PLUGIN_NAME{"TorchScatterMaxPlugin"};

// Static class fields initialization
PluginFieldCollection TorchScatterMaxPluginCreator::mFC{};
std::vector<PluginField> TorchScatterMaxPluginCreator::mPluginAttributes;

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

// create the plugin at runtime from a byte stream
TorchScatterMaxPlugin::TorchScatterMaxPlugin(int max_points_num,
             int max_pillars_num, int feature_num
) : max_points_num_(max_points_num),max_pillars_num_(max_pillars_num), feature_num_(feature_num)
{
}

TorchScatterMaxPlugin::TorchScatterMaxPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    max_points_num_ = readFromBuffer<int>(d);
    max_pillars_num_ = readFromBuffer<int>(d);
    feature_num_ = readFromBuffer<int>(d);
}

IPluginV2DynamicExt* TorchScatterMaxPlugin::clone() const noexcept
{
    auto* plugin = new TorchScatterMaxPlugin(max_points_num_,max_pillars_num_,  feature_num_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs TorchScatterMaxPlugin::getOutputDimensions(
int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
// 5 returns 
{
    // assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    auto batch_size = inputs[0].d[0];
    // auto batch_size = 1;
    // std::cout  << inputs[0].nbDims << " " << inputs[0].d[0]->getConstantValue() << " " << inputs[0].d[1]->getConstantValue() << " " << inputs[0].d[2]->getConstantValue() << std::endl;
    // std::cout  << inputs[1].nbDims << " " << inputs[1].d[0]->getConstantValue() << std::endl;
    if (outputIndex == 0)
    {
        // std::cout << "batch_size: " << batch_size->getConstantValue() << " voxel_num: " << voxelNum_ << " featurennum_: " << featureNum_ << std::endl;
        nvinfer1::DimsExprs dim0{};
        dim0.nbDims = 3;
        dim0.d[0] = batch_size;
        dim0.d[1] = exprBuilder.constant(max_points_num_);
        dim0.d[2] = exprBuilder.constant(feature_num_);
        return dim0; // features 1 50000 96  max
    }
       if (outputIndex == 1)
    {
        // std::cout << "batch_size: " << batch_size->getConstantValue() << " voxel_num: " << voxelNum_ << " featurennum_: " << featureNum_ << std::endl;
        nvinfer1::DimsExprs dim1{};
        dim1.nbDims = 3;
        dim1.d[0] = batch_size;
        dim1.d[1] = exprBuilder.constant(max_pillars_num_);
        dim1.d[2] = exprBuilder.constant(feature_num_);
        return dim1; // features 1 10000 96
    }
}

bool TorchScatterMaxPlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // PLUGIN_ASSERT(nbInputs == 2);
    // PLUGIN_ASSERT(nbOutputs == 2);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       //features
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)       // point_index_in_voxel
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)       // point_num_in_voxel
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)       // voxel_num 
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 4)       // feature 1 max_points 96
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
     if (pos == 5)       // feature 1 max_voxels 96
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    return false;
}

void TorchScatterMaxPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t TorchScatterMaxPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

__global__ void generateMax_kernel(
      float* point_features_data,unsigned int* point_index_in_voxel_data,
                                         unsigned int * point_num_in_voxel_data, 
                                unsigned int *voxel_num_data, float* max_point_features_data, 
                                float* max_voxel_features_data,int feature_num)
{
    // printf("point_size:%d\n",*points_size);
    unsigned int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxel_idx >= *voxel_num_data) return;

    // init max_array
    // float *array = (float*)malloc(feature_num*sizeof(float));
    float  array[ARRAY_LEN];
    for(int i=0;i<ARRAY_LEN;i++)
    {
        array[i] = -1000000.0;
    }

    unsigned int *point_index_in_voxel_addr  = point_index_in_voxel_data + voxel_idx * POINTS_NUM_PER_VOXEL;
    unsigned int point_num = *(point_num_in_voxel_data+voxel_idx);

        // get max array
        for(int i=0;i<point_num;i++)
        {
            for(int k=0;k<feature_num;k++)
            {
                if (point_features_data[(*(point_index_in_voxel_addr+i))*feature_num+k]>array[k])
                {
                    // printf("point_size:%f\n",point_features_data[(*(point_index_in_voxel_addr+i))*feature_num+k]);
                //       if(*(point_index_in_voxel_addr+i) == 0)
                // {
                //     printf("point_size:%f\n",point_features_data[(*(point_index_in_voxel_addr+i))*feature_num+k]);
                // }
                    array[k] = point_features_data[(*(point_index_in_voxel_addr+i))*feature_num+k];
                }
            }
        }

        // save to max_voxel_features_data
        for(int i=0; i<feature_num;i++)
        {
            max_voxel_features_data[voxel_idx*feature_num+i] = array[i];
        }

        // save to max_point_features_data
        for(int i=0;i<point_num;i++)
        {
        for(int k=0;k<feature_num;k++)
        {
                int index_i = (*(point_index_in_voxel_addr+i))*feature_num+k;
                // if(*(point_index_in_voxel_addr+i) == 0)
                // {
                //     printf("point_size:%f\n",point_features_data[(*(point_index_in_voxel_addr+i))*feature_num+k]);
                // }
                max_point_features_data[index_i] = array[k];
        }
        }
        // free(array);
}


cudaError_t generateMax_launch(float* point_features_data,unsigned int* point_index_in_voxel_data,
                                         unsigned int * point_num_in_voxel_data, 
                                unsigned int *voxel_num_data, float* max_point_features_data, float* max_voxel_features_data, int feature_num,
                            cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_PILLARS_NUM+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  generateMax_kernel<<<blocks, threads, 0, stream>>>
       (point_features_data,
        point_index_in_voxel_data, 
        point_num_in_voxel_data,voxel_num_data,max_point_features_data,max_voxel_features_data,feature_num);
  cudaError_t err = cudaGetLastError();
  return err;
}


int TorchScatterMaxPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];
    // int maxNumPoints = inputDesc[0].dims.d[1];

    //TRT-input
    // std::cout << "voxelgenerator batch_size: " << batchSize << std::endl;
    float * point_features_data = const_cast<float *>((const float *)inputs[0]);
    unsigned int* point_index_in_voxel_data = const_cast<unsigned int *>((const unsigned int *)inputs[1]);
    unsigned int* point_num_in_voxel_data = const_cast<unsigned int *>((const unsigned int *)inputs[2]);
    unsigned int* voxel_num_data = const_cast<unsigned int *>((const unsigned int *)inputs[3]);

    //TRT-output
    float *max_point_features_data = (float *)(outputs[0]);  // 1 max_point_num 96
    float  *max_voxel_features_data = (float*)(outputs[1]);  // 1 max_voxel_num 96
  
    unsigned int max_point_features_data_size = batchSize * max_points_num_  * feature_num_ * sizeof(float);
    unsigned int max_voxel_features_data_size = batchSize * max_pillars_num_ * feature_num_ * sizeof(float);
    
    checkCudaErrors(cudaMemsetAsync(max_point_features_data, 0, max_point_features_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(max_voxel_features_data,0,max_voxel_features_data_size,stream));


    checkCudaErrors(generateMax_launch(
         point_features_data,point_index_in_voxel_data,point_num_in_voxel_data, voxel_num_data,max_point_features_data,max_voxel_features_data, 
         feature_num_, stream));
    return 0;
}

nvinfer1::DataType TorchScatterMaxPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

const char* TorchScatterMaxPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* TorchScatterMaxPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int TorchScatterMaxPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int TorchScatterMaxPlugin::initialize() noexcept
{
    return 0;
}

void TorchScatterMaxPlugin::terminate() noexcept
{
}

size_t TorchScatterMaxPlugin::getSerializationSize() const noexcept
{
    return  3 * sizeof(int);
}

void TorchScatterMaxPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, max_points_num_);
    writeToBuffer<int>(d, max_pillars_num_);
    writeToBuffer<int>(d, feature_num_);
}

void TorchScatterMaxPlugin::destroy() noexcept
{
    delete this;
}

void TorchScatterMaxPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* TorchScatterMaxPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}


TorchScatterMaxPluginCreator::TorchScatterMaxPluginCreator()
{
    
    mPluginAttributes.clear();

    // std::cout <<  *max_num_points_per_voxel_ptr << std::endl;
    mPluginAttributes.emplace_back(PluginField("max_points_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_pillars_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("feature_num", nullptr, PluginFieldType::kINT32, 1));
    
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* TorchScatterMaxPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* TorchScatterMaxPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* TorchScatterMaxPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* TorchScatterMaxPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int max_points_num = 0;
    int max_pillars_num = 0;
    int feature_num = 0;
   
    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "max_points_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_points_num = d[0];
        }
        else if (!strcmp(attr_name, "max_pillars_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_pillars_num = d[0];
        }
            else if (!strcmp(attr_name, "feature_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            feature_num = d[0];
        }
    }
    // std::cout << max_voxels << " " << max_points << " " <<voxel_feature_num << " " << point_cloud_range[0] << " " << point_cloud_range[1] << " "
    // << point_cloud_range[2] << " "<< point_cloud_range[3] << " " << point_cloud_range[4] << " " << point_cloud_range[5] << " " << voxel_size[0] << " "
    // << voxel_size[1] << " " << voxel_size[2] << std::endl;
    std::cout <<  max_points_num  << " " << max_pillars_num << " "<< feature_num << " " << std::endl;
    IPluginV2DynamicExt* plugin = new TorchScatterMaxPlugin(max_points_num, max_pillars_num,feature_num);
    return plugin;
}

IPluginV2* TorchScatterMaxPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new TorchScatterMaxPlugin(serialData, serialLength);
}

void TorchScatterMaxPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* TorchScatterMaxPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

TorchScatterMaxPluginCreator::~TorchScatterMaxPluginCreator()
{
}