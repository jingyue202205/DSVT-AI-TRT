#include "map2bev.h"

using namespace nvinfer1;
using nvinfer1::Map2BevPlugin;
using nvinfer1::Map2BevPluginCreator;
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
static const char* PLUGIN_NAME{"Map2BevPlugin"};

// Static class fields initialization
PluginFieldCollection Map2BevPluginCreator::mFC{};
std::vector<PluginField> Map2BevPluginCreator::mPluginAttributes;

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
Map2BevPlugin::Map2BevPlugin(
              int max_pillars_num, int channel_num,int grid_size_x,int grid_size_y) :   
    max_pillars_num_(max_pillars_num), channel_num_(channel_num),grid_size_x_(grid_size_x),grid_size_y_(grid_size_y)
{
}

Map2BevPlugin::Map2BevPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    max_pillars_num_ = readFromBuffer<int>(d);
    channel_num_ = readFromBuffer<int>(d);
    grid_size_x_ = readFromBuffer<int>(d);
    grid_size_y_ = readFromBuffer<int>(d);
    
}

IPluginV2DynamicExt* Map2BevPlugin::clone() const noexcept
{
    auto* plugin = new Map2BevPlugin( max_pillars_num_,channel_num_,grid_size_x_,grid_size_y_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs Map2BevPlugin::getOutputDimensions(
int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
// 5 returns 
{
    // assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    auto batch_size = inputs[0].d[0];
    // auto batch_size = 1;
    // std::cout  << inputs[0].nbDims << " " << inputs[0].d[0]->getConstantValue() << " " << inputs[0].d[1]->getConstantValue() << " " << inputs[0].d[2]->getConstantValue() << std::endl;
    // std::cout  << inputs[1].nbDims << " " << inputs[1].d[0]->getConstantValue() << std::endl;axis_id
    if (outputIndex == 0)
    {
        // std::cout << "batch_size: " << batch_size->getConstantValue() << " voxel_num: " << voxelNum_ << " featurennum_: " << featureNum_ << std::endl;
        nvinfer1::DimsExprs dim0{};
        dim0.nbDims = 4;
        dim0.d[0] = batch_size;
        dim0.d[1] = exprBuilder.constant(grid_size_x_);
        dim0.d[2] = exprBuilder.constant(grid_size_y_);
        dim0.d[3] = exprBuilder.constant(channel_num_);
        return dim0; // 1 192  468 468
    }
}

bool Map2BevPlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // PLUGIN_ASSERT(nbInputs == 2);
    // PLUGIN_ASSERT(nbOutputs == 2);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       // voxel_features  1 5504 192
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)       // coors  1 5504 4
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)       //valid_set_num 1
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)       //map feature 1 192  468 468
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    return false;
}

void Map2BevPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t Map2BevPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

// __global__ void Map2Bev_kernel(
//      float* voxel_features, float *pose_features, 
//             unsigned int *voxel_inds, float * query_features_data, float * key_features_data, 
//             float *value_features_data, int axis_id, unsigned int *valid_set_num, int max_win_num, int voxel_num_set,int channel_num)
// {
//     unsigned int set_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(set_idx >= *valid_set_num) return;

//     unsigned int * new_voxel_inds = voxel_inds + axis_id * max_win_num * voxel_num_set;
//     for(int i=0;i<voxel_num_set;i++)
//     {
//         unsigned int global_voxel_id = *(new_voxel_inds+set_idx*voxel_num_set+i);
//         float * new_voxel_features = voxel_features + global_voxel_id * channel_num;
//         float *new_pose_features = pose_features + global_voxel_id * channel_num;

//         for(int j=0;j<channel_num;j++)
//         {
//             *(query_features_data+set_idx*voxel_num_set*channel_num+i*channel_num+j) = *(new_voxel_features+j) + *(new_pose_features+j);
//             *(key_features_data+set_idx*voxel_num_set*channel_num+i*channel_num+j) = *(new_voxel_features+j) + *(new_pose_features+j);
//             *(value_features_data+set_idx*voxel_num_set*channel_num+i*channel_num+j) = *(new_voxel_features+j) ;
//         }
//     }
// }


// __global__ void Map2Bev_kernel(
//      float* voxel_features, float *pose_features, 
//             unsigned int *voxel_inds, float * query_features_data, float * key_features_data, 
//             float *value_features_data, int axis_id, unsigned int *valid_set_num, int max_win_num, int voxel_num_set,int channel_num)
// {
//     unsigned int set_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(set_idx >= (*valid_set_num)*voxel_num_set) return;

//     unsigned int * new_voxel_inds = voxel_inds + axis_id * max_win_num * voxel_num_set;
//     // for(int i=0;i<voxel_num_set;i++)
//     // {
//         unsigned int global_voxel_id = *(new_voxel_inds+set_idx);
//         float * new_voxel_features = voxel_features + global_voxel_id * channel_num;
//         float *new_pose_features = pose_features + global_voxel_id * channel_num;
//         // float a;
//         // float b;
//         // float c;
//         for(int j=0;j<channel_num;j++)
//         {
//             *(query_features_data+set_idx*channel_num+j) = *(new_voxel_features+j) + *(new_pose_features+j);
//             *(key_features_data+set_idx*channel_num+j) = *(new_voxel_features+j) + *(new_pose_features+j);
//             *(value_features_data+set_idx*channel_num+j) = *(new_voxel_features+j) ;
//             // a = *(new_voxel_features+j);
//             // b =  *(new_pose_features+j);
//             // c = a + b;
    
//             // // *(query_features_data+set_idx*channel_num+j) = c;
//             // // *(key_features_data+set_idx*channel_num+j) = c;
//             // // *(value_features_data+set_idx*channel_num+j) = a;

//             // atomicExch(query_features_data+set_idx*channel_num+j,c);
//             // atomicExch(key_features_data+set_idx*channel_num+j,c);
//             // atomicExch(value_features_data+set_idx*channel_num+j,a);
            

//         }
// }
  

  __global__ void Map2Bev_kernel(
     float* voxel_features, unsigned int *coors, 
            unsigned int *valid_voxel_num, float*map_features_data, int channel_num,int grid_size_x,int grid_size_y)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= (*valid_voxel_num)*channel_num) return;

    int voxel_idx = idx / channel_num;

    uint4 coor = ((uint4*)coors)[voxel_idx];
    unsigned int y_index = coor.z;
    unsigned int x_index = coor.w;
    int diff_index = idx % channel_num;

    *(map_features_data+y_index*grid_size_x*channel_num+x_index*channel_num+diff_index) = *(voxel_features+voxel_idx*channel_num+diff_index);

}

cudaError_t Map2Bev_launch(float* voxel_features, unsigned int *coors, 
            unsigned int *valid_voxel_num, float*map_features_data, int channel_num,int grid_size_x,int grid_size_y,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_PILLARS_NUM*POSEMBED_LAYBERS_OUT_FEATURES+threadNum-1)/threadNum);
  dim3 threads(threadNum);
    Map2Bev_kernel<<<blocks, threads, 0, stream>>>(
           voxel_features, coors, valid_voxel_num,map_features_data,channel_num,grid_size_x,grid_size_y);
  cudaError_t err = cudaGetLastError();
  return err;
}


int Map2BevPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];
    // int maxNumPoints = inputDesc[0].dims.d[1];

    //TRT-input
    // std::cout << "voxelgenerator batch_size: " << batchSize << std::endl;
    float * voxel_features = const_cast<float *>((const float *)inputs[0]);
    unsigned int* coors = const_cast<unsigned int *>((const unsigned int *)inputs[1]);
    unsigned int* valid_voxel_num =  const_cast<unsigned int *>((const unsigned int *)inputs[2]);

    //TRT-output
    
    float *map_features_data = (float*)(outputs[0]); // 1 192  468 468
    
    
    // initialize output
    unsigned int map_features_data_size = batchSize * grid_size_x_  * grid_size_y_ *  channel_num_ * sizeof(float);
    
    
    checkCudaErrors(cudaMemsetAsync(map_features_data, 0, map_features_data_size, stream));
  
    checkCudaErrors(Map2Bev_launch(
          voxel_features, coors, valid_voxel_num,map_features_data,channel_num_,grid_size_x_,grid_size_y_,
          stream));
    return 0;
}

nvinfer1::DataType Map2BevPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // if(index == 5)
    // {
    //     return  nvinfer1::DataType::kFLOAT;
    // }
    return inputTypes[0];
}

const char* Map2BevPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* Map2BevPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int Map2BevPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int Map2BevPlugin::initialize() noexcept
{
    return 0;
}

void Map2BevPlugin::terminate() noexcept
{
}

size_t Map2BevPlugin::getSerializationSize() const noexcept
{
    return   4 * sizeof(int);
}

void Map2BevPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, max_pillars_num_);
    writeToBuffer<int>(d, channel_num_);
    writeToBuffer<int>(d, grid_size_x_);
    writeToBuffer<int>(d, grid_size_y_);
}

void Map2BevPlugin::destroy() noexcept
{
    delete this;
}

void Map2BevPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* Map2BevPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}


Map2BevPluginCreator::Map2BevPluginCreator()
{
    
    mPluginAttributes.clear();

    // std::cout <<  *max_num_points_per_voxel_ptr << std::endl;max_win_num
    mPluginAttributes.emplace_back(PluginField("max_pillars_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("channel_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("grid_size_x", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("grid_size_y", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* Map2BevPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* Map2BevPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* Map2BevPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* Map2BevPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int max_pillars_num = 0;
    int channel_num = 0;
    int grid_size_x = 0;
    int grid_size_y = 0;
   
    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "max_pillars_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_pillars_num = d[0];
        }
        else if (!strcmp(attr_name, "channel_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            channel_num = d[0];
        }
        else if (!strcmp(attr_name, "grid_size_x"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            grid_size_x = d[0];
        }
        else if (!strcmp(attr_name, "grid_size_y"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            grid_size_y = d[0];
        }
    }

    std::cout <<  max_pillars_num  << " " << channel_num << " " << grid_size_x << " " << grid_size_y << " " << std::endl;
    IPluginV2DynamicExt* plugin = new Map2BevPlugin(max_pillars_num,channel_num, grid_size_x,grid_size_y);
    return plugin;
}

IPluginV2* Map2BevPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new Map2BevPlugin(serialData, serialLength);
}

void Map2BevPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* Map2BevPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

Map2BevPluginCreator::~Map2BevPluginCreator()
{
}