#include "getValueByIndex.h"

using namespace nvinfer1;
using nvinfer1::GetValueByIndexPlugin;
using nvinfer1::GetValueByIndexPluginCreator;
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
static const char* PLUGIN_NAME{"GetValueByIndexPlugin"};

// Static class fields initialization
PluginFieldCollection GetValueByIndexPluginCreator::mFC{};
std::vector<PluginField> GetValueByIndexPluginCreator::mPluginAttributes;

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
GetValueByIndexPlugin::GetValueByIndexPlugin(
               int voxel_num_set, int max_win_num, int channel_num, int axis_id) :  
               voxel_num_set_(voxel_num_set), 
    max_win_num_(max_win_num), channel_num_(channel_num), axis_id_(axis_id)
{
}

GetValueByIndexPlugin::GetValueByIndexPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    voxel_num_set_ = readFromBuffer<int>(d);
    max_win_num_ = readFromBuffer<int>(d);
    channel_num_ = readFromBuffer<int>(d);
    axis_id_ = readFromBuffer<int>(d);
}

IPluginV2DynamicExt* GetValueByIndexPlugin::clone() const noexcept
{
    auto* plugin = new GetValueByIndexPlugin( voxel_num_set_,max_win_num_,channel_num_,axis_id_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs GetValueByIndexPlugin::getOutputDimensions(
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
        dim0.d[1] = exprBuilder.constant(max_win_num_);
        dim0.d[2] = exprBuilder.constant(voxel_num_set_);
        dim0.d[3] = exprBuilder.constant(channel_num_);
        return dim0; // global_index 1 3200 600
    }
    if(outputIndex == 1){
        // std::cout << "batch_size: " << batch_size->getConstantValue() << "  voxel_num: " << voxelNum_ << " featurennum_: " << 4 << std::endl;
        nvinfer1::DimsExprs dim1{};
        dim1.nbDims = 4;
        dim1.d[0] = batch_size;
        dim1.d[1] = exprBuilder.constant(max_win_num_);
        dim1.d[2] = exprBuilder.constant(voxel_num_set_);
        dim1.d[3] = exprBuilder.constant(channel_num_);
        return dim1; // coors_in_win 1 3200 600 3
    }
      if(outputIndex == 2){
        // std::cout << "batch_size: " << batch_size->getConstantValue() << "  voxel_num: " << voxelNum_ << " featurennum_: " << 4 << std::endl;
        nvinfer1::DimsExprs dim2{};
        dim2.nbDims = 4;
        dim2.d[0] = batch_size;
        dim2.d[1] = exprBuilder.constant(max_win_num_);
        dim2.d[2] = exprBuilder.constant(voxel_num_set_);
        dim2.d[3] = exprBuilder.constant(channel_num_);
        return dim2; // voxel _num_in_win  1 3200
    }
}

bool GetValueByIndexPlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // PLUGIN_ASSERT(nbInputs == 2);
    // PLUGIN_ASSERT(nbOutputs == 2);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       // voxel_features  1 5504 192
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)       // pose_features  1 5504 192
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)       //voxel_inds 454 36
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)       //valid set num 1
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 4)       //query  454 36 192
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 5)       // key 454 36 192
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
     if (pos == 6)       //value 454 36 192
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    return false;
}

void GetValueByIndexPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t GetValueByIndexPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}

// __global__ void getValueByIndex_kernel(
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


// __global__ void getValueByIndex_kernel(
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
  

  __global__ void getValueByIndex_kernel(
     float* voxel_features, float *pose_features, 
            unsigned int *voxel_inds, float * query_features_data, float * key_features_data, 
            float *value_features_data, int axis_id, unsigned int *valid_set_num, int max_win_num, int voxel_num_set,int channel_num)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= (*valid_set_num)*voxel_num_set*channel_num) return;

    int set_idx = idx / channel_num;

    unsigned int * new_voxel_inds = voxel_inds + axis_id * max_win_num * voxel_num_set;

        unsigned int global_voxel_id = *(new_voxel_inds+set_idx);
        float * new_voxel_features = voxel_features + global_voxel_id * channel_num;
        float *new_pose_features = pose_features + global_voxel_id * channel_num;

        int diff_idx = idx % channel_num;
        *(query_features_data+idx) = *(new_voxel_features+diff_idx) + *(new_pose_features+diff_idx);
        *(key_features_data+idx) = *(new_voxel_features+diff_idx) + *(new_pose_features+diff_idx);
        *(value_features_data+idx) = *(new_voxel_features+diff_idx) ;
        
}

cudaError_t getValueByIndex_launch(float* voxel_features, float *pose_features, 
            unsigned int *voxel_inds, float * query_features_data, float * key_features_data, float *value_features_data, int axis_id, 
            unsigned int *valid_set_num,int max_win_num, int voxel_num_set, int channel_num,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_WIN_NUM*VOXEL_NUM_SET*POSEMBED_LAYBERS_OUT_FEATURES+threadNum-1)/threadNum);
  dim3 threads(threadNum);
    getValueByIndex_kernel<<<blocks, threads, 0, stream>>>(
       voxel_features, pose_features, 
            voxel_inds, query_features_data, key_features_data, value_features_data,  axis_id, valid_set_num,max_win_num, voxel_num_set,channel_num);
  cudaError_t err = cudaGetLastError();
  return err;
}


int GetValueByIndexPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];
    // int maxNumPoints = inputDesc[0].dims.d[1];

    //TRT-input
    // std::cout << "voxelgenerator batch_size: " << batchSize << std::endl;
    float * voxel_features = const_cast<float *>((const float *)inputs[0]);
    float * pose_features = const_cast<float *>((const float *)inputs[1]);
    unsigned int* voxel_inds = const_cast<unsigned int *>((const unsigned int *)inputs[2]);
    unsigned int* valid_set_num =  const_cast<unsigned int *>((const unsigned int *)inputs[3]);

    //TRT-output
    
    float *query_features_data = (float*)(outputs[0]); // 1 max_win_num 36 192
    float *key_features_data = (float*)(outputs[1]); // 1 max_win_num 36 192
    float *value_features_data = (float*)(outputs[2]); // 1 max_win_num 36 192
    
    // initialize output
    unsigned int query_features_data_size = batchSize * max_win_num_  * voxel_num_set_ *  channel_num_ * sizeof(float);
    unsigned int key_features_data_size = batchSize * max_win_num_  * voxel_num_set_ *  channel_num_ * sizeof(float);
    unsigned int value_features_data_size = batchSize * max_win_num_  * voxel_num_set_ *  channel_num_ * sizeof(float);
    
    checkCudaErrors(cudaMemsetAsync(query_features_data, 0, query_features_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(key_features_data,0,key_features_data_size,stream));
    checkCudaErrors(cudaMemsetAsync(value_features_data, 0, value_features_data_size, stream));

    checkCudaErrors(getValueByIndex_launch(
          voxel_features, pose_features, voxel_inds, query_features_data, key_features_data,value_features_data,axis_id_,valid_set_num,max_win_num_, voxel_num_set_,channel_num_,
          stream));
    return 0;
}

nvinfer1::DataType GetValueByIndexPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // if(index == 5)
    // {
    //     return  nvinfer1::DataType::kFLOAT;
    // }
    return inputTypes[0];
}

const char* GetValueByIndexPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* GetValueByIndexPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int GetValueByIndexPlugin::getNbOutputs() const noexcept
{
    return 3;
}

int GetValueByIndexPlugin::initialize() noexcept
{
    return 0;
}

void GetValueByIndexPlugin::terminate() noexcept
{
}

size_t GetValueByIndexPlugin::getSerializationSize() const noexcept
{
    return   4 * sizeof(int);
}

void GetValueByIndexPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, voxel_num_set_);
    writeToBuffer<int>(d, max_win_num_);
    writeToBuffer<int>(d, channel_num_);
    writeToBuffer<int>(d, axis_id_);
}

void GetValueByIndexPlugin::destroy() noexcept
{
    delete this;
}

void GetValueByIndexPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* GetValueByIndexPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}


GetValueByIndexPluginCreator::GetValueByIndexPluginCreator()
{
    
    mPluginAttributes.clear();

    // std::cout <<  *max_num_points_per_voxel_ptr << std::endl;max_win_num
    mPluginAttributes.emplace_back(PluginField("max_win_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("voxel_num_set", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("channel_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("axis_id", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GetValueByIndexPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* GetValueByIndexPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* GetValueByIndexPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GetValueByIndexPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int max_win_num = 0;
    int voxel_num_set = 0;
    int channel_num = 0;
    int axis_id = 0;  // 0 y 1 x
   
    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "max_win_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_win_num = d[0];
        }
        else if (!strcmp(attr_name, "voxel_num_set"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            voxel_num_set = d[0];
        }
        else if (!strcmp(attr_name, "channel_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            channel_num = d[0];
        }
        else if (!strcmp(attr_name, "axis_id"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            axis_id = d[0];
        }
    }
    // std::cout << max_voxels << " " << max_points << " " <<voxel_feature_num << " " << point_cloud_range[0] << " " << point_cloud_range[1] << " "
    // << point_cloud_range[2] << " "<< point_cloud_range[3] << " " << point_cloud_range[4] << " " << point_cloud_range[5] << " " << voxel_size[0] << " "
    // << voxel_size[1] << " " << voxel_size[2] << std::endl;
    std::cout <<  max_win_num  << " " << voxel_num_set << " " << channel_num << " " << axis_id << " " << std::endl;
    IPluginV2DynamicExt* plugin = new GetValueByIndexPlugin(voxel_num_set,max_win_num,channel_num, axis_id);
    return plugin;
}

IPluginV2* GetValueByIndexPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new GetValueByIndexPlugin(serialData, serialLength);
}

void GetValueByIndexPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* GetValueByIndexPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

GetValueByIndexPluginCreator::~GetValueByIndexPluginCreator()
{
}