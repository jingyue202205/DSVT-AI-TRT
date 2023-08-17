#include "points2Features.h"

using namespace nvinfer1;
using nvinfer1::Points2FeaturesPlugin;
using nvinfer1::Points2FeaturesPluginCreator;
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
static const char* PLUGIN_NAME{"Points2FeaturesPlugin"};

// Static class fields initialization
PluginFieldCollection Points2FeaturesPluginCreator::mFC{};
std::vector<PluginField> Points2FeaturesPluginCreator::mPluginAttributes;

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
Points2FeaturesPlugin::Points2FeaturesPlugin(
             int max_points_num, int max_points_num_voxel_filter, int max_pillars_num, int point_feature_num, int feature_num, int max_num_points_per_voxel,
              float x_min, float x_max, float y_min, float y_max, float z_min, float z_max,
              float voxel_x, float voxel_y, float voxel_z,int grid_size_x, int grid_size_y, int grid_size_z
) : max_points_num_(max_points_num), max_points_num_voxel_filter_(max_points_num_voxel_filter),max_pillars_num_(max_pillars_num), point_feature_num_(point_feature_num), 
    feature_num_(feature_num),max_num_points_per_voxel_(max_num_points_per_voxel),
    min_x_range_(x_min), max_x_range_(x_max), min_y_range_(y_min),
    max_y_range_(y_max), min_z_range_(z_min), max_z_range_(z_max),
    voxel_x_size_(voxel_x), voxel_y_size_(voxel_y),
    voxel_z_size_(voxel_z), grid_size_x_(grid_size_x), grid_size_y_(grid_size_y), grid_size_z_(grid_size_z)
{
}

Points2FeaturesPlugin::Points2FeaturesPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    max_points_num_ = readFromBuffer<int>(d);
    max_points_num_voxel_filter_ = readFromBuffer<int>(d);
    max_pillars_num_ = readFromBuffer<int>(d);
    point_feature_num_ = readFromBuffer<int>(d);
    feature_num_ = readFromBuffer<int>(d);
    max_num_points_per_voxel_ = readFromBuffer<int>(d);

    min_x_range_ = readFromBuffer<float>(d);
    max_x_range_ = readFromBuffer<float>(d);
    min_y_range_ = readFromBuffer<float>(d);
    max_y_range_ = readFromBuffer<float>(d);
    min_z_range_ = readFromBuffer<float>(d);
    max_z_range_ = readFromBuffer<float>(d);
    voxel_x_size_ = readFromBuffer<float>(d);
    voxel_y_size_ = readFromBuffer<float>(d);
    voxel_z_size_ = readFromBuffer<float>(d);

    grid_size_x_ = readFromBuffer<int>(d);
    grid_size_y_ = readFromBuffer<int>(d);
    grid_size_z_ = readFromBuffer<int>(d);
}

IPluginV2DynamicExt* Points2FeaturesPlugin::clone() const noexcept
{
    auto* plugin = new Points2FeaturesPlugin(max_points_num_, max_points_num_voxel_filter_, max_pillars_num_, point_feature_num_, feature_num_, max_num_points_per_voxel_,
     min_x_range_, max_x_range_,min_y_range_, max_y_range_, min_z_range_, max_z_range_, 
     voxel_x_size_, voxel_y_size_, voxel_z_size_,grid_size_x_,grid_size_y_,grid_size_z_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs Points2FeaturesPlugin::getOutputDimensions(
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
        dim0.d[1] = exprBuilder.constant(max_points_num_voxel_filter_);
        dim0.d[2] = exprBuilder.constant(feature_num_);
        return dim0; // features 1 60000 10
    }
    if(outputIndex == 1){
        // std::cout << "batch_size: " << batch_size->getConstantValue() << "  voxel_num: " << voxelNum_ << " featurennum_: " << 4 << std::endl;
        nvinfer1::DimsExprs dim1{};
        dim1.nbDims = 3;
        dim1.d[0] = batch_size;
        dim1.d[1] = exprBuilder.constant(max_pillars_num_);
        dim1.d[2] = exprBuilder.constant(max_num_points_per_voxel_);
        return dim1; // voxel  _ point index
    }
      if(outputIndex == 2){
        // std::cout << "batch_size: " << batch_size->getConstantValue() << "  voxel_num: " << voxelNum_ << " featurennum_: " << 4 << std::endl;
        nvinfer1::DimsExprs dim2{};
        dim2.nbDims = 3;
        dim2.d[0] = batch_size;
        dim2.d[1] = exprBuilder.constant(max_pillars_num_);
        dim2.d[2] = exprBuilder.constant(4);
        return dim2; // voxel _coords
    }
         if(outputIndex == 3){
        // std::cout << "batch_size: " << batch_size->getConstantValue() << "  voxel_num: " << voxelNum_ << " featurennum_: " << 4 << std::endl;
        nvinfer1::DimsExprs dim3{};
        dim3.nbDims = 3;
        dim3.d[0] = batch_size;
        dim3.d[1] = exprBuilder.constant(max_pillars_num_);
        dim3.d[2] = exprBuilder.constant(1);
        return dim3; // point num in voxel
    }
      if(outputIndex == 4)
    {   
        // std::cout << "batch_size: " << batch_size->getConstantValue() << std::endl;
        nvinfer1::DimsExprs dim4{};
        dim4.nbDims = 1;
        dim4.d[0] = batch_size;
        return dim4;  // valid voxel_num
    }
     if(outputIndex == 5)
    {   
        // std::cout << "batch_size: " << batch_size->getConstantValue() << std::endl;
        nvinfer1::DimsExprs dim5{};
        dim5.nbDims = 1;
        dim5.d[0] = batch_size;
        return dim5;  // valid point_num
    }
}

bool Points2FeaturesPlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // PLUGIN_ASSERT(nbInputs == 2);
    // PLUGIN_ASSERT(nbOutputs == 2);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       // PointCloud Array --- x, y, z, i   dim: 1  40000 4
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)       // Point Num
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)       // features, dim: 1 40000 10
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)       // voxe point index, dim: 1 x 20000 x 32
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 4)       // voxelCoords, dim: 1 x 20000 x 4
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
     if (pos == 5)       // point num in voxel, dim: 1 x 20000 x 1
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 6)    // voxel_num valid
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
     if (pos == 7)    // point_num valid
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    return false;
}

void Points2FeaturesPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t Points2FeaturesPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    int batchSize = inputs[0].dims.d[0];
    int dense_voxel_num = grid_size_z_ * grid_size_y_ * grid_size_x_;
    size_t mask_size = batchSize * dense_voxel_num * sizeof(unsigned int);
    size_t global_voxels_size = batchSize * dense_voxel_num * max_num_points_per_voxel_ * point_feature_num_ * sizeof(float);
    size_t voxels_size = batchSize * max_pillars_num_ * max_num_points_per_voxel_ * point_feature_num_ * sizeof(float);
    
    size_t workspaces[3];
    workspaces[0] = mask_size;
    workspaces[1] = global_voxels_size;
    workspaces[2] = voxels_size;

    return  calculateTotalWorkspaceSize(workspaces, 3);
}

// __global__ void generateAverage_kernel(float *point_features_data,
//         unsigned int *point_index_in_voxel_data,unsigned int* point_num_in_voxel_data,unsigned int *voxel_num_data)
// {
//     int voxelidx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(voxelidx >= *voxel_num_data) return;

//     unsigned int *point_index_in_voxel_addr  = point_index_in_voxel_data + voxelidx * POINTS_NUM_PER_VOXEL;
//     unsigned int point_num = *(point_num_in_voxel_data+voxelidx);

//     float cluster_center_x = 0;
//     float cluster_center_y = 0;
//     float cluster_center_z = 0;
//     // printf("point_num: %d\n", point_num);
//     // if (voxelidx == 0)
//     // {
//     //     printf('voxelidx: %d,%d\n',voxelidx,point_num);
//     // }
//     for(int i=0;i<point_num;i++)
//     {
//         float x = point_features_data[(*(point_index_in_voxel_addr+i))*FEATURES_NUM+0];
//         float y = point_features_data[(*(point_index_in_voxel_addr+i))*FEATURES_NUM+1];
//         float z = point_features_data[(*(point_index_in_voxel_addr+i))*FEATURES_NUM+2];
//         cluster_center_x +=  x;
//         cluster_center_y +=  y;
//         cluster_center_z += z;
//         if (voxelidx == 0)
//         {
//             printf("voxelidx:%d, %d,%d,%f,%f,%f,%f,%f,%f\n",i,voxelidx,point_num,x,y,z,cluster_center_x,cluster_center_y,cluster_center_z);
//         }
//     }
//      if (voxelidx == 0)
//         {
//             printf("voxelidx: %d,%d,%f,%f,%f\n",voxelidx,point_num,cluster_center_x,cluster_center_y,cluster_center_z);
//         }
//     cluster_center_x = cluster_center_x / point_num;
//     cluster_center_y = cluster_center_y / point_num;
//     cluster_center_z = cluster_center_z / point_num;

//     if (voxelidx == 0)
//         {
//             printf("voxelidx: %d,%d,%f,%f,%f\n",voxelidx,point_num,cluster_center_x,cluster_center_y,cluster_center_z);
//         }
//     // for(int i=0;i<point_num;i++)
//     // {
//     //     if(*(point_index_in_voxel_addr+i)==1)
//     //     {   
//     //         printf("point_num: %d\n", point_num);
//     //         printf("center_x:%f,center_y:%f,ceter_z:%f\n",cluster_center_x,cluster_center_y,cluster_center_z);
//     //     }
//     // }

//     for(int i=0;i<point_num;i++)
//     {
//             float x = point_features_data[(*(point_index_in_voxel_addr+i))*FEATURES_NUM+0];
//             float y = point_features_data[(*(point_index_in_voxel_addr+i))*FEATURES_NUM+1];
//             float z = point_features_data[(*(point_index_in_voxel_addr+i))*FEATURES_NUM+2];
//             point_features_data[(*(point_index_in_voxel_addr+i))*FEATURES_NUM+4] = x - cluster_center_x;
//             point_features_data[(*(point_index_in_voxel_addr+i))*FEATURES_NUM+5] = y - cluster_center_y;
//             point_features_data[(*(point_index_in_voxel_addr+i))*FEATURES_NUM+6] = z - cluster_center_z;

//             if (voxelidx == 0)
//         {
//             printf("voxelidx:%d, %d,%d,%f,%f,%f,%f,%f,%f\n",i,voxelidx,point_num,x-cluster_center_x,y-cluster_center_y,z-cluster_center_z,
//             point_features_data[(*(point_index_in_voxel_addr+i))*FEATURES_NUM+4],point_features_data[(*(point_index_in_voxel_addr+i))*FEATURES_NUM+5],point_features_data[(*(point_index_in_voxel_addr+i))*FEATURES_NUM+6]);
//         }

//     }

//     #if 0
//      if (index_z == 32 && index_y == 731 && index_x == 95)
//      {
//          printf("sum: %f,%f,%f,%f\n",sum_x,sum_y,sum_z,sum_i);
//      }
//      #endif

//     // float x = sum_x/num_point;
//     // float y = sum_y/num_point;
//     // float z = sum_z/num_point;
//     // float inten = sum_i / num_point;

//     #if 0
//     if (index_z == 32 && index_y == 731 && index_x == 95)
//      {
//          printf("address_sum: %f,%f,%f,%f\n",*(address),*(address+1),*(address+2),*(address+3));
//      }
//     #endif
// }


// cudaError_t generateAverage_launch(float *point_features_data,
//         unsigned int *point_index_in_voxel_data,unsigned int* point_num_in_voxel_data,unsigned int* voxel_num_data,
//         cudaStream_t stream)
// {
//   int threadNum = THREADS_FOR_VOXEL;
//   dim3 blocks((MAX_VOXELS_NUM+threadNum-1)/threadNum);
//   dim3 threads(threadNum);
//   generateAverage_kernel<<<blocks, threads, 0, stream>>>
//        (point_features_data,point_index_in_voxel_data,point_num_in_voxel_data,voxel_num_data);
//   cudaError_t err = cudaGetLastError();
//   return err;
// }

// __global__ void generateNewFeature_kernel(float *point_features_data,
//         unsigned int *point_index_in_voxel_data,unsigned int* point_num_in_voxel_data,unsigned int* voxel_num_data, float *new_point_features_data,
//         unsigned int *new_point_index_in_voxel_data, unsigned int *point_num_data)
// {
//     int voxelidx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(voxelidx >= *voxel_num_data) return;

//     unsigned int *point_index_in_voxel_addr  = new_point_index_in_voxel_data + voxelidx * POINTS_NUM_PER_VOXEL;
//     unsigned int point_num = *(point_num_in_voxel_data+voxelidx);

//      for(int i=0;i<point_num;i++)
//     {
//         unsigned int current_point_id = atomicAdd(point_num_data,1);
//         for(int j = 0;j<FEATURES_NUM;j++)
//         {
//             point_features_data[current_point_id*FEATURES_NUM+j] = new_point_features_data[(*(point_index_in_voxel_addr+i))*FEATURES_NUM+j];
//         }
//         point_index_in_voxel_data[voxelidx * POINTS_NUM_PER_VOXEL+i] = current_point_id;
      
//     }
// }

// cudaError_t generateNewFeature_launch(float *point_features_data,
//         unsigned int *point_index_in_voxel_data,unsigned int* point_num_in_voxel_data,unsigned int* voxel_num_data, float *new_point_features_data,
//         unsigned int *new_point_index_in_voxel_data, unsigned int * point_num_data,
//         cudaStream_t stream)
// {
//   int threadNum = THREADS_FOR_VOXEL;
//   dim3 blocks((MAX_VOXELS_NUM+threadNum-1)/threadNum);
//   dim3 threads(threadNum);
//   generateNewFeature_kernel<<<blocks, threads, 0, stream>>>
//        (point_features_data,point_index_in_voxel_data,point_num_in_voxel_data,voxel_num_data, new_point_features_data, new_point_index_in_voxel_data, point_num_data);
//   cudaError_t err = cudaGetLastError();
//   return err;
// }


// __device__ void cuda_sleep(int64_t num_cycles)
// {
//     int64_t cycles = 0;
//     int64_t start = clock64();
//     while(cycles < num_cycles)
//     {
//         cycles = clock64() - start;
//     }
// }


// __global__ void generateVoxels_kernel(
//        float *points, unsigned int* points_size,
//         float min_x_range, float max_x_range,
//         float min_y_range, float max_y_range,
//         float min_z_range, float max_z_range,
//         float voxel_x_size, float voxel_y_size, float voxel_z_size,
//         int grid_x_size, int grid_y_size,int grid_z_size,
//         unsigned int *coor_to_voxelidx, float * point_features_data, 
//         unsigned int * point_index_in_voxel_data,unsigned int *coords_data, 
//         unsigned int *point_num_in_voxel_data,unsigned int *voxel_num_data)
// {
//     // printf("point_size:%d\n",*points_size);
//     unsigned int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(point_idx >= *points_size) return;
//     // printf("generatevoxel11111111");
//     float4 point = ((float4*)points)[point_idx];
    
//     if( !(point.x >= min_x_range && point.x < max_x_range
//         && point.y >= min_y_range && point.y < max_y_range
//         && point.z >= min_z_range && point.z < max_z_range) ) {
//       return;
//     }
//     // printf("generatevoxel222222\n");
//     int index_x = floorf((point.x - min_x_range) / voxel_x_size);
//     int index_y = floorf((point.y - min_y_range) / voxel_y_size);
//     int index_z = floorf((point.z - min_z_range) / voxel_z_size);
//     // printf("index_x: %d,%f,%f,%f\n",index_x,point.x,min_x_range,voxel_x_size);
//     // printf("index_i: %d,%d,%d\n",index_x,index_y,index_z);
//     // bool failed = false;
//     if ((index_x < 0) or (index_x >= grid_x_size) or (index_y < 0) or (index_y >= grid_y_size) or (index_z < 0) or (index_z >= grid_z_size))
//         return;

//     unsigned int voxel_index = index_z * (grid_y_size * grid_x_size) + index_y * grid_x_size + index_x;
//                                 // index_z * (grid_y_size * grid_x_size * 2) + index_y*grid_z_size *2 + index_x*2 + 
//     unsigned int point_id = atomicAdd(coor_to_voxelidx+voxel_index*2,1);
//     if(point_id >= POINTS_NUM_PER_VOXEL) return;    
    

//     // save point to point_feature
//     float * current_point_features_data_address = point_features_data + point_idx * FEATURES_NUM;  // 10
//     //  atomicExch(current_point_features_data_address+0,point.x);
//     //  atomicExch(current_point_features_data_address+1,point.y);
//     //  atomicExch(current_point_features_data_address+2,point.z);
//     //  atomicExch(current_point_features_data_address+3,point.w);
//      *(current_point_features_data_address+0) = point.x;
//      *(current_point_features_data_address+1) = point.y;
//      *(current_point_features_data_address+2) = point.z;
//      *(current_point_features_data_address+3) = point.w;

//      float f_center_x =  point.x  - ((index_x+0.5) * voxel_x_size + min_x_range);
//      float f_center_y = point.y - ((index_y+0.5) * voxel_y_size + min_y_range);
//      float f_center_z = point.z - ((index_z+0.5) * voxel_z_size + min_z_range);

//     //  atomicExch(current_point_features_data_address+7, f_center_x);
//     //  atomicExch(current_point_features_data_address+8, f_center_y);
//     //  atomicExch(current_point_features_data_address+9, f_center_z);
//      *(current_point_features_data_address+7) = f_center_x;
//      *(current_point_features_data_address+8) = f_center_y;
//      *(current_point_features_data_address+9) = f_center_z;

//     // 
//     unsigned int current_voxelid = 0;
//     if (point_id == 0)
//     {
//         //保存coor and current_voxel_id
//         current_voxelid = atomicAdd(voxel_num_data,1);
//         #if  0
//         if(current_voxelid == 0)
//         {
//             printf("current_voxelid:%d,%f,%f,%f,%d,%d,%d,%d\n",current_voxelid,point.x,point.y,point.z,index_x,index_y,index_z,voxel_index * 2 + 1);
//         }
//         #endif

//         //save current_voxelid
//         unsigned int *current_voxelid_address = coor_to_voxelidx + voxel_index * 2 + 1;
//         atomicExch(current_voxelid_address,current_voxelid);


//         //save coord
//         uint4 coord = {0,index_z,index_y,index_x};
//         ((uint4*)coords_data)[current_voxelid] = coord;

//     }
//     // nanosleep()
//     else{
//         // if(current_voxelid == 0 && )
//         // __nanosleep(100);
//         current_voxelid = coor_to_voxelidx[voxel_index*2+1];
//         if(current_voxelid == 0)
//         {
//             cuda_sleep(300000); //10000000
//             current_voxelid = coor_to_voxelidx[voxel_index*2+1];
//         }
//         #if 0
//          if(current_voxelid == 0)
//         {
//             printf("current_voxelid:%d,%f,%f,%f,%d,%d,%d,%d,%d\n",current_voxelid,point.x,point.y,point.z,index_x,index_y,index_z,point_id,voxel_index*2+1);
//         }
//         #endif
//     }

//      //point_num_in_voxel_data 
//     unsigned int *point_num_in_voxel_data_address = point_num_in_voxel_data + current_voxelid;
//     unsigned int total_num_in_voxel = *(coor_to_voxelidx+voxel_index*2);
//     if(total_num_in_voxel > POINTS_NUM_PER_VOXEL)
//         total_num_in_voxel = POINTS_NUM_PER_VOXEL;
//     atomicExch(point_num_in_voxel_data_address,total_num_in_voxel);

//     // save point index 
//     unsigned int * point_index_in_voxel_data_addr = point_index_in_voxel_data + current_voxelid * POINTS_NUM_PER_VOXEL;
//     // atomicExch(point_index_in_voxel_data_addr+point_id, point_idx);
//     *(point_index_in_voxel_data_addr+point_id) = point_idx;
    
//     #if 0
//     if (index_z == 32 && index_y == 731 && index_x == 95)
//     {
//         printf("point: %f,%f,%f,%f %d\n",point.x,point.y,point.z,point.w,point_id);
//         printf("adress: %f,%f,%f,%f\n",*address,*(address+1),*(address+2),*(address+3));
//         printf("fsfsfsfsf current_voxelid: %d\n",current_voxelid);
//         printf("coor_to_voxelidx[voxel_index*2]: %d\n",atomicAdd(coor_to_voxelidx+voxel_index*2,0));
//         printf("point_id: %d\n",point_id);
//         printf("num_points_per_voxel_address: %d\n",*num_points_per_voxel_address);
//     }
//     #endif

// }


// cudaError_t generateVoxels_launch(float *points, unsigned int* points_size,
//         float min_x_range, float max_x_range,
//         float min_y_range, float max_y_range,
//         float min_z_range, float max_z_range,
//         float voxel_x_size, float voxel_y_size, float voxel_z_size,
//         int grid_x_size, int grid_y_size,int grid_z_size,
//         unsigned int *coor_to_voxelidx,float * point_features_data, 
//         unsigned int * point_index_in_voxel_data, 
//         unsigned int *coords_data, unsigned int *point_num_in_voxel_data,
//         unsigned int * voxel_num_data, 
//         cudaStream_t stream)
// {
//   int threadNum = THREADS_FOR_VOXEL;

//   dim3 blocks((MAX_POINTS_NUM+threadNum-1)/threadNum);
//   dim3 threads(threadNum);
//   generateVoxels_kernel<<<blocks, threads, 0, stream>>>
//        (points, points_size,
//         min_x_range, max_x_range,
//         min_y_range, max_y_range,
//         min_z_range, max_z_range,
//         voxel_x_size, voxel_y_size, voxel_z_size,
//         grid_x_size, grid_y_size,grid_z_size,
//         coor_to_voxelidx,point_features_data,
//         point_index_in_voxel_data, coords_data,
//         point_num_in_voxel_data,voxel_num_data);
//   cudaError_t err = cudaGetLastError();
//   return err;
// }


// int Points2FeaturesPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
//     const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
//     cudaStream_t stream) noexcept
// {
//     int batchSize = inputDesc[0].dims.d[0];
//     // int maxNumPoints = inputDesc[0].dims.d[1];

//     //TRT-input
//     // std::cout << "voxelgenerator batch_size: " << batchSize << std::endl;
//     float * pointCloud = const_cast<float *>((const float *)inputs[0]);
//     unsigned int* pointNum = const_cast<unsigned int *>((const unsigned int *)inputs[1]);

//     //TRT-output
//     float *point_features_data = (float *)(outputs[0]);  // 1 max_point_num 10
//     unsigned int *point_index_in_voxel_data = (unsigned int *)(outputs[1]);  // 1 max_pillars_num 32
//     unsigned int *coords_data = (unsigned int *)(outputs[2]);  // 1 max_pillars_num 4
//     unsigned int *point_num_in_voxel_data = (unsigned int *)(outputs[3]);  // 1 max_pillars_num 1
//     unsigned int *voxel_num_data = (unsigned int *)(outputs[4]);  //  valid voxel num
//     unsigned int *point_num_data = (unsigned int*)(outputs[5]);  // valid point_num

//     // unsigned int *params_data = (unsigned int *)(outputs[2]);
//     int dense_voxel_num = grid_size_z_ * grid_size_y_ * grid_size_x_;
//     // std::cout << grid_size_x_ << " " << grid_size_y_ << " " << grid_size_z_ << std::endl;
//     size_t coor_to_voxelidx_size = batchSize * dense_voxel_num * 2* sizeof(unsigned int);
//     size_t new_point_features_size = batchSize * max_points_num_ * feature_num_ * sizeof(float);
//     size_t new_point_index_in_voxel_size = batchSize * max_pillars_num_ * max_num_points_per_voxel_ * sizeof(unsigned int);
    
//     size_t workspaces[3];
//     workspaces[0] = coor_to_voxelidx_size;
//     workspaces[1] = new_point_features_size;
//     workspaces[2] = new_point_index_in_voxel_size;
//     size_t total_workspace = calculateTotalWorkspaceSize(workspaces, 3);

//     unsigned int* coor_to_voxelidx = static_cast<unsigned int*>(workspace);
   
//     float* new_point_features_data = reinterpret_cast<float*>(
//         nextWorkspacePtr(reinterpret_cast<int8_t*>(coor_to_voxelidx), coor_to_voxelidx_size));
//     unsigned int* new_point_index_in_voxel_data = reinterpret_cast<unsigned int*>(
//         nextWorkspacePtr(reinterpret_cast<int8_t*>(new_point_features_data), new_point_features_size));
    
//     // Initialize workspace memory
//     checkCudaErrors(cudaMemsetAsync(coor_to_voxelidx, 0, total_workspace, stream)); // total_workspace
    
//     // initialize output
//     unsigned int point_features_data_size = batchSize * max_points_num_voxel_filter_  * feature_num_ * sizeof(float);
//     unsigned int point_index_in_voxel_data_size = batchSize * max_pillars_num_ * max_num_points_per_voxel_ * sizeof(unsigned int);
//     unsigned int coords_data_size = batchSize * max_pillars_num_ * 4 * sizeof(unsigned int);
//     unsigned int point_num_in_voxel_data_size = batchSize * max_pillars_num_ * 1 * sizeof(unsigned int);
//     unsigned int voxel_num_data_size = batchSize * sizeof(unsigned int);
//     unsigned int point_num_data_size = batchSize * sizeof(unsigned int);
    
//     checkCudaErrors(cudaMemsetAsync(point_features_data, 0, point_features_data_size, stream));
//     checkCudaErrors(cudaMemsetAsync(point_index_in_voxel_data,0,point_index_in_voxel_data_size,stream));
//     checkCudaErrors(cudaMemsetAsync(coords_data, 0, coords_data_size, stream));
//     checkCudaErrors(cudaMemsetAsync(point_num_in_voxel_data, 0, point_num_in_voxel_data_size, stream));
//     checkCudaErrors(cudaMemsetAsync(voxel_num_data, 0, voxel_num_data_size, stream));
//     checkCudaErrors(cudaMemsetAsync(point_num_data,0,point_num_data_size,stream));


//     checkCudaErrors(generateVoxels_launch(
//           pointCloud, pointNum,
//           min_x_range_, max_x_range_,
//           min_y_range_, max_y_range_,
//           min_z_range_, max_z_range_,
//           voxel_x_size_, voxel_y_size_, voxel_z_size_,
//           grid_size_x_, grid_size_y_, grid_size_z_,
//           coor_to_voxelidx,new_point_features_data,new_point_index_in_voxel_data,
//           coords_data,point_num_in_voxel_data,voxel_num_data, stream));
//     // cluster_center
//     checkCudaErrors(generateAverage_launch(new_point_features_data, new_point_index_in_voxel_data,
//         point_num_in_voxel_data,voxel_num_data, stream));

//     // generate new point_features and point_index_in_voxel
//     checkCudaErrors(generateNewFeature_launch(point_features_data, point_index_in_voxel_data,
//         point_num_in_voxel_data,voxel_num_data,new_point_features_data, new_point_index_in_voxel_data,point_num_data, stream));
//     return 0;
// }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void generateVoxels_random_kernel(float *points, unsigned int* points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size,
        unsigned int *mask, float *voxels)
{
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(point_idx >= *points_size) return;

 
  float4 point = ((float4*)points)[point_idx];

  if(point.x<min_x_range||point.x>=max_x_range
    || point.y<min_y_range||point.y>=max_y_range
    || point.z<min_z_range||point.z>=max_z_range) return;

  int voxel_idx = floorf((point.x - min_x_range)/pillar_x_size);
  int voxel_idy = floorf((point.y - min_y_range)/pillar_y_size);
  unsigned int voxel_index = voxel_idy * grid_x_size
                            + voxel_idx;
//    if (point_idx == 14)
//   {
//     printf("point:%f,%f,%f,%f,%d,%d %d\n",point.x,point.y,point.z,point.w,voxel_idx,voxel_idy );
//   }


  unsigned int point_id = atomicAdd(&(mask[voxel_index]), 1);

  if(point_id >= POINTS_NUM_PER_VOXEL) return;
  float *address = voxels + (voxel_index*POINTS_NUM_PER_VOXEL + point_id)*4;
  atomicExch(address+0, point.x);
  atomicExch(address+1, point.y);
  atomicExch(address+2, point.z);
  atomicExch(address+3, point.w);
}

cudaError_t generateVoxels_random_launch(float *points, unsigned int* points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size,
        unsigned int *mask, float *voxels,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((MAX_POINTS_NUM+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  generateVoxels_random_kernel<<<blocks, threads, 0, stream>>>
    (points, points_size,
        min_x_range, max_x_range,
        min_y_range, max_y_range,
        min_z_range, max_z_range,
        pillar_x_size, pillar_y_size, pillar_z_size,
        grid_y_size, grid_x_size,
        mask, voxels);
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void generateBaseFeatures_kernel(unsigned int *mask, float *voxels,
        int grid_y_size, int grid_x_size,
        unsigned int *pillar_num,
        float *voxel_features,
        unsigned int *voxel_num,
        unsigned int *voxel_idxs)
{
  unsigned int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int voxel_idy = blockIdx.y * blockDim.y + threadIdx.y;

  if(voxel_idx >= grid_x_size ||voxel_idy >= grid_y_size) return;

  unsigned int voxel_index = voxel_idy * grid_x_size
                           + voxel_idx;
  unsigned int count = mask[voxel_index];
  if( !(count>0) ) return;
  count = count<POINTS_NUM_PER_VOXEL?count:POINTS_NUM_PER_VOXEL;

  unsigned int current_pillarId = 0;
  current_pillarId = atomicAdd(pillar_num, 1);

  voxel_num[current_pillarId] = count;

  uint4 idx = {0, 0, voxel_idy, voxel_idx};
  ((uint4*)voxel_idxs)[current_pillarId] = idx;

  for (int i=0; i<count; i++){
    int inIndex = voxel_index*POINTS_NUM_PER_VOXEL + i;
    int outIndex = current_pillarId*POINTS_NUM_PER_VOXEL + i;
    ((float4*)voxel_features)[outIndex] = ((float4*)voxels)[inIndex];
  }

  // clear buffer for next infer
  atomicExch(mask + voxel_index, 0);
}

// create 4 channels
cudaError_t generateBaseFeatures_launch(unsigned int *mask, float *voxels,
        int grid_y_size, int grid_x_size,
        unsigned int *pillar_num,
        float *voxel_features,
        unsigned int *voxel_num,
        unsigned int *voxel_idxs,
        cudaStream_t stream)
{
  dim3 threads = {32,32};
  dim3 blocks = {(grid_x_size + threads.x -1)/threads.x,
                 (grid_y_size + threads.y -1)/threads.y};

  generateBaseFeatures_kernel<<<blocks, threads, 0, stream>>>
      (mask, voxels, grid_y_size, grid_x_size,
       pillar_num,
       voxel_features,
       voxel_num,
       voxel_idxs);
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void generateFeatures_kernel(
       float *voxel_data, unsigned int* voxel_num,
       unsigned int *point_num_in_pillar_data,
        float *point_features_data,
        unsigned int * point_index_in_pillar_data, unsigned int* point_num,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float voxel_x_size, float voxel_y_size, float voxel_z_size
        )
{
    unsigned int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxel_idx >=*voxel_num) return;

    int point_num_in_pillar = point_num_in_pillar_data[voxel_idx];


    float cluster_center_x = 0;
    float cluster_center_y = 0;
    float cluster_center_z = 0;
  
    for(int i=0;i<point_num_in_pillar;i++)
    {
        float x = voxel_data[voxel_idx*POINTS_NUM_PER_VOXEL*4+i*4+0];
        float y = voxel_data[voxel_idx*POINTS_NUM_PER_VOXEL*4+i*4+1];
        float z = voxel_data[voxel_idx*POINTS_NUM_PER_VOXEL*4+i*4+2];
        cluster_center_x +=  x;
        cluster_center_y +=  y;
        cluster_center_z += z;
    }
    cluster_center_x = cluster_center_x / point_num_in_pillar;
    cluster_center_y = cluster_center_y / point_num_in_pillar;
    cluster_center_z = cluster_center_z / point_num_in_pillar;

    for(int i=0;i<point_num_in_pillar;i++)
    {   
        // save point_index_in_pillar
        unsigned int point_index = atomicAdd(point_num,1);
        point_index_in_pillar_data[voxel_idx*POINTS_NUM_PER_VOXEL+i] = point_index;
        
        float x = voxel_data[voxel_idx*POINTS_NUM_PER_VOXEL*4+i*4+0];
        float y = voxel_data[voxel_idx*POINTS_NUM_PER_VOXEL*4+i*4+1];
        float z = voxel_data[voxel_idx*POINTS_NUM_PER_VOXEL*4+i*4+2];
        float intensity = voxel_data[voxel_idx*POINTS_NUM_PER_VOXEL*4+i*4+3];
        
        // put xyzi to features
        point_features_data[point_index*FEATURES_NUM+0] = x;
        point_features_data[point_index*FEATURES_NUM+1] = y;
        point_features_data[point_index*FEATURES_NUM+2] = z;
        point_features_data[point_index*FEATURES_NUM+3] = intensity;


        int index_x = floorf((x - min_x_range) / voxel_x_size);
        int index_y = floorf((y - min_y_range) / voxel_y_size);
        int index_z = floorf((z - min_z_range) / voxel_z_size);


        float f_center_x =  x  - ((index_x+0.5) * voxel_x_size + min_x_range);
        float f_center_y = y - ((index_y+0.5) * voxel_y_size + min_y_range);
        float f_center_z = z - ((index_z+0.5) * voxel_z_size + min_z_range);

        // put f_center to features
        point_features_data[point_index*FEATURES_NUM+7] = f_center_x;
        point_features_data[point_index*FEATURES_NUM+8] = f_center_y;
        point_features_data[point_index*FEATURES_NUM+9] = f_center_z;

        // put cluster_center to features
        point_features_data[point_index*FEATURES_NUM+4] = x - cluster_center_x;
        point_features_data[point_index*FEATURES_NUM+5] = y - cluster_center_y;
        point_features_data[point_index*FEATURES_NUM+6] = z - cluster_center_z;

    }
    
}  


cudaError_t generateFeatures_launch(float *voxels_data, unsigned int *voxel_num,
        unsigned int *point_num_in_pillar_data,
          float *point_features_data,
          unsigned int * point_index_in_pillar_data, unsigned int* point_num,
           float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float voxel_x_size, float voxel_y_size, float voxel_z_size,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_PILLARS_NUM+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  generateFeatures_kernel<<<blocks, threads, 0, stream>>>
       (voxels_data, voxel_num,
       point_num_in_pillar_data,
        point_features_data,point_index_in_pillar_data,point_num,
         min_x_range, max_x_range,
        min_y_range, max_y_range,
        min_z_range, max_z_range,
        voxel_x_size, voxel_y_size, voxel_z_size
        );
  cudaError_t err = cudaGetLastError();
  return err;
}


int Points2FeaturesPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];
    // int maxNumPoints = inputDesc[0].dims.d[1];

    //TRT-input
    // std::cout << "voxelgenerator batch_size: " << batchSize << std::endl;
    float * pointCloud = const_cast<float *>((const float *)inputs[0]);
    unsigned int* pointNum = const_cast<unsigned int *>((const unsigned int *)inputs[1]);

    //TRT-output
    float *point_features_data = (float *)(outputs[0]);  // 1 max_point_num 10
    unsigned int *point_index_in_voxel_data = (unsigned int *)(outputs[1]);  // 1 max_pillars_num 32
    unsigned int *coords_data = (unsigned int *)(outputs[2]);  // 1 max_pillars_num 4
    unsigned int *point_num_in_voxel_data = (unsigned int *)(outputs[3]);  // 1 max_pillars_num 1
    unsigned int *pillar_num_data = (unsigned int *)(outputs[4]);  //  valid pillar num
    unsigned int *point_num_data = (unsigned int*)(outputs[5]);  // valid point_num

    // unsigned int *params_data = (unsigned int *)(outputs[2]);
    int dense_voxel_num = grid_size_z_ * grid_size_y_ * grid_size_x_;
    // std::cout << grid_size_x_ << " " << grid_size_y_ << " " << grid_size_z_ << std::endl;
    size_t mask_size = batchSize * dense_voxel_num * sizeof(unsigned int);
    size_t global_voxels_size = batchSize * dense_voxel_num * max_num_points_per_voxel_ * point_feature_num_ * sizeof(float);
    size_t voxels_size = batchSize * max_pillars_num_ * max_num_points_per_voxel_ * point_feature_num_ * sizeof(float);
    
    size_t workspaces[3];
    workspaces[0] = mask_size;
    workspaces[1] = global_voxels_size;
    workspaces[2] = voxels_size;
    size_t total_workspace = calculateTotalWorkspaceSize(workspaces, 3);

    unsigned int* mask = static_cast<unsigned int*>(workspace);
   
    float* global_voxels_data = reinterpret_cast<float*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(mask), mask_size));
    float* voxels_data = reinterpret_cast<float*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(global_voxels_data), global_voxels_size));
    
    // Initialize workspace memory
    checkCudaErrors(cudaMemsetAsync(mask, 0, total_workspace, stream)); // total_workspace
    
    // initialize output
    unsigned int point_features_data_size = batchSize * max_points_num_voxel_filter_  * feature_num_ * sizeof(float);
    unsigned int point_index_in_voxel_data_size = batchSize * max_pillars_num_ * max_num_points_per_voxel_ * sizeof(unsigned int);
    unsigned int coords_data_size = batchSize * max_pillars_num_ * 4 * sizeof(unsigned int);
    unsigned int point_num_in_voxel_data_size = batchSize * max_pillars_num_ * 1 * sizeof(unsigned int);
    unsigned int pillar_num_data_size = batchSize * sizeof(unsigned int);
    unsigned int point_num_data_size = batchSize * sizeof(unsigned int);
    
    checkCudaErrors(cudaMemsetAsync(point_features_data, 0, point_features_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(point_index_in_voxel_data,0,point_index_in_voxel_data_size,stream));
    checkCudaErrors(cudaMemsetAsync(coords_data, 0, coords_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(point_num_in_voxel_data, 0, point_num_in_voxel_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(pillar_num_data, 0, pillar_num_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(point_num_data,0,point_num_data_size,stream));
    // std::cout << batchSize << std::endl;
    // std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << *pointNum << std::endl;
    checkCudaErrors(generateVoxels_random_launch(
        pointCloud, pointNum,
        min_x_range_,max_x_range_,
        min_y_range_, max_y_range_,
        min_z_range_, max_z_range_,
        voxel_x_size_, voxel_y_size_, voxel_z_size_,
        grid_size_y_,grid_size_x_,mask,global_voxels_data,stream
    ));

    checkCudaErrors(generateBaseFeatures_launch(
        mask, global_voxels_data,
        grid_size_y_,grid_size_x_,
        pillar_num_data,
        voxels_data,
        point_num_in_voxel_data,
        coords_data,stream));


    checkCudaErrors(generateFeatures_launch(
        voxels_data, pillar_num_data,
        point_num_in_voxel_data,
        point_features_data,point_index_in_voxel_data,point_num_data,
         min_x_range_, max_x_range_,
        min_y_range_, max_y_range_,
        min_z_range_, max_z_range_,
        voxel_x_size_, voxel_y_size_, voxel_z_size_,
          stream));
    // // cluster_center
    // checkCudaErrors(generateAverage_launch(new_point_features_data, new_point_index_in_voxel_data,
    //     point_num_in_voxel_data,voxel_num_data, stream));

    // // generate new point_features and point_index_in_voxel
    // checkCudaErrors(generateNewFeature_launch(point_features_data, point_index_in_voxel_data,
    //     point_num_in_voxel_data,voxel_num_data,new_point_features_data, new_point_index_in_voxel_data,point_num_data, stream));
    return 0;
}







////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

nvinfer1::DataType Points2FeaturesPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    if(index == 0)
      return inputTypes[0];
    return inputTypes[1];
}

const char* Points2FeaturesPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* Points2FeaturesPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int Points2FeaturesPlugin::getNbOutputs() const noexcept
{
    return 6;
}

int Points2FeaturesPlugin::initialize() noexcept
{
    return 0;
}

void Points2FeaturesPlugin::terminate() noexcept
{
}

size_t Points2FeaturesPlugin::getSerializationSize() const noexcept
{
    return  9 * sizeof(float) + 9 * sizeof(int);
}

void Points2FeaturesPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, max_points_num_);
    writeToBuffer<int>(d, max_points_num_voxel_filter_);
    writeToBuffer<int>(d, max_pillars_num_);
    writeToBuffer<int>(d, point_feature_num_);
    writeToBuffer<int>(d, feature_num_);
    writeToBuffer<int>(d, max_num_points_per_voxel_);

    writeToBuffer<float>(d, min_x_range_);
    writeToBuffer<float>(d, max_x_range_);
    writeToBuffer<float>(d, min_y_range_);
    writeToBuffer<float>(d, max_y_range_);
    writeToBuffer<float>(d, min_z_range_);
    writeToBuffer<float>(d, max_z_range_);
    writeToBuffer<float>(d, voxel_x_size_);
    writeToBuffer<float>(d, voxel_y_size_);
    writeToBuffer<float>(d, voxel_z_size_);

    writeToBuffer<int>(d, grid_size_x_);
    writeToBuffer<int>(d, grid_size_y_);
    writeToBuffer<int>(d, grid_size_z_);
}

void Points2FeaturesPlugin::destroy() noexcept
{
    delete this;
}

void Points2FeaturesPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* Points2FeaturesPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}


Points2FeaturesPluginCreator::Points2FeaturesPluginCreator()
{
    
    mPluginAttributes.clear();

    // std::cout <<  *max_num_points_per_voxel_ptr << std::endl;
    mPluginAttributes.emplace_back(PluginField("max_points_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_points_num_voxel_filter", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_pillars_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("point_feature_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("feature_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_num_points_per_voxel", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("point_cloud_range", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("voxel_size", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("grid_size", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* Points2FeaturesPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* Points2FeaturesPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* Points2FeaturesPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* Points2FeaturesPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int max_points_num = 0;
    int max_points_num_voxel_filter = 0;
    int max_pillars_num = 0;
    int point_feature_num = 0;
    int feature_num = 0;
    int max_num_points_per_voxel = 0;
    float point_cloud_range[6] = {0.0f};
    float voxel_size[3] = {0.0f};
    int grid_size[3] = {0};
    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "max_points_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_points_num = d[0];
        }
        else  if (!strcmp(attr_name, "max_points_num_voxel_filter"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_points_num_voxel_filter = d[0];
        }
        else if (!strcmp(attr_name, "max_pillars_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_pillars_num = d[0];
        }
        else if (!strcmp(attr_name, "point_feature_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            point_feature_num = d[0];
        }
            else if (!strcmp(attr_name, "feature_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            feature_num = d[0];
        }
            else if (!strcmp(attr_name, "max_num_points_per_voxel"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_num_points_per_voxel = d[0];
        }
        else if (!strcmp(attr_name, "point_cloud_range"))
        {
            const float* d = static_cast<const float*>(fields[i].data);
            point_cloud_range[0] = d[0];
            point_cloud_range[1] = d[1];
            point_cloud_range[2] = d[2];
            point_cloud_range[3] = d[3];
            point_cloud_range[4] = d[4];
            point_cloud_range[5] = d[5];
        }
        else if (!strcmp(attr_name, "voxel_size"))
        {
            const float* d = static_cast<const float*>(fields[i].data);
            voxel_size[0] = d[0];
            voxel_size[1] = d[1];
            voxel_size[2] = d[2];
        }
          else if (!strcmp(attr_name, "grid_size"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            grid_size[0] = d[0];
            grid_size[1] = d[1];
            grid_size[2] = d[2];
        }
    }
    // std::cout << max_voxels << " " << max_points << " " <<voxel_feature_num << " " << point_cloud_range[0] << " " << point_cloud_range[1] << " "
    // << point_cloud_range[2] << " "<< point_cloud_range[3] << " " << point_cloud_range[4] << " " << point_cloud_range[5] << " " << voxel_size[0] << " "
    // << voxel_size[1] << " " << voxel_size[2] << std::endl;
    std::cout <<  max_points_num  << " " << max_points_num_voxel_filter << "  " << max_pillars_num << " " << point_feature_num << " " << feature_num << " " << max_num_points_per_voxel << " " << 
    point_cloud_range[0] << " " << point_cloud_range[3] << " " << point_cloud_range[1] << " " << point_cloud_range[4] << " " << point_cloud_range[2] << " " << point_cloud_range[5] << " " << 
    voxel_size[0] << " " << voxel_size[1] << " " << voxel_size[2] << " " << grid_size[0] << " " << grid_size[1] << " " << grid_size[2] << std::endl;
    IPluginV2DynamicExt* plugin = new Points2FeaturesPlugin(max_points_num, max_points_num_voxel_filter, max_pillars_num,point_feature_num, feature_num, max_num_points_per_voxel, point_cloud_range[0],
        point_cloud_range[3], point_cloud_range[1], point_cloud_range[4], point_cloud_range[2],
        point_cloud_range[5], voxel_size[0], voxel_size[1], voxel_size[2],grid_size[0],grid_size[1],grid_size[2]);
    return plugin;
}

IPluginV2* Points2FeaturesPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new Points2FeaturesPlugin(serialData, serialLength);
}

void Points2FeaturesPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* Points2FeaturesPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

Points2FeaturesPluginCreator::~Points2FeaturesPluginCreator()
{
}