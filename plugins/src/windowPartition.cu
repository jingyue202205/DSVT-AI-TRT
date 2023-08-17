#include "windowPartition.h"

using namespace nvinfer1;
using nvinfer1::WindowPartitionPlugin;
using nvinfer1::WindowPartitionPluginCreator;
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
static const char* PLUGIN_NAME{"WindowPartitionPlugin"};

// Static class fields initialization
PluginFieldCollection WindowPartitionPluginCreator::mFC{};
std::vector<PluginField> WindowPartitionPluginCreator::mPluginAttributes;

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
WindowPartitionPlugin::WindowPartitionPlugin(
               int sparse_shape_x, int sparse_shape_y, int sparse_shape_z, 
                                            int win_shape_x, int win_shape_y, int win_shape_z, int shift_x, int shift_y, int shift_z,int max_win_num, int max_voxel_num_per_win
) :  sparse_shape_x_(sparse_shape_x), 
    sparse_shape_y_(sparse_shape_y), sparse_shape_z_(sparse_shape_z), 
    win_shape_x_(win_shape_x), win_shape_y_(win_shape_y), win_shape_z_(win_shape_z), 
    shift_x_(shift_x),  shift_y_(shift_y), shift_z_(shift_z),max_win_num_(max_win_num),max_voxel_num_per_win_(max_voxel_num_per_win)
{
}

WindowPartitionPlugin::WindowPartitionPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    sparse_shape_x_ = readFromBuffer<int>(d);
    sparse_shape_y_ = readFromBuffer<int>(d);
    sparse_shape_z_ = readFromBuffer<int>(d);
    win_shape_x_ = readFromBuffer<int>(d);
    win_shape_y_ = readFromBuffer<int>(d);
    win_shape_z_ = readFromBuffer<int>(d);
    shift_x_ = readFromBuffer<int>(d);
    shift_y_ = readFromBuffer<int>(d);
    shift_z_ = readFromBuffer<int>(d);
    max_win_num_ = readFromBuffer<int>(d);
    max_voxel_num_per_win_ = readFromBuffer<int>(d);
}

IPluginV2DynamicExt* WindowPartitionPlugin::clone() const noexcept
{
    auto* plugin = new WindowPartitionPlugin( sparse_shape_x_,sparse_shape_y_,sparse_shape_z_,
                                                            win_shape_x_,win_shape_y_,win_shape_z_,shift_x_,shift_y_,shift_z_,max_win_num_,max_voxel_num_per_win_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs WindowPartitionPlugin::getOutputDimensions(
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
        dim0.d[1] = exprBuilder.constant(max_win_num_);
        dim0.d[2] = exprBuilder.constant(max_voxel_num_per_win_);
        return dim0; // global_index 1 3200 600
    }
    if(outputIndex == 1){
        // std::cout << "batch_size: " << batch_size->getConstantValue() << "  voxel_num: " << voxelNum_ << " featurennum_: " << 4 << std::endl;
        nvinfer1::DimsExprs dim1{};
        dim1.nbDims = 4;
        dim1.d[0] = batch_size;
        dim1.d[1] = exprBuilder.constant(max_win_num_);
        dim1.d[2] = exprBuilder.constant(max_voxel_num_per_win_);
        dim1.d[3] = exprBuilder.constant(3);
        return dim1; // coors_in_win 1 3200 600 3
    }
      if(outputIndex == 2){
        // std::cout << "batch_size: " << batch_size->getConstantValue() << "  voxel_num: " << voxelNum_ << " featurennum_: " << 4 << std::endl;
        nvinfer1::DimsExprs dim2{};
        dim2.nbDims = 2;
        dim2.d[0] = batch_size;
        dim2.d[1] = exprBuilder.constant(max_win_num_);
        return dim2; // voxel _num_in_win  1 3200
    }
     if(outputIndex == 3)
    {   
        // std::cout << "batch_size: " << batch_size->getConstantValue() << std::endl;
        nvinfer1::DimsExprs dim3{};
        dim3.nbDims = 1;
        dim3.d[0] = batch_size;
        return dim3;  // valid win_num
    }
      if(outputIndex == 4)
    {   
        // std::cout << "batch_size: " << batch_size->getConstantValue() << std::endl;
        nvinfer1::DimsExprs dim4{};
        dim4.nbDims = 3;
        dim4.d[0] = batch_size;
        dim4.d[1] = exprBuilder.constant(MAX_PILLARS_NUM);
        dim4.d[2] = exprBuilder.constant(3);
        return dim4;  // coors_in_win_2d  1 MAX_PILLARS_NUM  3   
    }
      if(outputIndex == 5)
    {   
        // std::cout << "batch_size: " << batch_size->getConstantValue() << std::endl;
        nvinfer1::DimsExprs dim5{};
        dim5.nbDims = 3;
        dim5.d[0] = batch_size;
        dim5.d[1] = exprBuilder.constant(MAX_PILLARS_NUM);
        dim5.d[2] = exprBuilder.constant(2);
        return dim5;  // coors_in_win_x_y  1 MAX_PILLARS_NUM  2   
    }
}

bool WindowPartitionPlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // PLUGIN_ASSERT(nbInputs == 2);
    // PLUGIN_ASSERT(nbOutputs == 2);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       // coors 5504 * 4 
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)       // valid voxel Num
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)       //global_index 1 3200 600
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)       //coors_in_win 1 3200 600 3
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 4)       //voxel _num_in_win 1 3200
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
     if (pos == 5)       //valid win_num 1 
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 6)       //coors_in_win_2d  1 7000 3
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
        if (pos == 7)       //coors_in_win_x_y  1 7000 2
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    return false;
}

void WindowPartitionPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t WindowPartitionPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    int batchSize = inputs[0].dims.d[0];
     int max_num_win_x = int(ceilf(sparse_shape_x_ / win_shape_x_) + 1);
    int max_num_win_y =  int(ceilf(sparse_shape_y_ / win_shape_y_) + 1);
    int max_num_win_z =  int(ceilf(sparse_shape_z_ / win_shape_z_) + 1);
    int dense_win_num =  max_num_win_x * max_num_win_y * max_num_win_z;
    size_t coor_to_winidx_size = batchSize * dense_win_num * 2* sizeof(unsigned int);

    size_t workspaces[1];
    workspaces[0] = coor_to_winidx_size;

    return  calculateTotalWorkspaceSize(workspaces, 1);
}

__device__ void cuda_sleep(int64_t num_cycles)
{
    int64_t cycles = 0;
    int64_t start = clock64();
    while(cycles < num_cycles)
    {
        cycles = clock64() - start;
    }
}

__global__ void splitWindow_kernel(
     unsigned int *voxel_coors, unsigned int* valid_voxel_num,
        unsigned int *global_index_data,unsigned int * coors_in_win_data, unsigned int* coors_in_win_2d_data, float *coors_in_win_x_y_data,
        unsigned int * voxel_num_in_win_data, 
        unsigned int *win_num_data, unsigned int* coor_to_winidx,
        int shift_x ,int shift_y, int shift_z, unsigned int max_num_win_x, unsigned int max_num_win_y, unsigned int max_num_win_z,
        int win_shape_x, int win_shape_y, int win_shape_z)
{
    unsigned int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxel_idx >= *valid_voxel_num) return;
    // if(voxel_idx == *valid_voxel_num - 1)
    //     printf("voxel_idx: %d,%d\n",voxel_idx,*valid_voxel_num);
    uint4 coor = ((uint4*)voxel_coors)[voxel_idx];
    
    unsigned int shifted_coor_x = coor.w + shift_x;
    unsigned int shifted_coor_y = coor.z + shift_y;
    unsigned int shifted_coor_z = coor.y + shift_z;

    unsigned int win_coor_x = int(floorf(shifted_coor_x / win_shape_x));
    unsigned int win_coor_y = int(floorf(shifted_coor_y/win_shape_y));
    unsigned int win_coor_z = int(floorf(shifted_coor_z / win_shape_z));
    
    // printf("max_num_win:%d,%d,%d\n",max_num_win_x,max_num_win_y,max_num_win_z);
    unsigned int win_index = win_coor_z * (max_num_win_y * max_num_win_x) + win_coor_y * max_num_win_x + win_coor_x;
    //  printf("win_index:%d\n",win_index);     
    // printf("shifted_coor: %d,%d,%d,%d,%d,%d\n",shifted_coor_x,shifted_coor_y,shifted_coor_z,shift_x,shift_y,shift_z);                  
    unsigned int voxel_id_in_win = atomicAdd(coor_to_winidx+win_index*2,1); // voxel_id in win
    if(voxel_id_in_win >= MAX_VOXEL_NUM_PER_WIN) return;    
    
    unsigned int current_win_id = 0;
    if (voxel_id_in_win == 0)
    {
        //save current_win_id
        current_win_id = atomicAdd(win_num_data,1);
        #if  0
        if(current_voxelid == 0)
        {
            printf("current_voxelid:%d,%f,%f,%f,%d,%d,%d,%d\n",current_voxelid,point.x,point.y,point.z,index_x,index_y,index_z,voxel_index * 2 + 1);
        }
        #endif
        //save current_win_id
        unsigned int *current_win_id_address = coor_to_winidx + win_index * 2 + 1;
        atomicExch(current_win_id_address,current_win_id);

    }
   else{
        // if(current_voxelid == 0 && )
        // __nanosleep(100);
        current_win_id = coor_to_winidx[win_index*2+1];
        if(current_win_id == 0)
        {
            cuda_sleep(300000); //10000000
            current_win_id = coor_to_winidx[win_index*2+1];
        }
        #if 0
         if(current_voxelid == 0)
        {
            printf("current_voxelid:%d,%f,%f,%f,%d,%d,%d,%d,%d\n",current_voxelid,point.x,point.y,point.z,index_x,index_y,index_z,point_id,voxel_index*2+1);
        }
        #endif
    }

     //voxel_num_in_win_data 
    unsigned int *voxel_num_in_win_data_address = voxel_num_in_win_data + current_win_id;
    unsigned int total_num_in_win = *(coor_to_winidx+win_index*2);
    if(total_num_in_win > MAX_VOXEL_NUM_PER_WIN)
        total_num_in_win = MAX_VOXEL_NUM_PER_WIN;
    atomicExch(voxel_num_in_win_data_address,total_num_in_win);

    // save global index 
    unsigned int * global_index_data_addr = global_index_data + current_win_id * MAX_VOXEL_NUM_PER_WIN;
    atomicExch(global_index_data_addr+voxel_id_in_win, voxel_idx);

    // save coors_in_win
    int coor_in_win_x = shifted_coor_x  % win_shape_x;
    int coor_in_win_y = shifted_coor_y % win_shape_y;
    int coor_in_win_z = shifted_coor_z % win_shape_z;

     //save coord
    *(coors_in_win_data+current_win_id*MAX_VOXEL_NUM_PER_WIN*3+voxel_id_in_win*3+0) = coor_in_win_z;
    *(coors_in_win_data+current_win_id*MAX_VOXEL_NUM_PER_WIN*3+voxel_id_in_win*3+1) = coor_in_win_y;
    *(coors_in_win_data+current_win_id*MAX_VOXEL_NUM_PER_WIN*3+voxel_id_in_win*3+2) = coor_in_win_x;

    // save coord 2d
    *(coors_in_win_2d_data+voxel_idx*3+0) = coor_in_win_z;
    *(coors_in_win_2d_data+voxel_idx*3+1) = coor_in_win_y;
    *(coors_in_win_2d_data+voxel_idx*3+2) = coor_in_win_x;

    // save coord x_y
    *(coors_in_win_x_y_data+voxel_idx*2+0) =  float(coor_in_win_x) - float(win_shape_x) / 2;
    *(coors_in_win_x_y_data+voxel_idx*2+1) = float(coor_in_win_y) - float(win_shape_y) / 2;
    
    #if 0
    if (index_z == 32 && index_y == 731 && index_x == 95)
    {
        printf("point: %f,%f,%f,%f %d\n",point.x,point.y,point.z,point.w,point_id);
        printf("adress: %f,%f,%f,%f\n",*address,*(address+1),*(address+2),*(address+3));
        printf("fsfsfsfsf current_voxelid: %d\n",current_voxelid);
        printf("coor_to_voxelidx[voxel_index*2]: %d\n",atomicAdd(coor_to_voxelidx+voxel_index*2,0));
        printf("point_id: %d\n",point_id);
        printf("num_points_per_voxel_address: %d\n",*num_points_per_voxel_address);
    }
    #endif
}
  

cudaError_t splitWindow_launch(unsigned int *voxel_coors, unsigned int* valid_voxel_num,
        unsigned int *global_index_data,unsigned int * coors_in_win_data, unsigned int* coors_in_win_2d_data,float* coors_in_win_x_y_data,
        unsigned int * voxel_num_in_win_data, 
        unsigned int *win_num_data, unsigned int*coor_to_winidx, int shift_x, int shift_y, int shift_z,
        unsigned int max_num_win_x, unsigned int max_num_win_y,unsigned int max_num_win_z,
        int win_shape_x,int win_shape_y, int win_shape_z,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_PILLARS_NUM+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  splitWindow_kernel<<<blocks, threads, 0, stream>>>(
       voxel_coors,valid_voxel_num, global_index_data, coors_in_win_data,coors_in_win_2d_data,coors_in_win_x_y_data, voxel_num_in_win_data, win_num_data,coor_to_winidx,
       shift_x,shift_y, shift_z, max_num_win_x,max_num_win_y,max_num_win_z,win_shape_x,win_shape_y,win_shape_z);
  cudaError_t err = cudaGetLastError();
  return err;
}


int WindowPartitionPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];
    // int maxNumPoints = inputDesc[0].dims.d[1];

    //TRT-input
    // std::cout << "voxelgenerator batch_size: " << batchSize << std::endl;
    unsigned int * voxel_coors = const_cast<unsigned int *>((const unsigned int *)inputs[0]);
    unsigned int* valid_voxel_num = const_cast<unsigned int *>((const unsigned int *)inputs[1]);

    //TRT-output
    unsigned int  *global_index_data = (unsigned int *)(outputs[0]);  // 1 max_win_num  max_voxel_num_per_win
    unsigned int *coors_in_win_data = (unsigned int *)(outputs[1]);  // 1 max_win_num max_voxel_num_per_win 3
    unsigned int * voxel_num_in_win_data = (unsigned int *)(outputs[2]);  // 1 max_win_num 
    unsigned int *win_num_data = (unsigned int *)(outputs[3]);  //  valid win num
    unsigned int *coors_in_win_2d_data = (unsigned int*)(outputs[4]); // 1 max_voxel_num  3
    float *coors_in_win_x_y_data = (float*)(outputs[5]); // 1 max_voxel_num 2
   
    // // unsigned int *params_data = (unsigned int *)(outputs[2]);
    int max_num_win_x = int(ceilf(sparse_shape_x_ / win_shape_x_) + 1);
    int max_num_win_y =  int(ceilf(sparse_shape_y_ / win_shape_y_) + 1);
    int max_num_win_z =  int(ceilf(sparse_shape_z_ / win_shape_z_) + 1);
    int dense_win_num =  max_num_win_x * max_num_win_y * max_num_win_z;
    size_t coor_to_winidx_size = batchSize * dense_win_num * 2* sizeof(unsigned int);
   
    
    size_t workspaces[1];
    workspaces[0] = coor_to_winidx_size;
    // workspaces[1] = new_point_features_size;
    // workspaces[2] = new_point_index_in_voxel_size;
    size_t total_workspace = calculateTotalWorkspaceSize(workspaces, 1);

    unsigned int* coor_to_winidx = static_cast<unsigned int*>(workspace);
    // float* new_point_features_data = reinterpret_cast<float*>(
    //     nextWorkspacePtr(reinterpret_cast<int8_t*>(coor_to_voxelidx), coor_to_voxelidx_size));
    // unsigned int* new_point_index_in_voxel_data = reinterpret_cast<unsigned int*>(
    //     nextWorkspacePtr(reinterpret_cast<int8_t*>(new_point_features_data), new_point_features_size));
    
    // // Initialize workspace memory
    checkCudaErrors(cudaMemsetAsync(coor_to_winidx, 0, total_workspace, stream)); // total_workspace
    
    // initialize output
    unsigned int global_index_data_size = batchSize * max_win_num_  * max_voxel_num_per_win_ * sizeof(unsigned int);
    unsigned int coors_in_win_data_size = batchSize * max_win_num_ * max_voxel_num_per_win_ *  3 * sizeof(unsigned int);
    unsigned int voxel_num_in_win_data_size = batchSize * max_win_num_ * sizeof(unsigned int);
    unsigned int win_num_data_size = batchSize * sizeof(unsigned int);
    unsigned int coors_in_win_2d_data_size = batchSize * MAX_PILLARS_NUM * 3 * sizeof(unsigned int);
    unsigned int coors_in_win_x_y_data_size = batchSize * MAX_PILLARS_NUM * 2 * sizeof(float);
   
    
    checkCudaErrors(cudaMemsetAsync(global_index_data, 0, global_index_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(coors_in_win_data,0,coors_in_win_data_size,stream));
    checkCudaErrors(cudaMemsetAsync(voxel_num_in_win_data, 0, voxel_num_in_win_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(win_num_data, 0, win_num_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(coors_in_win_2d_data, 0, coors_in_win_2d_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(coors_in_win_x_y_data, 0, coors_in_win_x_y_data_size, stream));

    // std::cout << sparse_shape_x_ << " " << sparse_shape_y_ << " " << sparse_shape_z_ << std::endl;
    // std::cout << shift_x_ << " "  << shift_y_ << " " << shift_z_ << " " << max_num_win_x << " " <<  max_num_win_y << " " <<  max_num_win_z << " " << win_shape_x_ << " " << win_shape_y_ << " " <<  win_shape_z_ << std::endl;
    checkCudaErrors(splitWindow_launch(
          voxel_coors,valid_voxel_num, global_index_data, coors_in_win_data, coors_in_win_2d_data,coors_in_win_x_y_data,voxel_num_in_win_data, win_num_data, coor_to_winidx, 
          shift_x_,shift_y_,shift_z_, max_num_win_x,max_num_win_y,max_num_win_z,win_shape_x_,win_shape_y_,win_shape_z_,
          stream));
    return 0;
}

nvinfer1::DataType WindowPartitionPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    if(index == 5)
    {
        return  nvinfer1::DataType::kFLOAT;
    }
    return inputTypes[0];
}

const char* WindowPartitionPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* WindowPartitionPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int WindowPartitionPlugin::getNbOutputs() const noexcept
{
    return 6;
}

int WindowPartitionPlugin::initialize() noexcept
{
    return 0;
}

void WindowPartitionPlugin::terminate() noexcept
{
}

size_t WindowPartitionPlugin::getSerializationSize() const noexcept
{
    return   11 * sizeof(int);
}

void WindowPartitionPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, sparse_shape_x_);
    writeToBuffer<int>(d, sparse_shape_y_);
    writeToBuffer<int>(d, sparse_shape_z_);
    writeToBuffer<int>(d, win_shape_x_);
    writeToBuffer<int>(d, win_shape_y_);
    writeToBuffer<int>(d, win_shape_z_);
    writeToBuffer<int>(d, shift_x_);
    writeToBuffer<int>(d, shift_y_);
    writeToBuffer<int>(d, shift_z_);
    writeToBuffer<int>(d, max_win_num_);
    writeToBuffer<int>(d, max_voxel_num_per_win_);
}

void WindowPartitionPlugin::destroy() noexcept
{
    delete this;
}

void WindowPartitionPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* WindowPartitionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}


WindowPartitionPluginCreator::WindowPartitionPluginCreator()
{
    
    mPluginAttributes.clear();

    // std::cout <<  *max_num_points_per_voxel_ptr << std::endl;
    mPluginAttributes.emplace_back(PluginField("max_win_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_voxel_num_per_win", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("sparse_shape", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("win_shape", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("shift_list", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* WindowPartitionPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* WindowPartitionPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* WindowPartitionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* WindowPartitionPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int max_win_num = 0;
    int max_voxel_num_per_win = 0;
   
    int sparse_shape[3] = {0};
    int win_shape[3] = {0};
    int shift_list[3] = {0};
    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "max_win_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_win_num = d[0];
        }
        else if (!strcmp(attr_name, "max_voxel_num_per_win"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_voxel_num_per_win = d[0];
        }
        else if (!strcmp(attr_name, "sparse_shape"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            sparse_shape[0] = d[0];
            sparse_shape[1] = d[1];
            sparse_shape[2] = d[2];
        }
        else if (!strcmp(attr_name, "win_shape"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            win_shape[0] = d[0];
            win_shape[1] = d[1];
            win_shape[2] = d[2];
        }
          else if (!strcmp(attr_name, "shift_list"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            shift_list[0] = d[0];
            shift_list[1] = d[1];
            shift_list[2] = d[2];
        }
    }
    // std::cout << max_voxels << " " << max_points << " " <<voxel_feature_num << " " << point_cloud_range[0] << " " << point_cloud_range[1] << " "
    // << point_cloud_range[2] << " "<< point_cloud_range[3] << " " << point_cloud_range[4] << " " << point_cloud_range[5] << " " << voxel_size[0] << " "
    // << voxel_size[1] << " " << voxel_size[2] << std::endl;
    std::cout <<  max_win_num  << " " << max_voxel_num_per_win << " " << sparse_shape[0] << " " << sparse_shape[1] << " " << sparse_shape[2] << " " 
     << win_shape[0] << " " << win_shape[1] << " " << win_shape[2] << " " << shift_list[0] << " " << shift_list[1] << " " << shift_list[2] << " " << std::endl;
    IPluginV2DynamicExt* plugin = new WindowPartitionPlugin(sparse_shape[0],sparse_shape[1],sparse_shape[2],win_shape[0],win_shape[1],win_shape[2],shift_list[0],shift_list[1],shift_list[2],
                                           max_win_num,max_voxel_num_per_win );
    return plugin;
}

IPluginV2* WindowPartitionPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new WindowPartitionPlugin(serialData, serialLength);
}

void WindowPartitionPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* WindowPartitionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

WindowPartitionPluginCreator::~WindowPartitionPluginCreator()
{
}