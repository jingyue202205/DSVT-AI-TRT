#include "getSet.h"

using namespace nvinfer1;
using nvinfer1::GetSetPlugin;
using nvinfer1::GetSetPluginCreator;
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
static const char* PLUGIN_NAME{"GetSetPlugin"};

// Static class fields initialization
PluginFieldCollection GetSetPluginCreator::mFC{};
std::vector<PluginField> GetSetPluginCreator::mPluginAttributes;

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
GetSetPlugin::GetSetPlugin(
               int voxel_num_set,int max_win_num, int max_voxel_num_per_win,int win_shape_x,int win_shape_y,int win_shape_z
) :  voxel_num_set_(voxel_num_set), 
    max_win_num_(max_win_num),max_voxel_num_per_win_(max_voxel_num_per_win),
    win_shape_x_(win_shape_x),win_shape_y_(win_shape_y),win_shape_z_(win_shape_z)
{
}

GetSetPlugin::GetSetPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    voxel_num_set_ = readFromBuffer<int>(d);
    max_win_num_ = readFromBuffer<int>(d);
    max_voxel_num_per_win_ = readFromBuffer<int>(d);
    win_shape_x_ = readFromBuffer<int>(d);
    win_shape_y_ = readFromBuffer<int>(d);
    win_shape_z_ = readFromBuffer<int>(d);
}

IPluginV2DynamicExt* GetSetPlugin::clone() const noexcept
{
    auto* plugin = new GetSetPlugin( voxel_num_set_, max_win_num_,max_voxel_num_per_win_,win_shape_x_,win_shape_y_,win_shape_z_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs GetSetPlugin::getOutputDimensions(
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
        dim0.nbDims = 4;
        dim0.d[0] = batch_size;
        dim0.d[1] = exprBuilder.constant(2);
        dim0.d[2] = exprBuilder.constant(max_win_num_);
        dim0.d[3] = exprBuilder.constant(voxel_num_set_);
        return dim0; // global_index 1 2 3200 36
    }
    if(outputIndex == 1){
        // std::cout << "batch_size: " << batch_size->getConstantValue() << "  voxel_num: " << voxelNum_ << " featurennum_: " << 4 << std::endl;
        nvinfer1::DimsExprs dim1{};
        dim1.nbDims = 4;
        dim1.d[0] = batch_size;
        dim1.d[1] =  exprBuilder.constant(2);
        dim1.d[2] = exprBuilder.constant(max_win_num_);
        dim1.d[3] = exprBuilder.constant(voxel_num_set_);
        return dim1; // set_voxel_mask  1 2 3200 36
    }
    if(outputIndex == 2){
       nvinfer1::DimsExprs dim2{};
        dim2.nbDims = 1;
        dim2.d[0] = batch_size;
        return dim2; // valid set num
    }
        if(outputIndex == 3){
       nvinfer1::DimsExprs dim3{};
        dim3.nbDims = 4;
        dim3.d[0] = batch_size;
        dim3.d[1] =  exprBuilder.constant(max_win_num_);
        dim3.d[2] =  exprBuilder.constant(NUM_HEADS);
        dim3.d[3] = exprBuilder.constant(voxel_num_set_);
        return dim3; // 1 max_wim_num 8 32
    }
        if(outputIndex == 4){
       nvinfer1::DimsExprs dim4{};
        dim4.nbDims = 4;
        dim4.d[0] = batch_size;
        dim4.d[1] =  exprBuilder.constant(max_win_num_);
        dim4.d[2] =  exprBuilder.constant(NUM_HEADS);
        dim4.d[3] = exprBuilder.constant(voxel_num_set_);
        return dim4; // 1 max_wim_num 8 32
       }
}

bool GetSetPlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // PLUGIN_ASSERT(nbInputs == 2);
    // PLUGIN_ASSERT(nbOutputs == 2);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       // input global_index  3200 * 600
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)       // input coors_in_win  3200 * 600 * 3
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)       //input voxel_num_in_win  3200
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)       // input valid win
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 4)       //output  2 * 3200 * 36  global_index_in_set
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
     if (pos == 5)       //output  set_voxel_mask  2 * 3200 * 36
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 6)       //output valid set num  
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
      if (pos == 7)       //output 1  3200 8 36  
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
      if (pos == 8)       //output  1  3200 8 36  
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    return false;
}

void GetSetPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t GetSetPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    int batchSize = inputs[0].dims.d[0];
    // workspace buffer
    size_t global_index_sorty_size = batchSize * max_win_num_  * max_voxel_num_per_win_ * sizeof(unsigned int);   // 3200 600  max_win_num = max_set_num
    size_t global_index_sortx_size = batchSize * max_win_num_  * max_voxel_num_per_win_ * sizeof(unsigned int);   //3200 600
    size_t local_index_in_set_size = batchSize * max_win_num_ * voxel_num_set_ * sizeof(unsigned int);   
    size_t  win_index_size =  batchSize * max_win_num_ * sizeof(unsigned int);
    
    size_t workspaces[4];
    workspaces[0] = global_index_sorty_size;
    workspaces[1] = global_index_sortx_size;
    workspaces[2] = local_index_in_set_size;
    workspaces[3] = win_index_size;
    size_t total_workspace = calculateTotalWorkspaceSize(workspaces, 4);

    return  total_workspace;
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

__device__ void swap(unsigned int *a, unsigned int*b)
{
    unsigned int t= *a;
    *a = *b;
    *b = t;
}

__device__ int partition(unsigned int *arr, unsigned  int *coor_array,  int l, int h)
{
    int x = coor_array[h];
    int i = l - 1;

    for(int j = l; j<= h-1; j++)
    {
        if(coor_array[j] <= x)
        {
            i++;
            swap(&coor_array[i],&coor_array[j]);
            swap(&arr[i],&arr[j]);
        }
    }
    swap(&coor_array[i+1],&coor_array[h]);
    swap(&arr[i+1],&arr[h]);
    return (i+1);
}

__device__ void quickSortIterative(unsigned int *arr, unsigned int *coor_array, int l, int h)
{
    // create an auxiliary stack
    int stack[MAX_VOXEL_NUM_PER_WIN];

    // init top of stack
    int top = -1;

    // push init values of l and h to stack
    stack[++top] = l;
    stack[++top] = h;

    while(top>=0)
    {
         h = stack[top--];
         l = stack[top--];
         
         int p = partition(arr,coor_array,l,h);

         if(p-1>l)
         {
            stack[++top] = l;
            stack[++top] = p -1;
         }

         if(p+1 <h)
         {
            stack[++top] = p+1;
            stack[++top] = h;
         }
    }
}
 
__global__ void getLocalIndex_kernel(
        unsigned int *global_index_data,
        unsigned int * voxel_num_in_win_data, 
        unsigned int *win_num_data, unsigned int*local_index_in_set_data, 
        unsigned int *win_index_data, unsigned int *set_num_data, int voxel_num_set)
{
    unsigned int win_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(win_idx >= *win_num_data) return;
    
    int set_num_in_current_win =  int(ceilf(float(voxel_num_in_win_data[win_idx]) / voxel_num_set));

    unsigned int set_idx = atomicAdd(set_num_data,set_num_in_current_win);
    // printf("voxel_num_set:%d\n",voxel_num_set);
    // printf("win_idx:%d,%d,%d,%d,%f\n",win_idx,*set_num_data,voxel_num_in_win_data[win_idx],voxel_num_set,float(voxel_num_in_win_data[win_idx]) / voxel_num_set);
    for(int i=set_idx; i<(set_idx+set_num_in_current_win); i++)
    {
        for(int k=0;k<voxel_num_set;k++)
        {
            int j = i - set_idx;
            int N = voxel_num_in_win_data[win_idx];
             local_index_in_set_data[i*voxel_num_set+k] = int(floorf((j*voxel_num_set+k)*N/voxel_num_set/set_num_in_current_win)) ;   // paper  eq.(3)
        }
        win_index_data[i] = win_idx;
    }
}
 

cudaError_t getLocalIndex_launch(
        unsigned int *global_index_data,
        unsigned int * voxel_num_in_win_data, 
        unsigned int *win_num_data, unsigned int*local_index_in_set_data, unsigned int *win_index_data, unsigned int *set_num_data, int voxel_num_set,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_WIN_NUM+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  getLocalIndex_kernel<<<blocks, threads, 0, stream>>>(
        global_index_data,voxel_num_in_win_data,win_num_data,local_index_in_set_data,win_index_data, set_num_data,voxel_num_set);
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void sortY_kernel(
        unsigned int *global_index_data,
        unsigned int* coors_in_win_data,
        unsigned int * voxel_num_in_win_data, 
        unsigned int *win_num_data, unsigned int* global_index_sorty_data,
        int max_voxel_num_per_win, int win_shape_x,int win_shape_y,int win_shape_z)
{
    unsigned int win_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(win_idx >= *win_num_data) return;

    int voxel_num_in_current_win =  voxel_num_in_win_data[win_idx];

    //copy global_index to global_index_sorty  and get coor_array
    unsigned int coor_array[MAX_VOXEL_NUM_PER_WIN];
    for(int m=0;m<voxel_num_in_current_win;m++)
    {
        global_index_sorty_data[win_idx*max_voxel_num_per_win+m] = global_index_data[win_idx*max_voxel_num_per_win+m];
        coor_array[m] = coors_in_win_data[win_idx*max_voxel_num_per_win*3+m*3+1] * win_shape_x*win_shape_z +  coors_in_win_data[win_idx*max_voxel_num_per_win*3+m*3+2] * win_shape_z  + 
                                coors_in_win_data[win_idx*max_voxel_num_per_win*3+m*3+0] ;
    }
    
    //bubble_sort     https://blog.csdn.net/dongming8886/article/details/123458790
    // int temp1 = 0;
    // int temp2 = 0;
    // for(int i=0;i<voxel_num_in_current_win-1;i++)
    // {
    //     int count = 0;
    //     for(int j=0;j<voxel_num_in_current_win-1-i;j++)
    //     {
    //         int before = coor_array[j] ;
    //         int after = coor_array[j+1] ;
    //         if(before>after)
    //         {
    //             temp1 = global_index_sorty_data[win_idx*max_voxel_num_per_win+j];
    //             global_index_sorty_data[win_idx*max_voxel_num_per_win+j] = global_index_sorty_data[win_idx*max_voxel_num_per_win+j+1];
    //             global_index_sorty_data[win_idx*max_voxel_num_per_win+j+1] = temp1;

    //             temp2 = coor_array[j];
    //             coor_array[j] = coor_array[j+1];
    //             coor_array[j+1] = temp2;
    //             count = 1;
    //         }
    //     }
    //     if(count==0)
    //     {
    //         break;
    //     }
        
    // }

    // quick_sort   https://www.geeksforgeeks.org/iterative-quick-sort/
    quickSortIterative(global_index_sorty_data+win_idx*max_voxel_num_per_win,coor_array,0,voxel_num_in_current_win-1);

}

cudaError_t sortY_launch(
        unsigned int *global_index_data,
        unsigned int *coors_in_win_data,
        unsigned int * voxel_num_in_win_data, 
        unsigned int *win_num_data, unsigned int*global_index_sorty_data,
        int max_voxel_num_per_win,
        int win_shape_x, int win_shape_y,int win_shape_z,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_WIN_NUM+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  sortY_kernel<<<blocks, threads, 0, stream>>>(
        global_index_data,coors_in_win_data, voxel_num_in_win_data,win_num_data,global_index_sorty_data,
        max_voxel_num_per_win,win_shape_x,win_shape_y,win_shape_z);
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void sortX_kernel(
        unsigned int *global_index_data,
        unsigned int* coors_in_win_data,
        unsigned int * voxel_num_in_win_data, 
        unsigned int *win_num_data, unsigned int* global_index_sortx_data,
        int max_voxel_num_per_win, int win_shape_x,int win_shape_y,int win_shape_z)
{
    unsigned int win_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(win_idx >= *win_num_data) return;

    int voxel_num_in_current_win =  voxel_num_in_win_data[win_idx];

     //copy global_index to global_index_sorty  and get coor_array
    unsigned int coor_array[MAX_VOXEL_NUM_PER_WIN];
    for(int m=0;m<voxel_num_in_current_win;m++)
    {
        global_index_sortx_data[win_idx*max_voxel_num_per_win+m] = global_index_data[win_idx*max_voxel_num_per_win+m];
        coor_array[m] = coors_in_win_data[win_idx*max_voxel_num_per_win*3+m*3+2] * win_shape_y*win_shape_z +  coors_in_win_data[win_idx*max_voxel_num_per_win*3+m*3+1] * win_shape_z  + 
                                coors_in_win_data[win_idx*max_voxel_num_per_win*3+m*3+0] ;
    }
    
    //bubble_sort
    // int temp1 = 0;
    // int temp2 = 0;
    // for(int i=0;i<voxel_num_in_current_win-1;i++)
    // {
    //     int count = 0;
    //     for(int j=0;j<voxel_num_in_current_win-1-i;j++)
    //     {
    //         int before = coor_array[j];
    //         int after = coor_array[j+1];
    //         if(before>after)
    //         {
    //             temp1 = global_index_sortx_data[win_idx*max_voxel_num_per_win+j];
    //             global_index_sortx_data[win_idx*max_voxel_num_per_win+j] = global_index_sortx_data[win_idx*max_voxel_num_per_win+j+1];
    //             global_index_sortx_data[win_idx*max_voxel_num_per_win+j+1] = temp1;

    //              temp2 = coor_array[j];
    //             coor_array[j] = coor_array[j+1];
    //             coor_array[j+1] = temp2;
    //             count = 1;
    //         }
    //     }
    //     if(count==0)
    //     {
    //         break;
    //     }
        
    // }
     quickSortIterative(global_index_sortx_data+win_idx*max_voxel_num_per_win,coor_array,0,voxel_num_in_current_win-1);

}

cudaError_t sortX_launch(
        unsigned int *global_index_data,
        unsigned int *coors_in_win_data,
        unsigned int * voxel_num_in_win_data, 
        unsigned int *win_num_data, unsigned int*global_index_sortx_data,
        int max_voxel_num_per_win,
        int win_shape_x, int win_shape_y,int win_shape_z,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_WIN_NUM+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  sortX_kernel<<<blocks, threads, 0, stream>>>(
        global_index_data,coors_in_win_data, voxel_num_in_win_data,win_num_data,global_index_sortx_data,
        max_voxel_num_per_win,win_shape_x,win_shape_y,win_shape_z);
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void useLocalIndexGetSortedGlobalIndex__kernel(
        unsigned int *global_index_in_set_data,
         unsigned int *global_index_sorty_data,
        unsigned int *global_index_sortx_data,
        unsigned int * local_index_in_set_data, 
        unsigned int *win_index_data, unsigned int*set_num_data,
        float * set_voxel_mask_data,
        int voxel_num_set)
{
    unsigned int set_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(set_idx >= *set_num_data) return;

    int win_index = win_index_data[set_idx];
    
    for(int i=0;i<voxel_num_set;i++)
    {
        // sorty   to global index
        global_index_in_set_data[0*MAX_WIN_NUM*voxel_num_set+set_idx*voxel_num_set+i] = 
                                        global_index_sorty_data[win_index*MAX_VOXEL_NUM_PER_WIN+local_index_in_set_data[set_idx*voxel_num_set+i]];
         //sortx to global index
        global_index_in_set_data[1*MAX_WIN_NUM*voxel_num_set+set_idx*voxel_num_set+i] = 
                                        global_index_sortx_data[win_index*MAX_VOXEL_NUM_PER_WIN+local_index_in_set_data[set_idx*voxel_num_set+i]];
    }

      // get set voxel mask
    for(int i=0;i<voxel_num_set;i++)
    {
        if(i==0)
        {
            set_voxel_mask_data[0*MAX_WIN_NUM*voxel_num_set+set_idx*voxel_num_set+i] = 0.0;
             set_voxel_mask_data[1*MAX_WIN_NUM*voxel_num_set+set_idx*voxel_num_set+i] = 0.0;
        }
        else{
            if(global_index_in_set_data[0*MAX_WIN_NUM*voxel_num_set+set_idx*voxel_num_set+i] == global_index_in_set_data[0*MAX_WIN_NUM*voxel_num_set+set_idx*voxel_num_set+i-1])
            {
                 set_voxel_mask_data[0*MAX_WIN_NUM*voxel_num_set+set_idx*voxel_num_set+i] = -3.4028235e+38;  //  -inf   -1/0.0     -3.4028235e+38
            }
            else{
                set_voxel_mask_data[0*MAX_WIN_NUM*voxel_num_set+set_idx*voxel_num_set+i] = 0.0;
            }

            if(global_index_in_set_data[1*MAX_WIN_NUM*voxel_num_set+set_idx*voxel_num_set+i] == global_index_in_set_data[1*MAX_WIN_NUM*voxel_num_set+set_idx*voxel_num_set+i-1])
            {
                 set_voxel_mask_data[1*MAX_WIN_NUM*voxel_num_set+set_idx*voxel_num_set+i] =  -3.4028235e+38;  // min negitive value in float    -3.4028235e+38   -inf  
            }
            else{
                set_voxel_mask_data[1*MAX_WIN_NUM*voxel_num_set+set_idx*voxel_num_set+i] = 0.0;
            }
        }
    }
}

cudaError_t useLocalIndexGetSortedGlobalIndex_launch(
        unsigned int *global_index_in_set_data,
        unsigned int *global_index_sorty_data,
        unsigned int *global_index_sortx_data,
        unsigned int * local_index_in_set_data, 
        unsigned int *win_index_data, unsigned int*set_num_data,
        float * set_voxel_mask_data,
        int voxel_num_set,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_WIN_NUM+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  useLocalIndexGetSortedGlobalIndex__kernel<<<blocks, threads, 0, stream>>>(global_index_in_set_data,
        global_index_sorty_data,global_index_sortx_data, local_index_in_set_data,win_index_data,set_num_data,set_voxel_mask_data,voxel_num_set);
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void splitAndExpandMask_kernel(
        float* set_voxel_mask_data_0_expand,
        float *set_voxel_mask_data_1_expand,
        unsigned int*set_num_data,
        float * set_voxel_mask_data,
        int voxel_num_set,int max_win_num)
{
    unsigned int set_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(set_idx >= *set_num_data) return;
    float *set_voxel_mask_data0 = set_voxel_mask_data + 0*max_win_num*voxel_num_set + set_idx*voxel_num_set;
    float *set_voxel_mask_data1 = set_voxel_mask_data + 1*max_win_num*voxel_num_set + set_idx*voxel_num_set;
    for(int i=0;i<voxel_num_set;i++)
    {
        for(int j=0;j<NUM_HEADS;j++)
        {
            *(set_voxel_mask_data_0_expand+set_idx*NUM_HEADS*voxel_num_set+j*voxel_num_set+i) = *(set_voxel_mask_data0+i);
            *(set_voxel_mask_data_1_expand+set_idx*NUM_HEADS*voxel_num_set+j*voxel_num_set+i) = *(set_voxel_mask_data1+i);
        }
    }
    
}


cudaError_t splitAndExpandMask_launch(
        float* set_voxel_mask_data_0_expand,
        float * set_voxel_mask_data_1_expand,
         unsigned int*set_num_data,
        float * set_voxel_mask_data,
        int voxel_num_set,int max_win_num,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_WIN_NUM+threadNum-1)/threadNum);
  dim3 threads(threadNum);
   splitAndExpandMask_kernel<<<blocks, threads, 0, stream>>>(set_voxel_mask_data_0_expand, set_voxel_mask_data_1_expand,set_num_data,set_voxel_mask_data,voxel_num_set,max_win_num);
  cudaError_t err = cudaGetLastError();
  return err;
}

int GetSetPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];
    // int maxNumPoints = inputDesc[0].dims.d[1];

    //TRT-input
    // std::cout << "voxelgenerator batch_size: " << batchSize << std::endl;
    unsigned int * global_index_data = const_cast<unsigned int *>((const unsigned int *)inputs[0]);
    unsigned int* coors_in_win_data = const_cast<unsigned int *>((const unsigned int *)inputs[1]);
    unsigned int * voxel_num_in_win_data = const_cast<unsigned int *>((const unsigned int *)inputs[2]);
    unsigned int* win_num_data = const_cast<unsigned int *>((const unsigned int *)inputs[3]);

    //TRT-output
    unsigned int  *global_index_in_set_data = (unsigned int *)(outputs[0]);  // 2 max_win_num  voxel_num_set_
    float * set_voxel_mask_data = (float *)(outputs[1]); // 2 max_win_num voxel_num_set
    unsigned int *set_num_data = (unsigned int *)(outputs[2]);  //  valid set num
    float * set_voxel_mask_data_0_expand = (float*)(outputs[3]); // 1 max_win_num 8 voxel_num_set
    float* set_voxel_mask_data_1_expand = (float*)(outputs[4]);  // 1 max_win_num 8 voxel_num_set
   
    // workspace buffer
    size_t global_index_sorty_size = batchSize * max_win_num_  * max_voxel_num_per_win_ * sizeof(unsigned int);
    size_t global_index_sortx_size = batchSize * max_win_num_  * max_voxel_num_per_win_ * sizeof(unsigned int);
    size_t local_index_in_set_size = batchSize * max_win_num_ * voxel_num_set_ * sizeof(unsigned int);
    size_t  win_index_size =  batchSize * max_win_num_ * sizeof(unsigned int);
    
    size_t workspaces[4];
    workspaces[0] = global_index_sorty_size;
    workspaces[1] = global_index_sortx_size;
    workspaces[2] = local_index_in_set_size;
    workspaces[3] = win_index_size;
    size_t total_workspace = calculateTotalWorkspaceSize(workspaces, 4);

    unsigned int* global_index_sorty_data = static_cast<unsigned int*>(workspace);
    unsigned int* global_index_sortx_data = reinterpret_cast<unsigned int*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(global_index_sorty_data), global_index_sorty_size));
    unsigned int* local_index_in_set_data = reinterpret_cast<unsigned int*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(global_index_sortx_data), global_index_sortx_size));
    unsigned int* win_index_data = reinterpret_cast<unsigned int*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(local_index_in_set_data), local_index_in_set_size)); //win_index_set_belong_to_
    
    // // Initialize workspace memory
    checkCudaErrors(cudaMemsetAsync(global_index_sorty_data, 0, total_workspace, stream)); // total_workspace
    
    // initialize output
    unsigned int global_index_in_set_data_size = batchSize * 2 *  max_win_num_  * voxel_num_set_ * sizeof(unsigned int);
    unsigned int set_voxel_mask_data_size = batchSize * 2 * max_win_num_ * voxel_num_set_ * sizeof(float);
    unsigned int set_num_data_size = batchSize * sizeof(unsigned int);
    unsigned int  set_voxel_mask_data_0_expand_size = batchSize  * max_win_num_ * voxel_num_set_* NUM_HEADS* sizeof(float);
    unsigned int  set_voxel_mask_data_1_expand_size = batchSize  * max_win_num_ * voxel_num_set_* NUM_HEADS* sizeof(float);

   
    checkCudaErrors(cudaMemsetAsync(global_index_in_set_data, 0, global_index_in_set_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(set_voxel_mask_data, 0, set_voxel_mask_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(set_num_data, 0, set_num_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(set_voxel_mask_data_0_expand, 0, set_voxel_mask_data_0_expand_size, stream));
    checkCudaErrors(cudaMemsetAsync(set_voxel_mask_data_1_expand, 0, set_voxel_mask_data_1_expand_size, stream));

    // std::cout << sparse_shape_x_ << " " << sparse_shape_y_ << " " << sparse_shape_z_ << std::endl;
    // std::cout << shift_x_ << " "  << shift_y_ << " " << shift_z_ << " " << max_num_win_x << " " <<  max_num_win_y << " " <<  max_num_win_z << " " << win_shape_x_ << " " << win_shape_y_ << " " <<  win_shape_z_ << std::endl;
    
    checkCudaErrors(getLocalIndex_launch(  // win ---->set
          global_index_data,voxel_num_in_win_data,win_num_data,local_index_in_set_data,win_index_data,set_num_data,voxel_num_set_,
          stream));
    checkCudaErrors(sortY_launch(
          global_index_data,coors_in_win_data,voxel_num_in_win_data,win_num_data,global_index_sorty_data,max_voxel_num_per_win_,win_shape_x_,win_shape_y_,win_shape_z_,
          stream));
     checkCudaErrors(sortX_launch(
          global_index_data,coors_in_win_data,voxel_num_in_win_data,win_num_data,global_index_sortx_data,max_voxel_num_per_win_,win_shape_x_,win_shape_y_,win_shape_z_,
          stream));
    checkCudaErrors(useLocalIndexGetSortedGlobalIndex_launch(global_index_in_set_data,global_index_sorty_data,global_index_sortx_data,local_index_in_set_data,win_index_data,
                                    set_num_data,set_voxel_mask_data,voxel_num_set_,stream));
    checkCudaErrors(splitAndExpandMask_launch(set_voxel_mask_data_0_expand,set_voxel_mask_data_1_expand,set_num_data,set_voxel_mask_data,voxel_num_set_,max_win_num_,stream));
    return 0;
}

nvinfer1::DataType GetSetPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    if(index == 1 || index == 3 || index ==4)
    {
        // cout << "inputTypes[0]: " << inputTypes[0] <<  std::endl;
        // return inputTypes[0];
        return nvinfer1::DataType::kFLOAT;
    }
    
    return inputTypes[0];
    // return nvinfer1::DataType::kINT32;
}

const char* GetSetPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* GetSetPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int GetSetPlugin::getNbOutputs() const noexcept
{
    return 5;
}

int GetSetPlugin::initialize() noexcept
{
    return 0;
}

void GetSetPlugin::terminate() noexcept
{
}

size_t GetSetPlugin::getSerializationSize() const noexcept
{
    return   6 * sizeof(int);
}

void GetSetPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, voxel_num_set_);
    writeToBuffer<int>(d, max_win_num_);
    writeToBuffer<int>(d, max_voxel_num_per_win_);
    writeToBuffer<int>(d, win_shape_x_);
    writeToBuffer<int>(d, win_shape_y_);
    writeToBuffer<int>(d, win_shape_z_);
}

void GetSetPlugin::destroy() noexcept
{
    delete this;
}

void GetSetPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* GetSetPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}


GetSetPluginCreator::GetSetPluginCreator()
{
    
    mPluginAttributes.clear();

    // std::cout <<  *max_num_points_per_voxel_ptr << std::endl;
    mPluginAttributes.emplace_back(PluginField("max_win_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_voxel_num_per_win", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("voxel_num_set", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("win_shape", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GetSetPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* GetSetPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* GetSetPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GetSetPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int max_win_num = 0;
    int max_voxel_num_per_win = 0;
    int voxel_num_set = 0;
   
    int win_shape[3] = {0};
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
        else if (!strcmp(attr_name, "voxel_num_set"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            voxel_num_set = d[0];
        }
        else if (!strcmp(attr_name, "win_shape"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            win_shape[0] = d[0];
            win_shape[1] = d[1];
            win_shape[2] = d[2];
        }
    }
    // std::cout << max_voxels << " " << max_points << " " <<voxel_feature_num << " " << point_cloud_range[0] << " " << point_cloud_range[1] << " "
    // << point_cloud_range[2] << " "<< point_cloud_range[3] << " " << point_cloud_range[4] << " " << point_cloud_range[5] << " " << voxel_size[0] << " "
    // << voxel_size[1] << " " << voxel_size[2] << std::endl;
    std::cout <<  max_win_num  << " " << max_voxel_num_per_win << " " << voxel_num_set << " " 
     << win_shape[0] << " " << win_shape[1] << " " << win_shape[2] << " " << std::endl;
    IPluginV2DynamicExt* plugin = new GetSetPlugin(voxel_num_set,max_win_num,max_voxel_num_per_win,win_shape[0],win_shape[1],win_shape[2]);    
    return plugin;
}

IPluginV2* GetSetPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new GetSetPlugin(serialData, serialLength);
}

void GetSetPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* GetSetPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

GetSetPluginCreator::~GetSetPluginCreator()
{
}