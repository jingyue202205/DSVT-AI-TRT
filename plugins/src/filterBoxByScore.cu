#include "filterBoxByScore.h"

using namespace nvinfer1;
using nvinfer1::FilterBoxByScorePlugin;
using nvinfer1::FilterBoxByScorePluginCreator;
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
static const char* PLUGIN_NAME{"FilterBoxByScorePlugin"};


// Static class fields initialization
PluginFieldCollection FilterBoxByScorePluginCreator::mFC{};
std::vector<PluginField> FilterBoxByScorePluginCreator::mPluginAttributes;

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

__device__ float Logist(float data){ return 1.0f / (1.0f + expf(-data)); };

// create the plugin at runtime from a byte stream
FilterBoxByScorePlugin::FilterBoxByScorePlugin(int max_top_k, float min_x_range,float max_x_range,float min_y_range,float max_y_range,float min_z_range,float max_z_range,
                                                            float voxel_x_size,float voxel_y_size,float voxel_z_size,float score_threshold)
: max_top_k_(max_top_k), min_x_range_(min_x_range), max_x_range_(max_x_range),
    min_y_range_(min_y_range),max_y_range_(max_y_range),min_z_range_(min_z_range),max_z_range_(max_z_range),voxel_x_size_(voxel_x_size),voxel_y_size_(voxel_y_size),voxel_z_size_(voxel_z_size),
    score_threshold_(score_threshold)
{   
}

FilterBoxByScorePlugin::FilterBoxByScorePlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    
    max_top_k_ = readFromBuffer<int>(d);
    
     min_x_range_ = readFromBuffer<float>(d);
    max_x_range_ = readFromBuffer<float>(d);
    min_y_range_ = readFromBuffer<float>(d);
    max_y_range_ = readFromBuffer<float>(d);
    min_z_range_ = readFromBuffer<float>(d);
    max_z_range_ = readFromBuffer<float>(d);
    voxel_x_size_ = readFromBuffer<float>(d);
    voxel_y_size_ = readFromBuffer<float>(d);
    voxel_z_size_ = readFromBuffer<float>(d);

    score_threshold_ = readFromBuffer<float>(d);

}

IPluginV2DynamicExt* FilterBoxByScorePlugin::clone() const noexcept
{
    auto* plugin = new FilterBoxByScorePlugin(max_top_k_, min_x_range_,max_x_range_,min_y_range_,max_y_range_,min_z_range_,max_z_range_,
                                                            voxel_x_size_,voxel_y_size_,voxel_z_size_,score_threshold_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs FilterBoxByScorePlugin::getOutputDimensions(
int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    auto batch_size = inputs[0].d[0];
    // auto line_num = inputs[0].d[1];
    // auto dim_num_0 = inputs[0].d[2];
    // auto dim_num_1 = inputs[1].d[2];
    // auto dim_num_2 = inputs[2].d[2];
    // auto dim_num_3 = inputs[3].d[2];

    // std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" << std::endl;
    // std::cout << batch_size->getConstantValue() << " " << line_num->getConstantValue() << " " 
    //         << dim_num_0->getConstantValue() << " " << dim_num_1->getConstantValue() << " "
    //         << dim_num_2->getConstantValue() << " " 
    //         << dim_num_3->getConstantValue() << std::endl; 
   
    if (outputIndex == 0)  // box_preds
    {
        nvinfer1::DimsExprs dim0{};
        dim0.nbDims = 3;
        dim0.d[0] = batch_size;
        dim0.d[1] = exprBuilder.constant(max_top_k_);
        dim0.d[2] = exprBuilder.constant(LAST_DIMS); 
        return dim0; 
    }

    //   if (outputIndex == 1) // class_ids_preds
    // {
    //     nvinfer1::DimsExprs dim1{};
    //     dim1.nbDims = 2;
    //     dim1.d[0] = batch_size;
    //     dim1.d[1] = exprBuilder.constant(max_top_k_);
       
    //     return dim1; 
    // }

    //   if (outputIndex == 2) // score_preds
    // {
    //     nvinfer1::DimsExprs dim2{};
    //     dim2.nbDims = 2;
    //     dim2.d[0] = batch_size;
    //     dim2.d[1] = exprBuilder.constant(max_top_k_);
    //     return dim2; 
    // }
    if (outputIndex == 1) // valid_line_num
    {
        nvinfer1::DimsExprs dim3{};
        dim3.nbDims = 1;
        dim3.d[0] = batch_size;
        return dim3; 
    }
}

bool FilterBoxByScorePlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       // topk_scores    hm_topk_1->getOutput(0)  float    1*500
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)      // topk_classes   topk_classes_floor_div_elementwise->getOutput(0)  int   1*500
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)        // xs    topk_xs_gather_reshape->getOutput(0)  int   1*500
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)      // xy  topk_ys_gather_reshape->getOutput(0) int 1*500
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 4)        //center  center_gather->getOutput(0)   float   1  1  500  2
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 5)      //center_z   center_z_gather->getOutput(0) float 1  1  500  1
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 6)       //angle  atan_angle->getOutput(0)  float   1  1   500  1
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 7)       // dim  dim_gather->getOutput(0) float  1  1  500  3
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 8)      // box_preds  1  500   7
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    // if (pos == 9)       // class_ids_preds  1  500
    // {
    //     return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    // }
    // if (pos == 10)      //score_preds  1  500
    // {
    //     return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    // }
     if (pos == 9)      //valid_line_num  1
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    
    return false;
}

void FilterBoxByScorePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t FilterBoxByScorePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}
   

__global__ void filter_box_by_score_kernel(float*topk_scores,unsigned int*topk_classes,unsigned int*xs,unsigned int*ys,float* center,float* center_z,float* angle,float* dim,
            float* output_box_preds,unsigned int*valid_line_num, 
         float min_x_range,float max_x_range,float min_y_range,float max_y_range,float min_z_range,float max_z_range,
         float voxel_x_size,float voxel_y_size, float voxel_z_size,float score_threshold)
{
    // printf("point_size:%d\n",*points_size);
    int line_idx = blockIdx.x * blockDim.x + threadIdx.x;
    float score = *(topk_scores+line_idx);

      unsigned int xs_one = *(xs+line_idx);
        unsigned int ys_one = *(ys+line_idx);
        float center_0_one = *(center+line_idx*2+0);
        float center_1_one = *(center+line_idx*2+1);
        float new_xs = xs_one + center_0_one;
        float new_ys = ys_one + center_1_one;
        new_xs = new_xs  * voxel_x_size + min_x_range;
        new_ys = new_ys * voxel_y_size + min_y_range;
        
        float center_z_one = *(center_z+line_idx);
        // printf("point: %f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",voxel_x_size,voxel_y_size,voxel_z_size,min_x_range,max_x_range,min_y_range,max_y_range,min_z_range,max_z_range,new_xs,new_ys,center_z_one);

        if( !(new_xs >= min_x_range && new_xs < max_x_range
        && new_ys >= min_y_range && new_ys < max_y_range
        && center_z_one >= min_z_range && center_z_one < max_z_range) ) {
      return;
    }

    if(score >= score_threshold)
    {
        int index = atomicAdd(valid_line_num,1);
        //  printf("index:%d\n",index);
        *(output_box_preds+index*9+0) = new_xs;
        *(output_box_preds+index*9+1) = new_ys;
        *(output_box_preds+index*9+2) = center_z_one;
        *(output_box_preds+index*9+3) = *(dim+line_idx*3+0);
          *(output_box_preds+index*9+4) = *(dim+line_idx*3+1);
        *(output_box_preds+index*9+5) =*(dim+line_idx*3+2);
        *(output_box_preds+index*9+6) = *(angle+line_idx);

           *(output_box_preds+index*9+7) = *(topk_classes+line_idx);
        *(output_box_preds+index*9+8) =  *(topk_scores+line_idx);
        
    }
}


cudaError_t filter_box_by_score_launch(float*topk_scores,unsigned int*topk_classes,unsigned int*xs,unsigned int*ys,float* center,float* center_z,float* angle,float* dim,
            float* output_box_preds,unsigned int*valid_line_num, 
         float min_x_range,float max_x_range,float min_y_range,float max_y_range,float min_z_range,float max_z_range,
         float voxel_x_size,float voxel_y_size, float voxel_z_size,float score_threshold, cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((HM_TOP_K+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  filter_box_by_score_kernel<<<blocks, threads, 0, stream>>>
       (  topk_scores,topk_classes,xs,ys,center,center_z,angle,dim,output_box_preds,valid_line_num, 
         min_x_range,max_x_range,min_y_range,max_y_range,min_z_range,max_z_range,voxel_x_size,voxel_y_size,voxel_z_size,score_threshold);
  cudaError_t err = cudaGetLastError();
  return err;
}

int FilterBoxByScorePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];

    // int maxNumPoints = inputDesc[0].dims.d[1];
    //TRT-input
        // topk_scores    hm_topk_1->getOutput(0)  float    1*500
    // topk_classes   topk_classes_floor_div_elementwise->getOutput(0)  int   1*500
    // xs    topk_xs_gather_reshape->getOutput(0)  int   1*500
    // ys  topk_ys_gather_reshape->getOutput(0) int 1*500
    //center  center_gather->getOutput(0)   float   1  1  500  2
    //center_z   center_z_gather->getOutput(0) float 1  1  500  1
    //angle  atan_angle->getOutput(0)  float   1  1   500  1
    // dim  dim_gather->getOutput(0) float  1  1  500  3
    float * topk_scores = const_cast<float *>((const float *)inputs[0]); // 1 * 500
    unsigned int * topk_classes = const_cast<unsigned int *>((const unsigned int *)inputs[1]);  // 1 500
    unsigned int  * xs = const_cast<unsigned int*>((const unsigned int*)inputs[2]);  // 1 500
    unsigned int  * ys = const_cast<unsigned int*>((const unsigned int*)inputs[3]); // 1 500
     float * center = const_cast<float *>((const float *)inputs[4]);  // 1   1   500  2
    float * center_z = const_cast<float *>((const float *)inputs[5]); //  1   1   500  1
    float * angle = const_cast<float *>((const float *)inputs[6]);  //  1   1   500  1
    float * dim = const_cast<float *>((const float *)inputs[7]);  //  1   1   500  3
    //TRT-output
    float *output_box_preds = (float *)(outputs[0]);  //   1  500  9
    // float *output_class_id_preds = (float *)(outputs[1]);  //  1  500
    // float *output_score_preds = (float *)(outputs[2]);   //  1   500
    unsigned int *valid_line_num = (unsigned int*)(outputs[1]);  //  1

    // init output
    unsigned int output_box_preds_data_size = batchSize * max_top_k_ * LAST_DIMS * sizeof(float);
    checkCudaErrors(cudaMemsetAsync(output_box_preds, 0, output_box_preds_data_size, stream));

    // unsigned int output_class_id_preds_size = batchSize * max_top_k_ * 1 * sizeof(float);
    // checkCudaErrors(cudaMemsetAsync(output_class_id_preds, 0, output_class_id_preds_size, stream));

    // unsigned int output_score_preds_data_size = batchSize * max_top_k_ * 1 * sizeof(float);
    // checkCudaErrors(cudaMemsetAsync(output_score_preds, 0, output_score_preds_data_size, stream));

    unsigned int valid_line_num_data_size = batchSize * 1 * sizeof(unsigned int);
    checkCudaErrors(cudaMemsetAsync(valid_line_num, 0, valid_line_num_data_size, stream));
    
    
    checkCudaErrors(filter_box_by_score_launch(
         topk_scores,topk_classes,xs,ys,center,center_z,angle,dim,output_box_preds,valid_line_num, 
         min_x_range_,max_x_range_,min_y_range_,max_y_range_,min_z_range_,max_z_range_,voxel_x_size_,voxel_y_size_,voxel_z_size_,score_threshold_,
         stream));

    // cout << "filter box by score finished" << std::endl;
    return 0;
}


nvinfer1::DataType FilterBoxByScorePlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{   
    if (index == 1)
        return nvinfer1::DataType::kINT32;
    return inputTypes[0];
}

const char* FilterBoxByScorePlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* FilterBoxByScorePlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int FilterBoxByScorePlugin::getNbOutputs() const noexcept
{
    return 2;
}

int FilterBoxByScorePlugin::initialize() noexcept
{
    return 0;
}

void FilterBoxByScorePlugin::terminate() noexcept
{
}


size_t FilterBoxByScorePlugin::getSerializationSize() const noexcept
{
    return  1 * sizeof(int)+10*sizeof(float);
}

void FilterBoxByScorePlugin::serialize(void* buffer) const noexcept
{

    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, max_top_k_);
    writeToBuffer<float>(d, min_x_range_);
    writeToBuffer<float>(d, max_x_range_);
    writeToBuffer<float>(d, min_y_range_);
    writeToBuffer<float>(d, max_y_range_);
    writeToBuffer<float>(d, min_z_range_);
    writeToBuffer<float>(d, max_z_range_);
    writeToBuffer<float>(d, voxel_x_size_);
    writeToBuffer<float>(d, voxel_y_size_);
    writeToBuffer<float>(d, voxel_z_size_);
    writeToBuffer<float>(d, score_threshold_);
}

void FilterBoxByScorePlugin::destroy() noexcept
{
    delete this;
}

void FilterBoxByScorePlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* FilterBoxByScorePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}



FilterBoxByScorePluginCreator::FilterBoxByScorePluginCreator()
{

    mPluginAttributes.clear();

      mPluginAttributes.emplace_back(PluginField("max_top_k", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("point_cloud_range", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("voxel_size", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* FilterBoxByScorePluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* FilterBoxByScorePluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* FilterBoxByScorePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* FilterBoxByScorePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;

    int max_top_k = 0;

    float min_x_range=0.0;
    float max_x_range=0.0;
    float min_y_range=0.0;
    float max_y_range=0.0;
    float min_z_range=0.0;
    float max_z_range=0.0;
    float voxel_x_size=0.0;
    float voxel_y_size=0.0;
    float voxel_z_size=0.0;
    float score_threshold=0.0;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if(!strcmp(attr_name, "max_top_k"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_top_k = d[0];
        }
          else if (!strcmp(attr_name, "point_cloud_range"))
        {
            const float* d = static_cast<const float*>(fields[i].data);
            min_x_range = d[0];
            max_x_range = d[1];
            min_y_range = d[2];
            max_y_range=d[3];
            min_z_range = d[4];
            max_z_range=d[5];
        }
          else if (!strcmp(attr_name, "voxel_size"))
        {
            const float* d = static_cast<const float*>(fields[i].data);
            voxel_x_size = d[0];
            voxel_y_size = d[1];
            voxel_z_size = d[2];
        }
         else if (!strcmp(attr_name, "score_threshold"))
        {
            const float* d = static_cast<const float*>(fields[i].data);
            score_threshold = d[0];
        }
        
    }
    std::cout << "filter box by score    " <<max_top_k << " " << min_x_range << " " 
     << max_x_range << " " << min_y_range << " " << max_y_range << " " << min_z_range << " "  
     << max_z_range <<  " " << voxel_x_size  << " " << voxel_y_size << " " << voxel_z_size << " "<< score_threshold<< std::endl;
    IPluginV2DynamicExt* plugin = new FilterBoxByScorePlugin(max_top_k, min_x_range,max_x_range,min_y_range,max_y_range,min_z_range,max_z_range,
                                                            voxel_x_size,voxel_y_size,voxel_z_size,score_threshold);
    return plugin;
}

IPluginV2* FilterBoxByScorePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new FilterBoxByScorePlugin(serialData, serialLength);
}

void FilterBoxByScorePluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* FilterBoxByScorePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

FilterBoxByScorePluginCreator::~FilterBoxByScorePluginCreator()
{
   
}