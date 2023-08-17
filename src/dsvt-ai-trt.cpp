#include <fstream>
#include <iostream>
#include <iomanip> //设置输出格式
#include <map>
#include <sstream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <chrono>
#include <dirent.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"

#include "logging.h"
#include "params.h"
#include "points2Features.h"
#include "torchScatterMax.h"
#include "windowPartition.h"
#include "getSet.h"
#include "getValueByIndex.h"
#include "mapSetFeature2voxel.h"
#include "layerNorm.h"
#include "gelu.h"
#include "map2bev.h"
#include "filterBoxByScore.h"

// #include "cnpy.h"

#include "helper.h"
#include "plugin_helper.h"

#include<complex>
#include<cstdlib>
#include<iostream>
#include<map>
#include<string>

#include <time.h>
#include <chrono>
#include <cmath>
#include <string>
#include <string.h>

using namespace nvinfer1;
using namespace std;
using namespace std::chrono;
using std::string;


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


// stuff we know about the network and the input/output blobs
const char* INPUT_POINTS = "points_data";
const char* INPUT_POINTS_SIZE = "points_size";
const char* OUTPUT_VOXELS = "voxels";
const char* OUTPUT_COORS = "coors";
const char* OUTPUT_VOXEL_NUM = "voxel_num";
static Logger gLogger;


class RTLogger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level message
        //if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR || severity == Severity::kINFO ) {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "trt_infer: " << msg << std::endl;
        }
    }
} rt_glogger;



ILayer* add_batchNorm1d_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
   
    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    // IShuffleLayer* shuffle1 = network->addShuffle(input);
    // auto dim1 = input.getDimensions();
    // dim1.d[0] = dim1.d[1];
    // dim1.d[1] = dim1.d[2];
    // dim1.d[2] = 1;
    // dim1.d[3] = 1;
    // dim1.nbDims = 4;
    // shuffle1->setReshapeDimensions(dim1);
    // assert(shuffle1);

    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    IShuffleLayer* shuffle2 = network->addShuffle(*scale_1->getOutput(0));
    auto dim2 = scale_1->getOutput(0)->getDimensions();
    dim2.d[2] = dim2.d[1];
    dim2.d[1] = dim2.d[0];
    dim2.d[0] = 1;
    dim2.nbDims = 3;
    shuffle2->setReshapeDimensions(dim2);
    assert(shuffle2);
    auto lr = network->addActivation(*shuffle2->getOutput(0), ActivationType::kRELU);
    lr->setAlpha(1e-8);
    return lr;
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBnLELU(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, 
                    std::string conv2d_prefix, std::string batchnorm_prefix) 
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[conv2d_prefix + ".weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), batchnorm_prefix, 1e-3);

    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    lr->setAlpha(1e-8);

    return lr;
}

ILayer* convBn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, 
                    std::string conv2d_prefix, std::string batchnorm_prefix) 
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[conv2d_prefix + ".weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), batchnorm_prefix, 1e-3);


    return bn1;
}

ILayer* conv_with_bias(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, 
                    std::string conv2d_prefix) 
{
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[conv2d_prefix + ".weight"], weightMap[conv2d_prefix + ".bias"]);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    return conv1;
}

ILayer* deconvBnLELU(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, 
                    std::string conv2d_prefix, std::string batchnorm_prefix) 
{
    
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IDeconvolutionLayer* conv1 = network->addDeconvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[conv2d_prefix + ".weight"], emptywts);
    assert(conv1);
    // conv1->setStrideNd(DimsHW{s, s});
    conv1->setStride(DimsHW{s, s});
    // conv1->setPaddingNd(DimsHW{p, p});
    conv1->setDilationNd(DimsHW{1,1});
    conv1->setPrePadding(DimsHW{p, p}); // pytorch padding
    // conv1->setPostPadding(DimsHW(output_padding+1,output_padding+1)); // pytorch output_padding
  
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), batchnorm_prefix, 1e-3);

    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    lr->setAlpha(1e-8);

    return lr;
}



ILayer* add_bottom_up_block_conv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, 
                          int conv_0_in_channel,  int conv_0_out_channel, int conv_0_ksize, int conv_0_stride, int conv_0_padding,
                          std::string conv_0_prefix, int batchnorm2d_0_num_features, std::string batchnorm2d_0_prefix,
                          int conv_1_in_channel,  int conv_1_out_channel, int conv_1_ksize, int conv_1_stride, int conv_1_padding,
                          std::string conv_1_prefix, int batchnorm2d_1_num_features, std::string batchnorm2d_1_prefix,
                          int conv_2_in_channel,  int conv_2_out_channel, int conv_2_ksize, int conv_2_stride, int conv_2_padding,
                          std::string conv_2_prefix, int batchnorm2d_2_num_features, std::string batchnorm2d_2_prefix
                          )
{
    auto bottom_up_conv_0_bn2d_relu = convBnLELU(network,weightMap,input,conv_0_out_channel,conv_0_ksize,conv_0_stride,conv_0_padding,
                        conv_0_prefix,batchnorm2d_0_prefix);
    auto bottom_up_conv_1_bn2d_relu = convBnLELU(network,weightMap,*bottom_up_conv_0_bn2d_relu->getOutput(0),conv_1_out_channel,conv_1_ksize,conv_1_stride,conv_1_padding,
                        conv_1_prefix,batchnorm2d_1_prefix);
    auto bottom_up_conv_2_bn2d_relu = convBnLELU(network,weightMap,*bottom_up_conv_1_bn2d_relu->getOutput(0),conv_2_out_channel,conv_2_ksize,conv_2_stride,conv_2_padding,
                        conv_2_prefix,batchnorm2d_2_prefix);
    return bottom_up_conv_2_bn2d_relu;
}


ILayer* fullyConnectedBnLELU(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, 
                    std::string  prefix, std::string batchnorm_prefix) 
{
    
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto input_squeeze = network->addShuffle(input);
    auto dim = input.getDimensions();
    dim.d[0] = dim.d[1];
    dim.d[1] = dim.d[2];
    dim.d[2] = 1;
    dim.d[3] = 1;
    dim.nbDims = 4;
    input_squeeze->setReshapeDimensions(dim);  // 50000 96 1 1
     
    auto fully_connected_layer = network->addFullyConnected(*input_squeeze->getOutput(0), outch, weightMap[prefix + ".weight"],emptywts);
    auto bn1d_relu = add_batchNorm1d_relu(network, weightMap,*fully_connected_layer->getOutput(0),batchnorm_prefix, 1e-5);
    return bn1d_relu;
}

ILayer* multHeadAttention(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input_q, ITensor& input_k,ITensor& input_v, ITensor& attn_mask, int outch, 
                    std::string  weight_prefix, std::string bias_prefix,std::string out_prefix ,int num_heads) 
{
    // transpose(1,0)     1 454 36 192 ---> 1 36 454 192
    auto input_q_transpose =  network->addShuffle(input_q);
    input_q_transpose->setFirstTranspose(Permutation{0,2,1,3});

    auto input_k_transpose =  network->addShuffle(input_k);
   input_k_transpose->setFirstTranspose(Permutation{0,2,1,3});   

    auto input_v_transpose =  network->addShuffle(input_v);
    input_v_transpose->setFirstTranspose(Permutation{0,2,1,3});


    // reshape 1 36 454 192 ---> 454*36 192 1 1  for fullyconnected
    auto input_q_transpose_reshape = network->addShuffle(*input_q_transpose->getOutput(0));
    auto input_k_transpose_reshape = network->addShuffle(*input_k_transpose->getOutput(0));
    auto input_v_transpose_reshape = network->addShuffle(*input_v_transpose->getOutput(0));

    auto dim = input_q.getDimensions();
    int dim_1 = dim.d[1];  // 454
    int dim_2 = dim.d[2];  // 36
    int dim_3 = dim.d[3];  // 192
    dim.d[0] = dim_1 * dim_2;
    dim.d[1] = dim.d[3];
    dim.d[2] = 1;
    dim.d[3] = 1;
    dim.nbDims = 4;
    input_q_transpose_reshape->setReshapeDimensions(dim);  // 454*36 192 1 1
    input_k_transpose_reshape->setReshapeDimensions(dim);
    input_v_transpose_reshape->setReshapeDimensions(dim);

     
    // for(int i=0;i<10;i++)
    // {
    //     std::cout << reinterpret_cast<const float*>(weightMap[weight_prefix + ".value"].values)[i] << "," ;
    // }
    // std::cout << std::endl;

     
    auto fully_connected_layer_q = network->addFullyConnected(*input_q_transpose_reshape->getOutput(0), outch, weightMap[weight_prefix + ".query"],weightMap[bias_prefix + ".query"]);
    auto fully_connected_layer_k = network->addFullyConnected(*input_k_transpose_reshape->getOutput(0), outch, weightMap[weight_prefix + ".key"],weightMap[bias_prefix + ".key"]);
    auto fully_connected_layer_v = network->addFullyConnected(*input_v_transpose_reshape->getOutput(0), outch, weightMap[weight_prefix + ".value"],weightMap[bias_prefix + ".value"]);

    auto fully_connected_layer_q_squeeze = network->addShuffle(*fully_connected_layer_q->getOutput(0));
    auto fully_connected_layer_k_squeeze = network->addShuffle(*fully_connected_layer_k->getOutput(0));
    auto fully_connected_layer_v_squeeze = network->addShuffle(*fully_connected_layer_v->getOutput(0));

    auto dim0 = input_q.getDimensions();
    dim0.d[0] = 1;
    dim0.d[1] = dim_2;
    dim0.d[2] = dim_1;
    dim0.d[3] = dim_3;
    dim0.nbDims = 4;
    fully_connected_layer_q_squeeze->setReshapeDimensions(dim0);  // 1 36 454  192
    fully_connected_layer_k_squeeze->setReshapeDimensions(dim0);  // 1 36  454 192
    fully_connected_layer_v_squeeze->setReshapeDimensions(dim0);   // 1 36  454  192

    //  auto fully_connected_layer_v_squeeze_transpose =  network->addShuffle(*fully_connected_layer_v_squeeze->getOutput(0));
    // fully_connected_layer_v_squeeze_transpose->setFirstTranspose(Permutation{0,2,1,3});

    

    // reshape from 1 36 454 192---->  36  454*8  24
    auto q_reshape = network->addShuffle(*fully_connected_layer_q_squeeze->getOutput(0));
    auto k_reshape = network->addShuffle(*fully_connected_layer_k_squeeze->getOutput(0));
    auto v_reshape = network->addShuffle(*fully_connected_layer_v_squeeze->getOutput(0));
    auto dim1 = input_q.getDimensions();
    dim1.d[0] = dim_2; // 36
    dim1.d[1] = dim_1 * num_heads;  // 454 * 8
    dim1.d[2] = int(dim_3 / num_heads);  // 192/8
    dim1.nbDims = 3;
    q_reshape->setReshapeDimensions(dim1);
    k_reshape->setReshapeDimensions(dim1);
    v_reshape->setReshapeDimensions(dim1);
     
    // transpose    0,1,2  ---> 1,0,2        454*8  36 24
    auto q_transpose =  network->addShuffle(*q_reshape->getOutput(0));
    q_transpose->setFirstTranspose(Permutation{1,0,2});

    auto k_transpose =  network->addShuffle(*k_reshape->getOutput(0));
    k_transpose->setFirstTranspose(Permutation{1,2,0});   // 454*8  24 36

    auto v_transpose =  network->addShuffle(*v_reshape->getOutput(0));
    v_transpose->setFirstTranspose(Permutation{1,0,2});


    // reshape mask  1 454 8 36 --->454*8 1  36
    auto attn_mask_reshape =   network->addShuffle(attn_mask);
    auto dim2 = attn_mask.getDimensions();
    dim2.d[0] = dim_1 * num_heads;
    dim2.d[1] = 1;
    dim2.d[2] = dim_2;
    dim2.nbDims = 3;
    attn_mask_reshape->setReshapeDimensions(dim2);  // 454*8 1 36

    // q / sqrt(24)    
// float shift = 0.0;
float scale = sqrt(dim_3 / num_heads);
// float scale = 0.0;//0.20412414523193154;
// std::cout << "scale: " << scale << std::endl;
// float power = 1.0;
// nvinfer1::Weights scaleShift{nvinfer1::DataType::kFLOAT, &shift, 1 };
// nvinfer1::Weights scaleScale{nvinfer1::DataType::kFLOAT, &scale, 1 };
// nvinfer1::Weights scalePower{nvinfer1::DataType::kFLOAT, &power, 1};
// // (x*scale+shift)^power   not work   
// auto q_scale = network->addScale(*q_transpose->getOutput(0),nvinfer1::ScaleMode::kUNIFORM,scaleShift, scaleScale, scalePower);

// 使用addScale无法实现tensor×constant
auto dimension = (*q_transpose->getOutput(0)).getDimensions();
float *scale_ptr = reinterpret_cast<float*>(malloc(sizeof(float) * dim_1*dim_2*dim_3));
    for (int i = 0; i <  dim_1*dim_2*dim_3; i++) {
        scale_ptr[i] = scale;
    }
nvinfer1::Weights scaleScale{nvinfer1::DataType::kFLOAT, scale_ptr, dim_1*dim_2*dim_3};
auto one_constant = network->addConstant(dimension,scaleScale);

auto element_wise_div = network->addElementWise(*q_transpose->getOutput(0),*one_constant->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);



// 454*8  36 24   x   454*8  24 36
auto bmm_q_k = network->addMatrixMultiply(*element_wise_div->getOutput(0),nvinfer1::MatrixOperation::kNONE, *k_transpose->getOutput(0),nvinfer1::MatrixOperation::kNONE);
// 454*8  36 36   + 454*8  1 36
auto element_wise_sum = network->addElementWise(*bmm_q_k->getOutput(0),*attn_mask_reshape->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);

auto softmax_layer = network->addSoftMax(*element_wise_sum->getOutput(0));
 softmax_layer->setAxes(1<<2);
// 454*8  36 36   x   454*8  36 24
auto bmm_attn_v = network->addMatrixMultiply(*softmax_layer->getOutput(0),nvinfer1::MatrixOperation::kNONE, *v_transpose->getOutput(0),nvinfer1::MatrixOperation::kNONE);
//454*8 36 24 --->36  454*8 24
auto bmm_attn_v_transpose = network->addShuffle(*bmm_attn_v->getOutput(0));
bmm_attn_v_transpose->setFirstTranspose(Permutation{1,0,2});

// //36  454*8 24  ----> 36 454 192
 auto bmm_attn_v_transpose_reshape = network->addShuffle(*bmm_attn_v_transpose->getOutput(0));
auto dim3 = input_q.getDimensions();
// dim_1 = dim.d[1];  // 454
// dim_2 = dim.d[2];  // 36
// dim_3 = dim.d[3];  // 192
dim3.d[0] = 1;
dim3.d[1] = dim_2;
dim3.d[2] = dim_1;
dim3.d[3] = dim_3;
dim3.nbDims = 4;
bmm_attn_v_transpose_reshape->setReshapeDimensions(dim3);  //1 36  454 192 

// 1 36  454 192  -->> 454*36 192 1  for fullyconnected
auto bmm_attn_v_transpose_reshape_1 = network->addShuffle(*bmm_attn_v_transpose_reshape->getOutput(0));
auto dim4 = input_q.getDimensions();
// dim_1 = dim.d[1];  // 454
// dim_2 = dim.d[2];  // 36
// dim_3 = dim.d[3];  // 192
dim4.d[0] = dim_1*dim_2;
dim4.d[1] = dim_3;
dim4.d[2] = 1;
dim4.d[3] = 1;
dim4.nbDims = 4;
bmm_attn_v_transpose_reshape_1->setReshapeDimensions(dim4);  // 454*36 192 1 1

auto fully_connected_layer = network->addFullyConnected(*bmm_attn_v_transpose_reshape_1->getOutput(0), outch, weightMap[out_prefix + ".weight"],weightMap[out_prefix + ".bias"]);

auto fully_connected_layer_reshape = network->addShuffle(*fully_connected_layer->getOutput(0));
fully_connected_layer_reshape->setReshapeDimensions(dim3);  // 1 36 454 192


auto fully_connected_layer_reshape_transpose = network->addShuffle(*fully_connected_layer_reshape->getOutput(0));
fully_connected_layer_reshape_transpose->setFirstTranspose(Permutation{0,2,1,3});  // 1 454 36 192

return fully_connected_layer_reshape_transpose;
}


ILayer* fullyConnectedBnLELU_fullyConnected(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch1,int outch2, 
                    std::string  prefix_1,  std::string  prefix_2,  std::string batchnorm_prefix) 
{
    
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    auto input_squeeze = network->addShuffle(input);
    auto dim = input.getDimensions();
    dim.d[0] = dim.d[1];
    dim.d[1] = dim.d[2];
    dim.d[2] = 1;
    dim.d[3] = 1;
    dim.nbDims = 4;
    input_squeeze->setReshapeDimensions(dim);  // 50000 96 1 1
     
    auto fully_connected_layer = network->addFullyConnected(*input_squeeze->getOutput(0), outch1, weightMap[prefix_1 + ".weight"],weightMap[prefix_1 + ".bias"]);
    auto bn1d_relu = add_batchNorm1d_relu(network, weightMap,*fully_connected_layer->getOutput(0),batchnorm_prefix, 1e-5);


     auto input_squeeze1 = network->addShuffle(*bn1d_relu->getOutput(0));
    auto dim1 = (*(bn1d_relu->getOutput(0))).getDimensions();
    dim1.d[0] = dim1.d[1];
    dim1.d[1] = dim1.d[2];
    dim1.d[2] = 1;
    dim1.d[3] = 1;
    dim1.nbDims = 4;
    input_squeeze1->setReshapeDimensions(dim1);  // 50000 96 1 1


    auto fully_connected_layer_1 = network->addFullyConnected(*input_squeeze1->getOutput(0), outch2, weightMap[prefix_2 + ".weight"],weightMap[prefix_2 + ".bias"]);
    return fully_connected_layer_1;
}

ILayer* fullyConnected_gelu_fullyConnected(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor *input1, ITensor *input2,  int outch1,int outch2, 
                    std::string  prefix_1,  std::string  prefix_2,int max_pillars_num,int channel_num) 
{
    auto input_squeeze = network->addShuffle(*input1);  // 1 7000 192
    auto dim = input1->getDimensions();
    dim.d[0] = dim.d[1];
    dim.d[1] = dim.d[2];
    dim.d[2] = 1;
    dim.d[3] = 1;
    dim.nbDims = 4;
    input_squeeze->setReshapeDimensions(dim);  // 7000 192 1 1
     
    auto fully_connected_layer = network->addFullyConnected(*input_squeeze->getOutput(0), outch1, weightMap[prefix_1 + ".weight"],weightMap[prefix_1 + ".bias"]);


     auto fully_connected_layer_reshape = network->addShuffle(*fully_connected_layer->getOutput(0));
    auto dim1 = (*(fully_connected_layer->getOutput(0))).getDimensions();
    int dim_0 = dim1.d[0];
    int dim_1 = dim1.d[1];
    dim1.d[0] = 1;
    dim1.d[1] = dim_0;
    dim1.d[2] = dim_1;
    dim1.nbDims = 3;
    fully_connected_layer_reshape->setReshapeDimensions(dim1);  // 1 7000 384

    auto gelu_op = add_gelu_op(network,fully_connected_layer_reshape->getOutput(0),input2,max_pillars_num,channel_num);

    auto gelu_op_reshape =  network->addShuffle(*gelu_op->getOutput(0));
    dim.d[1] = outch1;
    gelu_op_reshape->setReshapeDimensions(dim);

    auto fully_connected_layer_1 = network->addFullyConnected(*gelu_op_reshape->getOutput(0), outch2, weightMap[prefix_2 + ".weight"],weightMap[prefix_2 + ".bias"]);
    auto  fully_connected_layer_1_reshape = network->addShuffle(*fully_connected_layer_1->getOutput(0));
    fully_connected_layer_1_reshape->setReshapeDimensions( input1->getDimensions());
    return fully_connected_layer_1_reshape;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config) {
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);

    ITensor* point_data = network->addInput(INPUT_POINTS, DataType::kFLOAT, Dims3{1, MAX_POINTS_NUM,4});
    Dims dims1;
    dims1.d[0] = 1;
    dims1.nbDims = 1;
    ITensor* point_size = network->addInput(INPUT_POINTS_SIZE,DataType::kINT32,dims1);
    assert(point_data);
    assert(point_size);
    // return;

    std::map<std::string, Weights> weightMap = loadWeights_new("../dsvt.wts");
    std::cout << "load weights finished" << std::endl;


    std::map<std::string,Weights>::iterator iter2;
    
    std::string find_str = ".in_proj_";
     int j = 0;
    for(iter2 = weightMap.begin();iter2!=weightMap.end();iter2++ )
    {
        std::string name = iter2->first;
        // std::cout << name << std::endl;
        auto idx=name.find(find_str); 
        if (idx == string::npos )
        {
            ;
        }
        else
        {
            std::cout<< j<< ": " << name << std::endl;
            j += 1;
        }

    }
   

    auto voxelGenerator = add_voxel_generator(network,point_data,point_size,MAX_POINTS_NUM,MAX_POINTS_NUM_1,MAX_PILLARS_NUM,
                                              POINT_FEATURE_NUM,FEATURES_NUM,POINTS_NUM_PER_VOXEL,
                                        X_MIN,X_MAX,Y_MIN,Y_MAX,Z_MIN,Z_MAX,
                                        VOXEL_SIZE_X,VOXEL_SIZE_Y,VOXEL_SIZE_Z,
                                        GRID_SIZE_X,GRID_SIZE_Y,GRID_SIZE_Z);

    auto pfn_layer_0 = fullyConnectedBnLELU(network,weightMap,*voxelGenerator->getOutput(0),PFN_LAYER_0_OUT_CHANNEL,"module.vfe.pfn_layers.0.linear","module.vfe.pfn_layers.0.norm");

    auto torch_scatter_max_0 = add_torch_scatter_max(network,pfn_layer_0->getOutput(0),voxelGenerator->getOutput(1),voxelGenerator->getOutput(3),voxelGenerator->getOutput(4),
                                                                                                                MAX_POINTS_NUM_1,MAX_PILLARS_NUM,PFN_LAYER_0_OUT_CHANNEL);

    // concat
     ITensor* inputTensors[] = {pfn_layer_0->getOutput(0), torch_scatter_max_0->getOutput(0)};
    auto cat_tensor = network->addConcatenation(inputTensors, 2);
    cat_tensor->setAxis(2);

     auto pfn_layer_1 = fullyConnectedBnLELU(network,weightMap,*cat_tensor->getOutput(0),PFN_LAYER_1_OUT_CHANNEL,"module.vfe.pfn_layers.1.linear","module.vfe.pfn_layers.1.norm");

    auto torch_scatter_max_1 = add_torch_scatter_max(network,pfn_layer_1->getOutput(0),voxelGenerator->getOutput(1),voxelGenerator->getOutput(3),voxelGenerator->getOutput(4),
                                                                                                                MAX_POINTS_NUM_1,MAX_PILLARS_NUM,PFN_LAYER_1_OUT_CHANNEL);

    auto window_partition_0 =  add_window_partition(network,voxelGenerator->getOutput(2),voxelGenerator->getOutput(4),MAX_WIN_NUM,MAX_VOXEL_NUM_PER_WIN,
                                                            SPARSE_SHAPE_X,SPARSE_SHAPE_Y,SPARSE_SHAPE_Z,WIN_SHAPE_0_X,WIN_SHAPE_0_Y,WIN_SHAPE_0_Z,
                                                            SHIFT_0_X,SHIFT_0_Y,SHIFT_0_Z);
     auto window_partition_1 =  add_window_partition(network,voxelGenerator->getOutput(2),voxelGenerator->getOutput(4),MAX_WIN_NUM,MAX_VOXEL_NUM_PER_WIN,
                                                            SPARSE_SHAPE_X,SPARSE_SHAPE_Y,SPARSE_SHAPE_Z,WIN_SHAPE_1_X,WIN_SHAPE_1_Y,WIN_SHAPE_1_Z,
                                                            SHIFT_1_X,SHIFT_1_Y,SHIFT_1_Z);
    auto get_set_op_0 = add_get_set_op(network,window_partition_0->getOutput(0),window_partition_0->getOutput(1),window_partition_0->getOutput(2),window_partition_0->getOutput(3),
                                                                        MAX_WIN_NUM,MAX_VOXEL_NUM_PER_WIN,VOXEL_NUM_SET,WIN_SHAPE_0_X,WIN_SHAPE_0_Y,WIN_SHAPE_0_Z);// 12*12
    auto get_set_op_1 = add_get_set_op(network,window_partition_1->getOutput(0),window_partition_1->getOutput(1),window_partition_1->getOutput(2),window_partition_1->getOutput(3),
                                                                        MAX_WIN_NUM,MAX_VOXEL_NUM_PER_WIN,VOXEL_NUM_SET,WIN_SHAPE_1_X,WIN_SHAPE_1_Y,WIN_SHAPE_1_Z);//24*24

    auto embed_layer_0_0_0 = fullyConnectedBnLELU_fullyConnected(network, weightMap, *window_partition_0->getOutput(5), POSEMBED_LAYBERS_OUT_FEATURES, PFN_LAYER_1_OUT_CHANNEL,
                                                            "module.backbone_3d.input_layer.posembed_layers.0.0.0.position_embedding_head.0",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.0.0.position_embedding_head.3",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.0.0.position_embedding_head.1"); 
    auto embed_layer_0_1_0 = fullyConnectedBnLELU_fullyConnected(network, weightMap, *window_partition_0->getOutput(5), POSEMBED_LAYBERS_OUT_FEATURES, PFN_LAYER_1_OUT_CHANNEL,
                                                            "module.backbone_3d.input_layer.posembed_layers.0.1.0.position_embedding_head.0",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.1.0.position_embedding_head.3",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.1.0.position_embedding_head.1"); 
    auto embed_layer_0_2_0 = fullyConnectedBnLELU_fullyConnected(network, weightMap, *window_partition_0->getOutput(5), POSEMBED_LAYBERS_OUT_FEATURES, PFN_LAYER_1_OUT_CHANNEL,
                                                            "module.backbone_3d.input_layer.posembed_layers.0.2.0.position_embedding_head.0",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.2.0.position_embedding_head.3",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.2.0.position_embedding_head.1"); 
           
    auto embed_layer_0_3_0 = fullyConnectedBnLELU_fullyConnected(network, weightMap, *window_partition_0->getOutput(5), POSEMBED_LAYBERS_OUT_FEATURES, PFN_LAYER_1_OUT_CHANNEL,
                                                            "module.backbone_3d.input_layer.posembed_layers.0.3.0.position_embedding_head.0",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.3.0.position_embedding_head.3",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.3.0.position_embedding_head.1"); 
    
      auto embed_layer_0_0_1 = fullyConnectedBnLELU_fullyConnected(network, weightMap, *window_partition_1->getOutput(5), POSEMBED_LAYBERS_OUT_FEATURES, PFN_LAYER_1_OUT_CHANNEL,
                                                            "module.backbone_3d.input_layer.posembed_layers.0.0.1.position_embedding_head.0",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.0.1.position_embedding_head.3",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.0.1.position_embedding_head.1"); 
    auto embed_layer_0_1_1 = fullyConnectedBnLELU_fullyConnected(network, weightMap, *window_partition_1->getOutput(5), POSEMBED_LAYBERS_OUT_FEATURES, PFN_LAYER_1_OUT_CHANNEL,
                                                            "module.backbone_3d.input_layer.posembed_layers.0.1.1.position_embedding_head.0",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.1.1.position_embedding_head.3",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.1.1.position_embedding_head.1"); 
    auto embed_layer_0_2_1 = fullyConnectedBnLELU_fullyConnected(network, weightMap, *window_partition_1->getOutput(5), POSEMBED_LAYBERS_OUT_FEATURES, PFN_LAYER_1_OUT_CHANNEL,
                                                            "module.backbone_3d.input_layer.posembed_layers.0.2.1.position_embedding_head.0",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.2.1.position_embedding_head.3",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.2.1.position_embedding_head.1"); 
           
    auto embed_layer_0_3_1 = fullyConnectedBnLELU_fullyConnected(network, weightMap, *window_partition_1->getOutput(5), POSEMBED_LAYBERS_OUT_FEATURES, PFN_LAYER_1_OUT_CHANNEL,
                                                            "module.backbone_3d.input_layer.posembed_layers.0.3.1.position_embedding_head.0",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.3.1.position_embedding_head.3",
                                                            "module.backbone_3d.input_layer.posembed_layers.0.3.1.position_embedding_head.1"); 

   

    /*
    
     block_0
    
    
    */

    /*
    
                        multi_head_attntion_0_0
    
    */
   auto get_value_by_index_layer_0_0 = add_get_value_by_index_op(network,torch_scatter_max_1->getOutput(1),embed_layer_0_0_0->getOutput(0),
                                                                                                                                            get_set_op_0->getOutput(0),get_set_op_0->getOutput(2),MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                                            0);

    auto mulit_head_attention_0_0 = multHeadAttention(network,weightMap,*get_value_by_index_layer_0_0->getOutput(0),*get_value_by_index_layer_0_0->getOutput(1),
                                                                                                                    *get_value_by_index_layer_0_0->getOutput(2),*get_set_op_0->getOutput(3),POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                    "module.backbone_3d.stage_0.0.encoder_list.0.win_attn.self_attn.in_proj_weight",
                                                                                                                    "module.backbone_3d.stage_0.0.encoder_list.0.win_attn.self_attn.in_proj_bias",
                                                                                                                    "module.backbone_3d.stage_0.0.encoder_list.0.win_attn.self_attn.out_proj",NUM_HEADS);
    
    auto map_set_feature2voxel_op_0_0 = add_map_set_feature2voxel_op(network,mulit_head_attention_0_0->getOutput(0),
                                                                                                        get_set_op_0->getOutput(0),get_set_op_0->getOutput(2),
                                                                                                        MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,0,
                                                                                                        MAX_PILLARS_NUM);


    auto element_wise_add_0_0_1 = network->addElementWise(*map_set_feature2voxel_op_0_0->getOutput(0),*torch_scatter_max_1->getOutput(1),nvinfer1::ElementWiseOperation::kSUM);

    // multi_head_0_0_     src
    auto layer_norm_layer_0_0_1 = add_layer_norm_op(network, element_wise_add_0_0_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.0.encoder_list.0.win_attn.norm1.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.0.encoder_list.0.win_attn.norm1.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    
    // src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    // src2
    auto multi_head_0_0_linear_gelu_linear = fullyConnected_gelu_fullyConnected(network,weightMap,layer_norm_layer_0_0_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                            SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_1,SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_2,
                                                                                            "module.backbone_3d.stage_0.0.encoder_list.0.win_attn.linear1",
                                                                                            "module.backbone_3d.stage_0.0.encoder_list.0.win_attn.linear2", MAX_PILLARS_NUM,SET_ATTENTION_0_0_GELU_OUT_CHANNEL);
     // src = src + src2
    auto element_wise_add_0_0_2 = network->addElementWise(*layer_norm_layer_0_0_1->getOutput(0),*multi_head_0_0_linear_gelu_linear->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
     auto layer_norm_layer_0_0_2 = add_layer_norm_op(network, element_wise_add_0_0_2->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.0.encoder_list.0.win_attn.norm2.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.0.encoder_list.0.win_attn.norm2.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    auto element_wise_add_0_0_last = network->addElementWise(*layer_norm_layer_0_0_2->getOutput(0),*torch_scatter_max_1->getOutput(1),nvinfer1::ElementWiseOperation::kSUM);

     auto layer_norm_layer_0_0_last = add_layer_norm_op(network, element_wise_add_0_0_last->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.0.encoder_list.0.norm.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.0.encoder_list.0.norm.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
   
    /*
    
                        multi_head_attntion_0_1
    
    */
   auto get_value_by_index_layer_0_1 = add_get_value_by_index_op(network,layer_norm_layer_0_0_last->getOutput(0),embed_layer_0_0_1->getOutput(0),
                                                                                                                                            get_set_op_0->getOutput(0),get_set_op_0->getOutput(2),MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                                            1);
    auto mulit_head_attention_0_1 = multHeadAttention(network,weightMap,*get_value_by_index_layer_0_1->getOutput(0),*get_value_by_index_layer_0_1->getOutput(1),
                                                                                                                    *get_value_by_index_layer_0_1->getOutput(2),*get_set_op_0->getOutput(3),POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                    "module.backbone_3d.stage_0.0.encoder_list.1.win_attn.self_attn.in_proj_weight",
                                                                                                                    "module.backbone_3d.stage_0.0.encoder_list.1.win_attn.self_attn.in_proj_bias",
                                                                                                                    "module.backbone_3d.stage_0.0.encoder_list.1.win_attn.self_attn.out_proj",NUM_HEADS);

    auto map_set_feature2voxel_op_0_1 = add_map_set_feature2voxel_op(network,mulit_head_attention_0_1->getOutput(0),
                                                                                                        get_set_op_0->getOutput(0),get_set_op_0->getOutput(2),
                                                                                                        MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,1,
                                                                                                        MAX_PILLARS_NUM);


    auto element_wise_add_0_1_1 = network->addElementWise(*map_set_feature2voxel_op_0_1->getOutput(0),*layer_norm_layer_0_0_last->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);


      // multi_head_0_1_     src
    auto layer_norm_layer_0_1_1 = add_layer_norm_op(network, element_wise_add_0_1_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.0.encoder_list.1.win_attn.norm1.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.0.encoder_list.1.win_attn.norm1.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    
    // src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    // src2
    auto multi_head_0_1_linear_gelu_linear = fullyConnected_gelu_fullyConnected(network,weightMap,layer_norm_layer_0_1_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                            SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_1,SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_2,
                                                                                            "module.backbone_3d.stage_0.0.encoder_list.1.win_attn.linear1",
                                                                                            "module.backbone_3d.stage_0.0.encoder_list.1.win_attn.linear2", MAX_PILLARS_NUM,SET_ATTENTION_0_0_GELU_OUT_CHANNEL);
     // src = src + src2
    auto element_wise_add_0_1_2 = network->addElementWise(*layer_norm_layer_0_1_1->getOutput(0),*multi_head_0_1_linear_gelu_linear->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
     auto layer_norm_layer_0_1_2 = add_layer_norm_op(network, element_wise_add_0_1_2->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.0.encoder_list.1.win_attn.norm2.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.0.encoder_list.1.win_attn.norm2.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    auto element_wise_add_0_1_last = network->addElementWise(*layer_norm_layer_0_1_2->getOutput(0),*layer_norm_layer_0_0_last->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);

     auto layer_norm_layer_0_1_last = add_layer_norm_op(network, element_wise_add_0_1_last->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.0.encoder_list.1.norm.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.0.encoder_list.1.norm.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);

    auto element_wise_residual_layer_0 = network->addElementWise(*layer_norm_layer_0_1_last->getOutput(0),*torch_scatter_max_1->getOutput(1),nvinfer1::ElementWiseOperation::kSUM);

    auto layer_norm_residual_norm_layer_0 = add_layer_norm_op(network, element_wise_residual_layer_0->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.residual_norm_stage_0.0.weight"],
                                                                                                                weightMap["module.backbone_3d.residual_norm_stage_0.0.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);




     /*
    
     block_1
    
    
    */

    /*
    
                        multi_head_attntion_1_0
    
    */
   auto get_value_by_index_layer_1_0 = add_get_value_by_index_op(network,layer_norm_residual_norm_layer_0->getOutput(0),embed_layer_0_1_0->getOutput(0),
                                                                                                                                            get_set_op_1->getOutput(0),get_set_op_1->getOutput(2),MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                                            0);

    auto mulit_head_attention_1_0 = multHeadAttention(network,weightMap,*get_value_by_index_layer_1_0->getOutput(0),*get_value_by_index_layer_1_0->getOutput(1),
                                                                                                                    *get_value_by_index_layer_1_0->getOutput(2),*get_set_op_1->getOutput(3),POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                    "module.backbone_3d.stage_0.1.encoder_list.0.win_attn.self_attn.in_proj_weight",
                                                                                                                    "module.backbone_3d.stage_0.1.encoder_list.0.win_attn.self_attn.in_proj_bias",
                                                                                                                    "module.backbone_3d.stage_0.1.encoder_list.0.win_attn.self_attn.out_proj",NUM_HEADS);
    
    auto map_set_feature2voxel_op_1_0 = add_map_set_feature2voxel_op(network,mulit_head_attention_1_0->getOutput(0),
                                                                                                        get_set_op_1->getOutput(0),get_set_op_1->getOutput(2),
                                                                                                        MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,0,
                                                                                                        MAX_PILLARS_NUM);


    auto element_wise_add_1_0_1 = network->addElementWise(*map_set_feature2voxel_op_1_0->getOutput(0),*layer_norm_residual_norm_layer_0->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);

    // multi_head_1_0_     src
    auto layer_norm_layer_1_0_1 = add_layer_norm_op(network, element_wise_add_1_0_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.1.encoder_list.0.win_attn.norm1.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.1.encoder_list.0.win_attn.norm1.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    
    // src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    // src2
    auto multi_head_1_0_linear_gelu_linear = fullyConnected_gelu_fullyConnected(network,weightMap,layer_norm_layer_1_0_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                            SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_1,SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_2,
                                                                                            "module.backbone_3d.stage_0.1.encoder_list.0.win_attn.linear1",
                                                                                            "module.backbone_3d.stage_0.1.encoder_list.0.win_attn.linear2", MAX_PILLARS_NUM,SET_ATTENTION_0_0_GELU_OUT_CHANNEL);
     // src = src + src2
    auto element_wise_add_1_0_2 = network->addElementWise(*layer_norm_layer_1_0_1->getOutput(0),*multi_head_1_0_linear_gelu_linear->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
     auto layer_norm_layer_1_0_2 = add_layer_norm_op(network, element_wise_add_1_0_2->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.1.encoder_list.0.win_attn.norm2.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.1.encoder_list.0.win_attn.norm2.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    auto element_wise_add_1_0_last = network->addElementWise(*layer_norm_layer_1_0_2->getOutput(0),*layer_norm_residual_norm_layer_0->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);

     auto layer_norm_layer_1_0_last = add_layer_norm_op(network, element_wise_add_1_0_last->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.1.encoder_list.0.norm.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.1.encoder_list.0.norm.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
   
    /*
    
                        multi_head_attntion_1_1
    
    */
   auto get_value_by_index_layer_1_1 = add_get_value_by_index_op(network,layer_norm_layer_1_0_last->getOutput(0),embed_layer_0_1_1->getOutput(0),
                                                                                                                                            get_set_op_1->getOutput(0),get_set_op_1->getOutput(2),MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                                            1);
    auto mulit_head_attention_1_1 = multHeadAttention(network,weightMap,*get_value_by_index_layer_1_1->getOutput(0),*get_value_by_index_layer_1_1->getOutput(1),
                                                                                                                    *get_value_by_index_layer_1_1->getOutput(2),*get_set_op_1->getOutput(3),POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                    "module.backbone_3d.stage_0.1.encoder_list.1.win_attn.self_attn.in_proj_weight",
                                                                                                                    "module.backbone_3d.stage_0.1.encoder_list.1.win_attn.self_attn.in_proj_bias",
                                                                                                                    "module.backbone_3d.stage_0.1.encoder_list.1.win_attn.self_attn.out_proj",NUM_HEADS);

    auto map_set_feature2voxel_op_1_1 = add_map_set_feature2voxel_op(network,mulit_head_attention_1_1->getOutput(0),
                                                                                                        get_set_op_1->getOutput(0),get_set_op_1->getOutput(2),
                                                                                                        MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,1,
                                                                                                        MAX_PILLARS_NUM);


    auto element_wise_add_1_1_1 = network->addElementWise(*map_set_feature2voxel_op_1_1->getOutput(0),*layer_norm_layer_1_0_last->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);


      // multi_head_1_1_     src
    auto layer_norm_layer_1_1_1 = add_layer_norm_op(network, element_wise_add_1_1_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.1.encoder_list.1.win_attn.norm1.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.1.encoder_list.1.win_attn.norm1.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    
    // src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    // src2
    auto multi_head_1_1_linear_gelu_linear = fullyConnected_gelu_fullyConnected(network,weightMap,layer_norm_layer_1_1_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                            SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_1,SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_2,
                                                                                            "module.backbone_3d.stage_0.1.encoder_list.1.win_attn.linear1",
                                                                                            "module.backbone_3d.stage_0.1.encoder_list.1.win_attn.linear2", MAX_PILLARS_NUM,SET_ATTENTION_0_0_GELU_OUT_CHANNEL);
     // src = src + src2
    auto element_wise_add_1_1_2 = network->addElementWise(*layer_norm_layer_1_1_1->getOutput(0),*multi_head_1_1_linear_gelu_linear->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
     auto layer_norm_layer_1_1_2 = add_layer_norm_op(network, element_wise_add_1_1_2->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.1.encoder_list.1.win_attn.norm2.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.1.encoder_list.1.win_attn.norm2.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    auto element_wise_add_1_1_last = network->addElementWise(*layer_norm_layer_1_1_2->getOutput(0),*layer_norm_layer_1_0_last->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);

     auto layer_norm_layer_1_1_last = add_layer_norm_op(network, element_wise_add_1_1_last->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.1.encoder_list.1.norm.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.1.encoder_list.1.norm.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);

    auto element_wise_residual_layer_1 = network->addElementWise(*layer_norm_layer_1_1_last->getOutput(0),*layer_norm_residual_norm_layer_0->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);

    auto layer_norm_residual_norm_layer_1 = add_layer_norm_op(network, element_wise_residual_layer_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.residual_norm_stage_0.1.weight"],
                                                                                                                weightMap["module.backbone_3d.residual_norm_stage_0.1.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);







     /*
    
     block_2
    
    
    */

    /*
    
                        multi_head_attntion_2_0
    
    */
   auto get_value_by_index_layer_2_0 = add_get_value_by_index_op(network,layer_norm_residual_norm_layer_1->getOutput(0),embed_layer_0_2_0->getOutput(0),
                                                                                                                                            get_set_op_0->getOutput(0),get_set_op_0->getOutput(2),MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                                            0);

    auto mulit_head_attention_2_0 = multHeadAttention(network,weightMap,*get_value_by_index_layer_2_0->getOutput(0),*get_value_by_index_layer_2_0->getOutput(1),
                                                                                                                    *get_value_by_index_layer_2_0->getOutput(2),*get_set_op_0->getOutput(3),POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                    "module.backbone_3d.stage_0.2.encoder_list.0.win_attn.self_attn.in_proj_weight",
                                                                                                                    "module.backbone_3d.stage_0.2.encoder_list.0.win_attn.self_attn.in_proj_bias",
                                                                                                                    "module.backbone_3d.stage_0.2.encoder_list.0.win_attn.self_attn.out_proj",NUM_HEADS);
    
    auto map_set_feature2voxel_op_2_0 = add_map_set_feature2voxel_op(network,mulit_head_attention_2_0->getOutput(0),
                                                                                                        get_set_op_0->getOutput(0),get_set_op_0->getOutput(2),
                                                                                                        MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,0,
                                                                                                        MAX_PILLARS_NUM);


    auto element_wise_add_2_0_1 = network->addElementWise(*map_set_feature2voxel_op_2_0->getOutput(0),*layer_norm_residual_norm_layer_1->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);

    // multi_head_2_0_     src
    auto layer_norm_layer_2_0_1 = add_layer_norm_op(network, element_wise_add_2_0_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.2.encoder_list.0.win_attn.norm1.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.2.encoder_list.0.win_attn.norm1.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    
    // src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    // src2
    auto multi_head_2_0_linear_gelu_linear = fullyConnected_gelu_fullyConnected(network,weightMap,layer_norm_layer_2_0_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                            SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_1,SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_2,
                                                                                            "module.backbone_3d.stage_0.2.encoder_list.0.win_attn.linear1",
                                                                                            "module.backbone_3d.stage_0.2.encoder_list.0.win_attn.linear2", MAX_PILLARS_NUM,SET_ATTENTION_0_0_GELU_OUT_CHANNEL);
     // src = src + src2
    auto element_wise_add_2_0_2 = network->addElementWise(*layer_norm_layer_2_0_1->getOutput(0),*multi_head_2_0_linear_gelu_linear->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
     auto layer_norm_layer_2_0_2 = add_layer_norm_op(network, element_wise_add_2_0_2->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.2.encoder_list.0.win_attn.norm2.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.2.encoder_list.0.win_attn.norm2.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    auto element_wise_add_2_0_last = network->addElementWise(*layer_norm_layer_2_0_2->getOutput(0),*layer_norm_residual_norm_layer_1->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);

     auto layer_norm_layer_2_0_last = add_layer_norm_op(network, element_wise_add_2_0_last->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.2.encoder_list.0.norm.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.2.encoder_list.0.norm.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
   
    /*
    
                        multi_head_attntion_2_1
    
    */
   auto get_value_by_index_layer_2_1 = add_get_value_by_index_op(network,layer_norm_layer_2_0_last->getOutput(0),embed_layer_0_2_1->getOutput(0),
                                                                                                                                            get_set_op_0->getOutput(0),get_set_op_0->getOutput(2),MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                                            1);
    auto mulit_head_attention_2_1 = multHeadAttention(network,weightMap,*get_value_by_index_layer_2_1->getOutput(0),*get_value_by_index_layer_2_1->getOutput(1),
                                                                                                                    *get_value_by_index_layer_2_1->getOutput(2),*get_set_op_0->getOutput(3),POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                    "module.backbone_3d.stage_0.2.encoder_list.1.win_attn.self_attn.in_proj_weight",
                                                                                                                    "module.backbone_3d.stage_0.2.encoder_list.1.win_attn.self_attn.in_proj_bias",
                                                                                                                    "module.backbone_3d.stage_0.2.encoder_list.1.win_attn.self_attn.out_proj",NUM_HEADS);

    auto map_set_feature2voxel_op_2_1 = add_map_set_feature2voxel_op(network,mulit_head_attention_2_1->getOutput(0),
                                                                                                        get_set_op_0->getOutput(0),get_set_op_0->getOutput(2),
                                                                                                        MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,1,
                                                                                                        MAX_PILLARS_NUM);


    auto element_wise_add_2_1_1 = network->addElementWise(*map_set_feature2voxel_op_2_1->getOutput(0),*layer_norm_layer_2_0_last->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);


      // multi_head_2_1_     src
    auto layer_norm_layer_2_1_1 = add_layer_norm_op(network, element_wise_add_2_1_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.2.encoder_list.1.win_attn.norm1.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.2.encoder_list.1.win_attn.norm1.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    
    // src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    // src2
    auto multi_head_2_1_linear_gelu_linear = fullyConnected_gelu_fullyConnected(network,weightMap,layer_norm_layer_2_1_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                            SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_1,SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_2,
                                                                                            "module.backbone_3d.stage_0.2.encoder_list.1.win_attn.linear1",
                                                                                            "module.backbone_3d.stage_0.2.encoder_list.1.win_attn.linear2", MAX_PILLARS_NUM,SET_ATTENTION_0_0_GELU_OUT_CHANNEL);
     // src = src + src2
    auto element_wise_add_2_1_2 = network->addElementWise(*layer_norm_layer_2_1_1->getOutput(0),*multi_head_2_1_linear_gelu_linear->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
     auto layer_norm_layer_2_1_2 = add_layer_norm_op(network, element_wise_add_2_1_2->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.2.encoder_list.1.win_attn.norm2.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.2.encoder_list.1.win_attn.norm2.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    auto element_wise_add_2_1_last = network->addElementWise(*layer_norm_layer_2_1_2->getOutput(0),*layer_norm_layer_2_0_last->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);

     auto layer_norm_layer_2_1_last = add_layer_norm_op(network, element_wise_add_2_1_last->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.2.encoder_list.1.norm.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.2.encoder_list.1.norm.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);

    auto element_wise_residual_layer_2 = network->addElementWise(*layer_norm_layer_2_1_last->getOutput(0),*layer_norm_residual_norm_layer_1->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);

    auto layer_norm_residual_norm_layer_2 = add_layer_norm_op(network, element_wise_residual_layer_2->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.residual_norm_stage_0.2.weight"],
                                                                                                                weightMap["module.backbone_3d.residual_norm_stage_0.2.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);





     /*
    
     block_3
    
    
    */

    /*
    
                        multi_head_attntion_3_0
    
    */
   auto get_value_by_index_layer_3_0 = add_get_value_by_index_op(network,layer_norm_residual_norm_layer_2->getOutput(0),embed_layer_0_3_0->getOutput(0),
                                                                                                                                            get_set_op_1->getOutput(0),get_set_op_1->getOutput(2),MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                                            0);

    auto mulit_head_attention_3_0 = multHeadAttention(network,weightMap,*get_value_by_index_layer_3_0->getOutput(0),*get_value_by_index_layer_3_0->getOutput(1),
                                                                                                                    *get_value_by_index_layer_3_0->getOutput(2),*get_set_op_1->getOutput(3),POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                    "module.backbone_3d.stage_0.3.encoder_list.0.win_attn.self_attn.in_proj_weight",
                                                                                                                    "module.backbone_3d.stage_0.3.encoder_list.0.win_attn.self_attn.in_proj_bias",
                                                                                                                    "module.backbone_3d.stage_0.3.encoder_list.0.win_attn.self_attn.out_proj",NUM_HEADS);
    
    auto map_set_feature2voxel_op_3_0 = add_map_set_feature2voxel_op(network,mulit_head_attention_3_0->getOutput(0),
                                                                                                        get_set_op_1->getOutput(0),get_set_op_1->getOutput(2),
                                                                                                        MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,0,
                                                                                                        MAX_PILLARS_NUM);


    auto element_wise_add_3_0_1 = network->addElementWise(*map_set_feature2voxel_op_3_0->getOutput(0),*layer_norm_residual_norm_layer_2->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);

    // multi_head_3_0_     src
    auto layer_norm_layer_3_0_1 = add_layer_norm_op(network, element_wise_add_3_0_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.3.encoder_list.0.win_attn.norm1.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.3.encoder_list.0.win_attn.norm1.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    
    // src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    // src2
    auto multi_head_3_0_linear_gelu_linear = fullyConnected_gelu_fullyConnected(network,weightMap,layer_norm_layer_3_0_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                            SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_1,SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_2,
                                                                                            "module.backbone_3d.stage_0.3.encoder_list.0.win_attn.linear1",
                                                                                            "module.backbone_3d.stage_0.3.encoder_list.0.win_attn.linear2", MAX_PILLARS_NUM,SET_ATTENTION_0_0_GELU_OUT_CHANNEL);
     // src = src + src2
    auto element_wise_add_3_0_2 = network->addElementWise(*layer_norm_layer_3_0_1->getOutput(0),*multi_head_3_0_linear_gelu_linear->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
     auto layer_norm_layer_3_0_2 = add_layer_norm_op(network, element_wise_add_3_0_2->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.3.encoder_list.0.win_attn.norm2.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.3.encoder_list.0.win_attn.norm2.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    auto element_wise_add_3_0_last = network->addElementWise(*layer_norm_layer_3_0_2->getOutput(0),*layer_norm_residual_norm_layer_2->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);

     auto layer_norm_layer_3_0_last = add_layer_norm_op(network, element_wise_add_3_0_last->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.3.encoder_list.0.norm.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.3.encoder_list.0.norm.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
   
    /*
    
                        multi_head_attntion_3_1
    
    */
   auto get_value_by_index_layer_3_1 = add_get_value_by_index_op(network,layer_norm_layer_3_0_last->getOutput(0),embed_layer_0_3_1->getOutput(0),
                                                                                                                                            get_set_op_1->getOutput(0),get_set_op_1->getOutput(2),MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                                            1);
    auto mulit_head_attention_3_1 = multHeadAttention(network,weightMap,*get_value_by_index_layer_3_1->getOutput(0),*get_value_by_index_layer_3_1->getOutput(1),
                                                                                                                    *get_value_by_index_layer_3_1->getOutput(2),*get_set_op_1->getOutput(3),POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                    "module.backbone_3d.stage_0.3.encoder_list.1.win_attn.self_attn.in_proj_weight",
                                                                                                                    "module.backbone_3d.stage_0.3.encoder_list.1.win_attn.self_attn.in_proj_bias",
                                                                                                                    "module.backbone_3d.stage_0.3.encoder_list.1.win_attn.self_attn.out_proj",NUM_HEADS);

    auto map_set_feature2voxel_op_3_1 = add_map_set_feature2voxel_op(network,mulit_head_attention_3_1->getOutput(0),
                                                                                                        get_set_op_1->getOutput(0),get_set_op_1->getOutput(2),
                                                                                                        MAX_WIN_NUM,VOXEL_NUM_SET,POSEMBED_LAYBERS_OUT_FEATURES,1,
                                                                                                        MAX_PILLARS_NUM);


    auto element_wise_add_3_1_1 = network->addElementWise(*map_set_feature2voxel_op_3_1->getOutput(0),*layer_norm_layer_3_0_last->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);


      // multi_head_3_1_     src
    auto layer_norm_layer_3_1_1 = add_layer_norm_op(network, element_wise_add_3_1_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.3.encoder_list.1.win_attn.norm1.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.3.encoder_list.1.win_attn.norm1.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    
    // src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    // src2
    auto multi_head_3_1_linear_gelu_linear = fullyConnected_gelu_fullyConnected(network,weightMap,layer_norm_layer_3_1_1->getOutput(0),voxelGenerator->getOutput(4),
                                                                                            SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_1,SET_ATTENTION_0_0_OUT_CHANNEL_LINEAR_2,
                                                                                            "module.backbone_3d.stage_0.3.encoder_list.1.win_attn.linear1",
                                                                                            "module.backbone_3d.stage_0.3.encoder_list.1.win_attn.linear2", MAX_PILLARS_NUM,SET_ATTENTION_0_0_GELU_OUT_CHANNEL);
     // src = src + src2
    auto element_wise_add_3_1_2 = network->addElementWise(*layer_norm_layer_3_1_1->getOutput(0),*multi_head_3_1_linear_gelu_linear->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
     auto layer_norm_layer_3_1_2 = add_layer_norm_op(network, element_wise_add_3_1_2->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.3.encoder_list.1.win_attn.norm2.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.3.encoder_list.1.win_attn.norm2.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);
    auto element_wise_add_3_1_last = network->addElementWise(*layer_norm_layer_3_1_2->getOutput(0),*layer_norm_layer_3_0_last->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);

     auto layer_norm_layer_3_1_last = add_layer_norm_op(network, element_wise_add_3_1_last->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.stage_0.3.encoder_list.1.norm.weight"],
                                                                                                                weightMap["module.backbone_3d.stage_0.3.encoder_list.1.norm.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);

    auto element_wise_residual_layer_3 = network->addElementWise(*layer_norm_layer_3_1_last->getOutput(0),*layer_norm_residual_norm_layer_2->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);

    auto layer_norm_residual_norm_layer_3 = add_layer_norm_op(network, element_wise_residual_layer_3->getOutput(0),voxelGenerator->getOutput(4),
                                                                                                                        weightMap["module.backbone_3d.residual_norm_stage_0.3.weight"],
                                                                                                                weightMap["module.backbone_3d.residual_norm_stage_0.3.bias"],
                                                                                                                 MAX_PILLARS_NUM,POSEMBED_LAYBERS_OUT_FEATURES,POSEMBED_LAYBERS_OUT_FEATURES,
                                                                                                                EPS);


    /*
    
    map2bev
    
    */
   auto map2bev_op = add_map_2_bev_op(network,layer_norm_residual_norm_layer_3->getOutput(0),voxelGenerator->getOutput(2),voxelGenerator->getOutput(4),MAX_PILLARS_NUM,
                                                                                                                                                    POSEMBED_LAYBERS_OUT_FEATURES,GRID_SIZE_X,GRID_SIZE_Y);
    // 1 468 468 192  ---> 1 192 468 468

    auto map2bev_op_transpose =  network->addShuffle(*map2bev_op->getOutput(0));
    map2bev_op_transpose->setFirstTranspose(Permutation{0,3,1,2});



    /*
            backbone  2d     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    */
  /*
    backbone_2d block_0
  */
//block_0_0
auto backbone2d_block_0_0_1 = convBnLELU(network,weightMap,*map2bev_op_transpose->getOutput(0),
                                                                                                       BACKBONE_2D_BLOCK_0_0_1_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_0_0_1_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_0_0_1_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_0_0_1_PADDING,
                                                                                                       "module.backbone_2d.blocks.0.0.conv1",
                                                                                                       "module.backbone_2d.blocks.0.0.bn1" );
auto backbone2d_block_0_0_2 = convBn(network,weightMap,*backbone2d_block_0_0_1->getOutput(0),    // 
                                                                                                       BACKBONE_2D_BLOCK_0_0_2_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_0_0_2_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_0_0_2_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_0_0_2_PADDING,
                                                                                                       "module.backbone_2d.blocks.0.0.conv2",
                                                                                                       "module.backbone_2d.blocks.0.0.bn2");
auto backbone2d_block_0_0_downsample_layer = convBn(network,weightMap,*map2bev_op_transpose->getOutput(0),
                                                                                                       BACKBONE_2D_BLOCK_0_0_DOWNSAMPLE_LAYER_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_0_0_DOWNSAMPLE_LAYER_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_0_0_DOWNSAMPLE_LAYER_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_0_0_DOWNSAMPLE_LAYER_PADDING,
                                                                                                       "module.backbone_2d.blocks.0.0.downsample_layer.0",
                                                                                                       "module.backbone_2d.blocks.0.0.downsample_layer.1");
auto element_wise_backbone2d_block_0_0 = network->addElementWise(*backbone2d_block_0_0_2->getOutput(0),*backbone2d_block_0_0_downsample_layer->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
auto  relu_backbone2d_block_0_0= network->addActivation(*element_wise_backbone2d_block_0_0->getOutput(0), ActivationType::kRELU);
relu_backbone2d_block_0_0->setAlpha(1e-8);

//block_0_1
auto backbone2d_block_0_1_1 = convBnLELU(network,weightMap,*relu_backbone2d_block_0_0->getOutput(0),
                                                                                                       BACKBONE_2D_BLOCK_0_1_1_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_0_1_1_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_0_1_1_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_0_1_1_PADDING,
                                                                                                       "module.backbone_2d.blocks.0.1.conv1",
                                                                                                       "module.backbone_2d.blocks.0.1.bn1" );

auto backbone2d_block_0_1_2 = convBn(network,weightMap,*backbone2d_block_0_1_1->getOutput(0),    // 
                                                                                                       BACKBONE_2D_BLOCK_0_1_2_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_0_1_2_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_0_1_2_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_0_1_2_PADDING,
                                                                                                       "module.backbone_2d.blocks.0.1.conv2",
                                                                                                       "module.backbone_2d.blocks.0.1.bn2");
auto element_wise_backbone2d_block_0_1 = network->addElementWise(*backbone2d_block_0_1_2->getOutput(0),*relu_backbone2d_block_0_0->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
auto  relu_backbone2d_block_0_1= network->addActivation(*element_wise_backbone2d_block_0_1->getOutput(0), ActivationType::kRELU);
relu_backbone2d_block_0_1->setAlpha(1e-8);

//deblock_0
auto backbone2d_deblock_0  = deconvBnLELU(network,weightMap,*relu_backbone2d_block_0_1->getOutput(0),
                                                                                                    BACKBONE_2D_BLOCK_0_DECONV_OUT_CHANNEL,
                                                                                                    BACKBONE_2D_BLOCK_0_DECONV_KERNEL_SIZE,
                                                                                                    BACKBONE_2D_BLOCK_0_DECONV_STRIDE,
                                                                                                    BACKBONE_2D_BLOCK_0_DECONV_PADDING,
                                                                                                    "module.backbone_2d.deblocks.0.0",
                                                                                                    "module.backbone_2d.deblocks.0.1");

/*
      backbone_2d   block_1
*/
//block_1_0

auto backbone2d_block_1_0_1 = convBnLELU(network,weightMap,*relu_backbone2d_block_0_1->getOutput(0),
                                                                                                       BACKBONE_2D_BLOCK_1_0_1_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_1_0_1_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_1_0_1_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_1_0_1_PADDING,
                                                                                                       "module.backbone_2d.blocks.1.0.conv1",
                                                                                                       "module.backbone_2d.blocks.1.0.bn1" );
auto backbone2d_block_1_0_2 = convBn(network,weightMap,*backbone2d_block_1_0_1->getOutput(0),    // 
                                                                                                       BACKBONE_2D_BLOCK_1_0_2_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_1_0_2_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_1_0_2_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_1_0_2_PADDING,
                                                                                                       "module.backbone_2d.blocks.1.0.conv2",
                                                                                                       "module.backbone_2d.blocks.1.0.bn2");
auto backbone2d_block_1_0_downsample_layer = convBn(network,weightMap,*relu_backbone2d_block_0_1->getOutput(0),
                                                                                                       BACKBONE_2D_BLOCK_1_0_DOWNSAMPLE_LAYER_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_1_0_DOWNSAMPLE_LAYER_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_1_0_DOWNSAMPLE_LAYER_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_1_0_DOWNSAMPLE_LAYER_PADDING,
                                                                                                       "module.backbone_2d.blocks.1.0.downsample_layer.0",
                                                                                                       "module.backbone_2d.blocks.1.0.downsample_layer.1");
auto element_wise_backbone2d_block_1_0 = network->addElementWise(*backbone2d_block_1_0_2->getOutput(0),*backbone2d_block_1_0_downsample_layer->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
auto  relu_backbone2d_block_1_0= network->addActivation(*element_wise_backbone2d_block_1_0->getOutput(0), ActivationType::kRELU);
relu_backbone2d_block_1_0->setAlpha(1e-8);

//block_1_1

auto backbone2d_block_1_1_1 = convBnLELU(network,weightMap,*relu_backbone2d_block_1_0->getOutput(0),
                                                                                                       BACKBONE_2D_BLOCK_1_1_1_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_1_1_1_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_1_1_1_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_1_1_1_PADDING,
                                                                                                       "module.backbone_2d.blocks.1.1.conv1",
                                                                                                       "module.backbone_2d.blocks.1.1.bn1" );

auto backbone2d_block_1_1_2 = convBn(network,weightMap,*backbone2d_block_1_1_1->getOutput(0),    // 
                                                                                                       BACKBONE_2D_BLOCK_1_1_2_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_1_1_2_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_1_1_2_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_1_1_2_PADDING,
                                                                                                       "module.backbone_2d.blocks.1.1.conv2",
                                                                                                       "module.backbone_2d.blocks.1.1.bn2");
auto element_wise_backbone2d_block_1_1 = network->addElementWise(*backbone2d_block_1_1_2->getOutput(0),*relu_backbone2d_block_1_0->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
auto  relu_backbone2d_block_1_1= network->addActivation(*element_wise_backbone2d_block_1_1->getOutput(0), ActivationType::kRELU);
relu_backbone2d_block_1_1->setAlpha(1e-8);

//block_1_2

auto backbone2d_block_1_2_1 = convBnLELU(network,weightMap,*relu_backbone2d_block_1_1->getOutput(0),
                                                                                                       BACKBONE_2D_BLOCK_1_2_1_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_1_2_1_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_1_2_1_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_1_2_1_PADDING,
                                                                                                       "module.backbone_2d.blocks.1.2.conv1",
                                                                                                       "module.backbone_2d.blocks.1.2.bn1" );

auto backbone2d_block_1_2_2 = convBn(network,weightMap,*backbone2d_block_1_2_1->getOutput(0),    // 
                                                                                                       BACKBONE_2D_BLOCK_1_2_2_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_1_2_2_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_1_2_2_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_1_2_2_PADDING,
                                                                                                       "module.backbone_2d.blocks.1.2.conv2",
                                                                                                       "module.backbone_2d.blocks.1.2.bn2");
auto element_wise_backbone2d_block_1_2 = network->addElementWise(*backbone2d_block_1_2_2->getOutput(0),*relu_backbone2d_block_1_1->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
auto  relu_backbone2d_block_1_2= network->addActivation(*element_wise_backbone2d_block_1_2->getOutput(0), ActivationType::kRELU);
relu_backbone2d_block_1_2->setAlpha(1e-8);

//deblock_1
auto backbone2d_deblock_1  = deconvBnLELU(network,weightMap,*relu_backbone2d_block_1_2->getOutput(0),
                                                                                                    BACKBONE_2D_BLOCK_1_DECONV_OUT_CHANNEL,
                                                                                                    BACKBONE_2D_BLOCK_1_DECONV_KERNEL_SIZE,
                                                                                                    BACKBONE_2D_BLOCK_1_DECONV_STRIDE,
                                                                                                    BACKBONE_2D_BLOCK_1_DECONV_PADDING,
                                                                                                    "module.backbone_2d.deblocks.1.0",
                                                                                                    "module.backbone_2d.deblocks.1.1");



/*
      backbone_2d   block_2
*/
//block_2_0

auto backbone2d_block_2_0_1 = convBnLELU(network,weightMap,*relu_backbone2d_block_1_2->getOutput(0),
                                                                                                       BACKBONE_2D_BLOCK_2_0_1_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_2_0_1_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_2_0_1_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_2_0_1_PADDING,
                                                                                                       "module.backbone_2d.blocks.2.0.conv1",
                                                                                                       "module.backbone_2d.blocks.2.0.bn1" );
auto backbone2d_block_2_0_2 = convBn(network,weightMap,*backbone2d_block_2_0_1->getOutput(0),    // 
                                                                                                       BACKBONE_2D_BLOCK_2_0_2_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_2_0_2_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_2_0_2_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_2_0_2_PADDING,
                                                                                                       "module.backbone_2d.blocks.2.0.conv2",
                                                                                                       "module.backbone_2d.blocks.2.0.bn2");
auto backbone2d_block_2_0_downsample_layer = convBn(network,weightMap,*relu_backbone2d_block_1_2->getOutput(0),
                                                                                                       BACKBONE_2D_BLOCK_2_0_DOWNSAMPLE_LAYER_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_2_0_DOWNSAMPLE_LAYER_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_2_0_DOWNSAMPLE_LAYER_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_2_0_DOWNSAMPLE_LAYER_PADDING,
                                                                                                       "module.backbone_2d.blocks.2.0.downsample_layer.0",
                                                                                                       "module.backbone_2d.blocks.2.0.downsample_layer.1");
auto element_wise_backbone2d_block_2_0 = network->addElementWise(*backbone2d_block_2_0_2->getOutput(0),*backbone2d_block_2_0_downsample_layer->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
auto  relu_backbone2d_block_2_0= network->addActivation(*element_wise_backbone2d_block_2_0->getOutput(0), ActivationType::kRELU);
relu_backbone2d_block_2_0->setAlpha(1e-8);

//block_2_1

auto backbone2d_block_2_1_1 = convBnLELU(network,weightMap,*relu_backbone2d_block_2_0->getOutput(0),
                                                                                                       BACKBONE_2D_BLOCK_2_1_1_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_2_1_1_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_2_1_1_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_2_1_1_PADDING,
                                                                                                       "module.backbone_2d.blocks.2.1.conv1",
                                                                                                       "module.backbone_2d.blocks.2.1.bn1" );

auto backbone2d_block_2_1_2 = convBn(network,weightMap,*backbone2d_block_2_1_1->getOutput(0),    // 
                                                                                                       BACKBONE_2D_BLOCK_2_1_2_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_2_1_2_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_2_1_2_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_2_1_2_PADDING,
                                                                                                       "module.backbone_2d.blocks.2.1.conv2",
                                                                                                       "module.backbone_2d.blocks.2.1.bn2");
auto element_wise_backbone2d_block_2_1 = network->addElementWise(*backbone2d_block_2_1_2->getOutput(0),*relu_backbone2d_block_2_0->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
auto  relu_backbone2d_block_2_1= network->addActivation(*element_wise_backbone2d_block_2_1->getOutput(0), ActivationType::kRELU);
relu_backbone2d_block_2_1->setAlpha(1e-8);

//block_2_2

auto backbone2d_block_2_2_1 = convBnLELU(network,weightMap,*relu_backbone2d_block_2_1->getOutput(0),
                                                                                                       BACKBONE_2D_BLOCK_2_2_1_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_2_2_1_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_2_2_1_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_2_2_1_PADDING,
                                                                                                       "module.backbone_2d.blocks.2.2.conv1",
                                                                                                       "module.backbone_2d.blocks.2.2.bn1" );

auto backbone2d_block_2_2_2 = convBn(network,weightMap,*backbone2d_block_2_2_1->getOutput(0),    // 
                                                                                                       BACKBONE_2D_BLOCK_2_2_2_OUT_CHANNEL,
                                                                                                       BACKBONE_2D_BLOCK_2_2_2_KERNEL_SIZE,
                                                                                                       BACKBONE_2D_BLOCK_2_2_2_STRIDE,
                                                                                                       BACKBONE_2D_BLOCK_2_2_2_PADDING,
                                                                                                       "module.backbone_2d.blocks.2.2.conv2",
                                                                                                       "module.backbone_2d.blocks.2.2.bn2");
auto element_wise_backbone2d_block_2_2 = network->addElementWise(*backbone2d_block_2_2_2->getOutput(0),*relu_backbone2d_block_2_1->getOutput(0),nvinfer1::ElementWiseOperation::kSUM);
auto  relu_backbone2d_block_2_2= network->addActivation(*element_wise_backbone2d_block_2_2->getOutput(0), ActivationType::kRELU);
relu_backbone2d_block_2_2->setAlpha(1e-8);

//deblock_1
auto backbone2d_deblock_2  = deconvBnLELU(network,weightMap,*relu_backbone2d_block_2_2->getOutput(0),
                                                                                                    BACKBONE_2D_BLOCK_2_DECONV_OUT_CHANNEL,
                                                                                                    BACKBONE_2D_BLOCK_2_DECONV_KERNEL_SIZE,
                                                                                                    BACKBONE_2D_BLOCK_2_DECONV_STRIDE,
                                                                                                    BACKBONE_2D_BLOCK_2_DECONV_PADDING,
                                                                                                    "module.backbone_2d.deblocks.2.0",
                                                                                                    "module.backbone_2d.deblocks.2.1");

    ITensor* backbone2d_inputTensors[] = {backbone2d_deblock_0->getOutput(0), backbone2d_deblock_1->getOutput(0),backbone2d_deblock_2->getOutput(0)};
    auto backbone2d_cat_tensor = network->addConcatenation(backbone2d_inputTensors, 3);
    backbone2d_cat_tensor->setAxis(1);

//      /*
//             center_head <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
//     */
auto center_head_shared_conv  = convBnLELU(network,weightMap,*backbone2d_cat_tensor->getOutput(0),
                                                                                                       CENTER_HEAD_SHARED_CONV_OUT_CHANNEL,
                                                                                                       CENTER_HEAD_SHARED_CONV_KERNEL_SIZE,
                                                                                                       CENTER_HEAD_SHARED_CONV_STRIDE,
                                                                                                       CENTER_HEAD_SHARED_CONV_PADDING,
                                                                                                       "module.dense_head.shared_conv.0",
                                                                                                       "module.dense_head.shared_conv.1" );

//  center   
auto center_header  = convBnLELU(network,weightMap,*center_head_shared_conv->getOutput(0),
                                                                                                       CENTER_SEQ_0_CONV_0_OUT_CHANNEL,
                                                                                                       CENTER_SEQ_0_CONV_0_KERNEL_SIZE,
                                                                                                       CENTER_SEQ_0_CONV_0_STRIDE,
                                                                                                       CENTER_SEQ_0_CONV_0_IN_PADDING,
                                                                                                       "module.dense_head.heads_list.0.center.0.0",
                                                                                                       "module.dense_head.heads_list.0.center.0.1" );
auto center_header_last = conv_with_bias(network,weightMap,*center_header->getOutput(0),
                                                                                                       CENTER_CONV_1_OUT_CHANNEL,
                                                                                                       CENTER_CONV_1_KERNEL_SIZE,
                                                                                                       CENTER_CONV_1_STRIDE,
                                                                                                       CENTER_CONV_1_IN_PADDING,
                                                                                                       "module.dense_head.heads_list.0.center.1");
                                                

//center_z

auto center_z_header  = convBnLELU(network,weightMap,*center_head_shared_conv->getOutput(0),
                                                                                                       CENTER_Z_SEQ_0_CONV_0_OUT_CHANNEL,
                                                                                                       CENTER_Z_SEQ_0_CONV_0_KERNEL_SIZE,
                                                                                                       CENTER_Z_SEQ_0_CONV_0_STRIDE,
                                                                                                       CENTER_Z_SEQ_0_CONV_0_IN_PADDING,
                                                                                                       "module.dense_head.heads_list.0.center_z.0.0",
                                                                                                       "module.dense_head.heads_list.0.center_z.0.1" );
auto center_z_header_last = conv_with_bias(network,weightMap,*center_z_header->getOutput(0),
                                                                                                       CENTER_Z_CONV_1_OUT_CHANNEL,
                                                                                                       CENTER_Z_CONV_1_KERNEL_SIZE,
                                                                                                       CENTER_Z_CONV_1_STRIDE,
                                                                                                       CENTER_Z_CONV_1_IN_PADDING,
                                                                                                       "module.dense_head.heads_list.0.center_z.1");

//dim
auto dim_header  = convBnLELU(network,weightMap,*center_head_shared_conv->getOutput(0),
                                                                                                       DIM_SEQ_0_CONV_0_OUT_CHANNEL,
                                                                                                       DIM_SEQ_0_CONV_0_KERNEL_SIZE,
                                                                                                       DIM_SEQ_0_CONV_0_STRIDE,
                                                                                                       DIM_SEQ_0_CONV_0_IN_PADDING,
                                                                                                       "module.dense_head.heads_list.0.dim.0.0",
                                                                                                       "module.dense_head.heads_list.0.dim.0.1" );
auto dim_header_last = conv_with_bias(network,weightMap,*dim_header->getOutput(0),
                                                                                                       DIM_CONV_1_OUT_CHANNEL,
                                                                                                       DIM_CONV_1_KERNEL_SIZE,
                                                                                                       DIM_CONV_1_STRIDE,
                                                                                                       DIM_CONV_1_IN_PADDING,
                                                                                                       "module.dense_head.heads_list.0.dim.1");

//rot
auto rot_header  = convBnLELU(network,weightMap,*center_head_shared_conv->getOutput(0),
                                                                                                       ROT_SEQ_0_CONV_0_OUT_CHANNEL,
                                                                                                       ROT_SEQ_0_CONV_0_KERNEL_SIZE,
                                                                                                       ROT_SEQ_0_CONV_0_STRIDE,
                                                                                                       ROT_SEQ_0_CONV_0_IN_PADDING,
                                                                                                       "module.dense_head.heads_list.0.rot.0.0",
                                                                                                       "module.dense_head.heads_list.0.rot.0.1" );
auto rot_header_last = conv_with_bias(network,weightMap,*rot_header->getOutput(0),
                                                                                                       ROT_CONV_1_OUT_CHANNEL,
                                                                                                       ROT_CONV_1_KERNEL_SIZE,
                                                                                                       ROT_CONV_1_STRIDE,
                                                                                                       ROT_CONV_1_IN_PADDING,
                                                                                                       "module.dense_head.heads_list.0.rot.1");

// iou
auto iou_header  = convBnLELU(network,weightMap,*center_head_shared_conv->getOutput(0),
                                                                                                       IOU_SEQ_0_CONV_0_OUT_CHANNEL,
                                                                                                       IOU_SEQ_0_CONV_0_KERNEL_SIZE,
                                                                                                       IOU_SEQ_0_CONV_0_STRIDE,
                                                                                                       IOU_SEQ_0_CONV_0_IN_PADDING,
                                                                                                       "module.dense_head.heads_list.0.iou.0.0",
                                                                                                       "module.dense_head.heads_list.0.iou.0.1" );
auto iou_header_last = conv_with_bias(network,weightMap,*iou_header->getOutput(0),
                                                                                                       IOU_CONV_1_OUT_CHANNEL,
                                                                                                       IOU_CONV_1_KERNEL_SIZE,
                                                                                                       IOU_CONV_1_STRIDE,
                                                                                                       IOU_CONV_1_IN_PADDING,
                                                                                                       "module.dense_head.heads_list.0.iou.1");

//hm

auto hm_header  = convBnLELU(network,weightMap,*center_head_shared_conv->getOutput(0),
                                                                                                       HM_SEQ_0_CONV_0_OUT_CHANNEL,
                                                                                                       HM_SEQ_0_CONV_0_KERNEL_SIZE,
                                                                                                       HM_SEQ_0_CONV_0_STRIDE,
                                                                                                       HM_SEQ_0_CONV_0_IN_PADDING,
                                                                                                       "module.dense_head.heads_list.0.hm.0.0",
                                                                                                       "module.dense_head.heads_list.0.hm.0.1" );
auto  hm_header_last = conv_with_bias(network,weightMap,*hm_header->getOutput(0),
                                                                                                       HM_CONV_1_OUT_CHANNEL,
                                                                                                       HM_CONV_1_KERNEL_SIZE,
                                                                                                       HM_CONV_1_STRIDE,
                                                                                                       HM_CONV_1_IN_PADDING,
                                                                                                       "module.dense_head.heads_list.0.hm.1");


/*

            post_processing

*/

//  batch_hm

auto  hm_header_last_sigmoid= network->addActivation(*hm_header_last->getOutput(0), ActivationType::kSIGMOID );
hm_header_last_sigmoid->setAlpha(1e-8);

// batch_center

//batch_center_z

//batch_dim
auto dim_header_last_exp= network->addUnary(*dim_header_last->getOutput(0), UnaryOperation::kEXP );

//batch_rot_cos
    nvinfer1::Dims start_0{ 4, 0, 0, 0, 0 };
    nvinfer1::Dims size_0{ 4, 1, 1, GRID_SIZE_Y, GRID_SIZE_X};  
    nvinfer1::Dims stride_0{ 4, 1, 1, 1, 1 };

    auto batch_rot_cos_slice_0 = network->addSlice(*rot_header_last->getOutput(0),start_0,size_0,stride_0);

//batch_rot_sin
    nvinfer1::Dims start_1{ 4, 0, 1, 0, 0};
    nvinfer1::Dims size_1{ 4, 1, 1, GRID_SIZE_Y, GRID_SIZE_X};  
    nvinfer1::Dims stride_1{ 4, 1, 1, 1, 1 };

    auto batch_rot_sin_slice_1 = network->addSlice(*rot_header_last->getOutput(0),start_1,size_1,stride_1);

    //batch_iou    (pred_dict['iou']+1)*0.5


// topk

auto hm_header_last_sigmoid_reshape = network->addShuffle(*hm_header_last_sigmoid->getOutput(0));
auto hm_dim = (*hm_header_last_sigmoid->getOutput(0)).getDimensions();
int hm_dim_1 = hm_dim.d[1];
int hm_dim_2 = hm_dim.d[2];
int hm_dim_3 = hm_dim.d[3];
hm_dim.d[0] = 1;
hm_dim.d[1] = hm_dim_1;  // 10
hm_dim.d[2] = hm_dim_2*hm_dim_3;  // 468*468
hm_dim.nbDims = 3;
hm_header_last_sigmoid_reshape->setReshapeDimensions(hm_dim);  // 1,10,468*468

auto hm_topk  = network->addTopK(*hm_header_last_sigmoid_reshape->getOutput(0),TopKOperation::kMAX,HM_TOP_K,0x04);  // scores, index
// for     hm_top_inds  % (height*width)
// 1 get height*width
auto dimension_height_x_width = (*hm_topk->getOutput(0)).getDimensions();
int *height_x_width_ptr = reinterpret_cast<int*>(malloc(sizeof(int) * hm_dim_1*HM_TOP_K));
    for (int i = 0; i <  hm_dim_1*HM_TOP_K; i++) {
        height_x_width_ptr[i] = hm_dim_2*hm_dim_3; //468*468
    }
nvinfer1::Weights height_x_width_weights{nvinfer1::DataType::kINT32, height_x_width_ptr, 1*hm_dim_1*HM_TOP_K};
auto height_x_width_tensor = network->addConstant(dimension_height_x_width,height_x_width_weights);// weights ----> tensor
// 2    remainder = num - divisor * (num / divisor)
// auto floor = network->addElementWise(*input, *mod_val, ElementWiseOperation::kFLOOR_DIV);
// auto prod = network->addElementWise(*floor, *mod_val, ElementWiseOperation::kPROD);
// auto remainder = network->addElementWise(*input, *prod, ElementWiseOperation::kSUB);
auto hm_top_index_mod_floor_div_elementwise = network->addElementWise(*hm_topk->getOutput(1),*height_x_width_tensor->getOutput(0), nvinfer1::ElementWiseOperation::kFLOOR_DIV);
auto hm_top_index_mod_prod_elementwise = network->addElementWise(*hm_top_index_mod_floor_div_elementwise->getOutput(0),*height_x_width_tensor->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
auto hm_top_index_mod_sum_elementwise=network->addElementWise(*hm_topk->getOutput(1),*hm_top_index_mod_prod_elementwise->getOutput(0),nvinfer1::ElementWiseOperation::kSUB);

auto dimension_width = (*hm_topk->getOutput(0)).getDimensions();
int *width_ptr = reinterpret_cast<int*>(malloc(sizeof(int) * hm_dim_1*HM_TOP_K));
    for (int i = 0; i <  hm_dim_1*HM_TOP_K; i++) {
        width_ptr[i] = hm_dim_3; //468
    }
nvinfer1::Weights width_weights{nvinfer1::DataType::kINT32, width_ptr, 1*hm_dim_1*HM_TOP_K};
auto width_tensor = network->addConstant(dimension_width,width_weights);// weights ----> tensor

// auto topk_index_float = network->addCast(*hm_top_index_mod_sum_elementwise->getOutput(0),nvinfer1::DataType::kFLOAT);

auto topk_ys_floor_div_elementwise = network->addElementWise(*hm_top_index_mod_sum_elementwise->getOutput(0),*width_tensor->getOutput(0), nvinfer1::ElementWiseOperation::kFLOOR_DIV);

auto topk_xs_floor_div_elementwise = network->addElementWise(*hm_top_index_mod_sum_elementwise->getOutput(0),*width_tensor->getOutput(0), nvinfer1::ElementWiseOperation::kFLOOR_DIV);
auto topk_xs_mod_prod_elementwise = network->addElementWise(*topk_xs_floor_div_elementwise->getOutput(0),*width_tensor->getOutput(0), nvinfer1::ElementWiseOperation::kPROD);
auto topk_xs_mod_sum_elementwise=network->addElementWise(*hm_top_index_mod_sum_elementwise->getOutput(0),*topk_xs_mod_prod_elementwise->getOutput(0),nvinfer1::ElementWiseOperation::kSUB);

// 1 10 500----> 1 5000
auto dim_hm_topk_1 = (*hm_topk->getOutput(0)).getDimensions();
 dim_hm_topk_1.d[0] =  1;
 dim_hm_topk_1.d[1] = HM_TOP_K * hm_dim_1;
  dim_hm_topk_1.nbDims = 2;
auto hm_topk_1_input = network->addShuffle(*hm_topk->getOutput(0));
hm_topk_1_input->setReshapeDimensions(dim_hm_topk_1);

auto hm_topk_1  = network->addTopK(*hm_topk_1_input->getOutput(0),TopKOperation::kMAX,HM_TOP_K,0x02);  

// topk_ind // k
auto dimension_k = (*hm_topk_1->getOutput(0)).getDimensions();
int *k_ptr = reinterpret_cast<int*>(malloc(sizeof(int) * HM_TOP_K));
    for (int i = 0; i <  HM_TOP_K; i++) {
        k_ptr[i] = HM_TOP_K; //500
    }
nvinfer1::Weights k_weights{nvinfer1::DataType::kINT32, k_ptr, 1*HM_TOP_K};
auto k_tensor = network->addConstant(dimension_k,k_weights);// weights ----> tensor
auto topk_classes_floor_div_elementwise = network->addElementWise(*hm_topk_1->getOutput(1),*k_tensor->getOutput(0), nvinfer1::ElementWiseOperation::kFLOOR_DIV);

//gather feat
// topk_inds   ----> hm_top_index_mod_sum_elementwise->getOutput(0)     1   10  500   -> 1 5000
// topk_ind     ------>hm_topk_1->getOutput(1)    1  500
// topk_ys      ------->topk_ys_floor_div_elementwise->getOutput(0)   1   10  500   -> 1  5000
// topk_xs    ------->topk_xs_mod_sum_elementwise->getOutput(0)   1   10   500   ->  1 5000
// 1 10 500-> 1  5000
auto topk_inds_reshape =  network->addShuffle(*hm_top_index_mod_sum_elementwise->getOutput(0));
topk_inds_reshape->setReshapeDimensions(dim_hm_topk_1);
// 1 10 500-> 1  5000
auto topk_ys_reshape =  network->addShuffle(*topk_ys_floor_div_elementwise->getOutput(0));
topk_ys_reshape->setReshapeDimensions(dim_hm_topk_1);
// 1 10 500-> 1  5000
auto topk_xs_reshape =  network->addShuffle(*topk_xs_mod_sum_elementwise->getOutput(0));
topk_xs_reshape->setReshapeDimensions(dim_hm_topk_1);

auto topk_inds_gather = network->addGather(*topk_inds_reshape->getOutput(0),*hm_topk_1->getOutput(1),1);
auto topk_inds_gather_reshape = network->addShuffle(*topk_inds_gather->getOutput(0));
topk_inds_gather_reshape->setReshapeDimensions(dimension_k);


// dimension_k.d[0] = 1;
// dimension_k.d[1] = 1;
// dimension_k.d[2] = HM_TOP_K;
// dimension_k.d[3] = 1;
// dimension_k.nbDims = 4;
auto topk_ys_gather = network->addGather(*topk_ys_reshape->getOutput(0),*hm_topk_1->getOutput(1),1);
auto topk_ys_gather_reshape = network->addShuffle(*topk_ys_gather->getOutput(0));
topk_ys_gather_reshape->setReshapeDimensions(dimension_k);

auto topk_xs_gather = network->addGather(*topk_xs_reshape->getOutput(0),*hm_topk_1->getOutput(1),1);
auto topk_xs_gather_reshape = network->addShuffle(*topk_xs_gather->getOutput(0));
topk_xs_gather_reshape->setReshapeDimensions(dimension_k);   // 1   1  500 ->   1   500


// center   1  2  468  468   ----> 1 468 468 2 ----> 1 468*468 2
// center_z  1  1  468  468   ----> 1 468 468 1 ----> 1 468*468 1
//   rot_sin   1  1  468  468   ----> 1 468 468 1 ----> 1 468*468 1
// rot_cos 1  1  468  468   ----> 1 468 468 1 ----> 1 468*468 1
//dim 1  3  468  468   ----> 1 468 468 3 ----> 1 468*468 3

  auto center_transpose =  network->addShuffle(*center_header_last->getOutput(0));
  center_transpose->setFirstTranspose(Permutation{0,2,3,1});

  auto dim_center_transpose_reshape = (*center_transpose->getOutput(0)).getDimensions();
 dim_center_transpose_reshape.d[0] =  1;
 dim_center_transpose_reshape.d[1] = SPARSE_SHAPE_X*SPARSE_SHAPE_Y ;
 dim_center_transpose_reshape.d[2] = CENTER_CONV_1_OUT_CHANNEL;
  dim_center_transpose_reshape.nbDims = 3;
auto center_transpose_reshape = network->addShuffle(*center_transpose->getOutput(0));
center_transpose_reshape->setReshapeDimensions(dim_center_transpose_reshape);  //  1  468*468   2


  auto center_z_transpose =  network->addShuffle(*center_z_header_last->getOutput(0));
  center_z_transpose->setFirstTranspose(Permutation{0,2,3,1});

  auto dim_center_z_transpose_reshape = (*center_z_transpose->getOutput(0)).getDimensions();
 dim_center_z_transpose_reshape.d[0] =  1;
 dim_center_z_transpose_reshape.d[1] = SPARSE_SHAPE_X*SPARSE_SHAPE_Y  ;
 dim_center_z_transpose_reshape.d[2] = CENTER_Z_CONV_1_OUT_CHANNEL;
  dim_center_z_transpose_reshape.nbDims = 3;
auto center_z_transpose_reshape = network->addShuffle(*center_z_transpose->getOutput(0));
center_z_transpose_reshape->setReshapeDimensions(dim_center_z_transpose_reshape);   //1  468*468   1


  auto rot_cos_transpose =  network->addShuffle(*batch_rot_cos_slice_0->getOutput(0));
  rot_cos_transpose->setFirstTranspose(Permutation{0,2,3,1});
auto rot_cos_transpose_reshape = network->addShuffle(*rot_cos_transpose->getOutput(0));
rot_cos_transpose_reshape->setReshapeDimensions(dim_center_z_transpose_reshape); //  1  468*468   1

  auto rot_sin_transpose =  network->addShuffle(*batch_rot_sin_slice_1->getOutput(0));
  rot_sin_transpose->setFirstTranspose(Permutation{0,2,3,1});
auto rot_sin_transpose_reshape = network->addShuffle(*rot_sin_transpose->getOutput(0));
rot_sin_transpose_reshape->setReshapeDimensions(dim_center_z_transpose_reshape); //  1  468*468   1


  auto dim_transpose =  network->addShuffle(*dim_header_last_exp->getOutput(0));
  dim_transpose->setFirstTranspose(Permutation{0,2,3,1});

  auto dim_dim_transpose_reshape = (*dim_transpose->getOutput(0)).getDimensions();
 dim_dim_transpose_reshape.d[0] =  1;
 dim_dim_transpose_reshape.d[1] = SPARSE_SHAPE_X*SPARSE_SHAPE_Y  ;
 dim_dim_transpose_reshape.d[2] = DIM_CONV_1_OUT_CHANNEL;
  dim_dim_transpose_reshape.nbDims = 3;
auto dim_transpose_reshape = network->addShuffle(*dim_transpose->getOutput(0));
dim_transpose_reshape->setReshapeDimensions(dim_dim_transpose_reshape);   //1  468*468   3


// gather   
auto center_gather = network->addGather(*center_transpose_reshape->getOutput(0),*topk_inds_gather_reshape->getOutput(0),1);   // 1   1   500  2
auto center_z_gather = network->addGather(*center_z_transpose_reshape->getOutput(0),*topk_inds_gather_reshape->getOutput(0),1); //  1   1  500  1
auto rot_sin_gather = network->addGather(*rot_sin_transpose_reshape->getOutput(0),*topk_inds_gather_reshape->getOutput(0),1);   // 1  1  500  1
auto rot_cos_gather = network->addGather(*rot_cos_transpose_reshape->getOutput(0),*topk_inds_gather_reshape->getOutput(0),1);  // 1  1  500  1
auto dim_gather = network->addGather(*dim_transpose_reshape->getOutput(0),*topk_inds_gather_reshape->getOutput(0),1);   // 1 1  500  3

//angle = torch.atan2(rot_sin, rot_cos)
auto angle_div_elementwise = network->addElementWise(*rot_sin_gather->getOutput(0),*rot_cos_gather->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);
auto atan_angle =  network->addUnary(*angle_div_elementwise->getOutput(0), UnaryOperation::kATAN );

    // xs = xs.view(batch_size, K, 1) + center[:, :, 0:1]
    // ys = ys.view(batch_size, K, 1) + center[:, :, 1:2]
//xs = xs * feature_map_stride * voxel_size[0] + point_cloud_range[0]
// ys = ys * feature_map_stride * voxel_size[1] + point_cloud_range[1]

        // topk_scores    hm_topk_1->getOutput(0)  float    1*500
    // topk_classes   topk_classes_floor_div_elementwise->getOutput(0)  int   1*500
    // xs    topk_xs_gather_reshape->getOutput(0)  int   1*500
    // ys  topk_ys_gather_reshape->getOutput(0) int 1*500
    //center  center_gather->getOutput(0)   float   1  1  500  2
    //center_z   center_z_gather->getOutput(0) float 1  1  500  1
    //angle  atan_angle->getOutput(0)  float   1  1   500  1
    // dim  dim_gather->getOutput(0) float  1  1  500  3
auto filter_box_by_score_layer = add_filter_box_by_score_op(network,hm_topk_1->getOutput(0),topk_classes_floor_div_elementwise->getOutput(0),
                                                                                                                    topk_xs_gather_reshape->getOutput(0), topk_ys_gather_reshape->getOutput(0),
                                                                                                                    center_gather->getOutput(0),center_z_gather->getOutput(0),
                                                                                                                    atan_angle->getOutput(0),dim_gather->getOutput(0),HM_TOP_K,
                                                                                                                    X_MIN,X_MAX,Y_MIN,Y_MAX,Z_MIN,Z_MAX,
                                                                                                                    VOXEL_SIZE_X,VOXEL_SIZE_Y,VOXEL_SIZE_Z,
                                                                                                                    SCORE_THRESHOLD
                                                                                                                    );


    auto dim = filter_box_by_score_layer->getOutput(0)->getDimensions();
    std::cout << "output0 output shape: ";
    for (int i = 0; i < dim.nbDims; i++) {
        std::cout << dim.d[i] << " ";
    }
    std::cout << std::endl;

    // auto dim1 = get_value_by_index_layer_0_0->getOutput(1)->getDimensions();
    // std::cout << "output1 output shape: ";
    // for (int i = 0; i < dim1.nbDims; i++) {
    //     std::cout << dim1.d[i] << " ";
    // }
    // std::cout << std::endl;

    // auto dim2 = get_value_by_index_layer_0_0->getOutput(2)->getDimensions();
    // std::cout << "output2 output shape: ";
    // for (int i = 0; i < dim2.nbDims; i++) {
    //     std::cout << dim2.d[i] << " ";
    // }
    // std::cout << std::endl;

//    auto dim3 = window_partition_1->getOutput(3)->getDimensions();
//     std::cout << "output3 output shape: ";
//     for (int i = 0; i < dim3.nbDims; i++) {
//         std::cout << dim3.d[i] << " ";
//     }
//     std::cout << std::endl;

//     auto dim4 = voxelGenerator->getOutput(4)->getDimensions();
//     std::cout << "output4 output shape: ";
//     for (int i = 0; i < dim4.nbDims; i++) {
//         std::cout << dim4.d[i] << " ";
//     }
//     std::cout << std::endl;

    filter_box_by_score_layer->getOutput(0)->setName(OUTPUT_VOXELS);
    network->markOutput(*filter_box_by_score_layer->getOutput(0));

    // get_value_by_index_layer_0_0->getOutput(2)->setName(OUTPUT_COORS);
    // network->markOutput(*get_value_by_index_layer_0_0->getOutput(2));

    filter_box_by_score_layer->getOutput(1)->setName(OUTPUT_VOXEL_NUM); 
    network->markOutput(*filter_box_by_score_layer->getOutput(1));

    // Build engine
    config->setMaxWorkspaceSize(200 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();
    // pluginObj_voxelGenerator->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    // free params struct;
    // free(newSubmConv3dLayerpluginData);
    

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(rt_glogger); // rt_glogger
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config); // DataType::kFLOAT
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}


int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("se-ssd-spp.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 2 && std::string(argv[1]) == "-d") {
        std::ifstream file("se-ssd-spp.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./se-ssd-spp -s  // serialize model to plan file" << std::endl;
        std::cerr << "./se-ssd-spp -d// deserialize plan file and run inference" << std::endl;
        return -1;
    }
    std::cout << "detection start   " << std::endl;
    IRuntime* runtime = createInferRuntime(rt_glogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    context->setOptimizationProfile(0);

    // int feature_map_size_x = BACKBONE_FEATURE_MAP_SIZE_X;
    // int feature_map_size_y = BACKBONE_FEATURE_MAP_SIZE_Y;
    // int feature_map_size_z = BACKBONE_FEATURE_MAP_SIZE_Z;
    // int feature_map_channel = BACKBONE_FEATURE_MAP_CHANNEL;


    int line_num = 500;//MAX_WIN_NUM; 500
    int feature_map_channel = 9;//POSEMBED_LAYBERS_OUT_FEATURES;  9
    int voxel_feature_byte_size = 1 * line_num * feature_map_channel * sizeof(float);


    const ICudaEngine& work_engine = context->getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(work_engine.getNbBindings() == 4);
    void* buffers[4];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex1 = work_engine.getBindingIndex(INPUT_POINTS);
    const int inputIndex2 = work_engine.getBindingIndex(INPUT_POINTS_SIZE);
    const int outputIndex1 = work_engine.getBindingIndex(OUTPUT_VOXELS);
    // const int outputIndex2 = work_engine.getBindingIndex(OUTPUT_COORS);
    const int outputIndex3 = work_engine.getBindingIndex(OUTPUT_VOXEL_NUM);

    context->setBindingDimensions(inputIndex1, Dims3{1, MAX_POINTS_NUM,4});
    Dims dims1;
    dims1.d[0] = 1;
    dims1.nbDims = 1;
    context->setBindingDimensions(inputIndex2,dims1);

    // Create GPU buffers on device
    checkCudaErrors(cudaMalloc(&buffers[inputIndex1], 1 * MAX_POINTS_NUM * 4* sizeof(float)));
    checkCudaErrors(cudaMalloc(&buffers[inputIndex2], 1 * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&buffers[outputIndex1],voxel_feature_byte_size));
    // checkCudaErrors(cudaMalloc(&buffers[outputIndex2],voxel_feature_byte_size));
    checkCudaErrors(cudaMalloc(&buffers[outputIndex3],1 * sizeof(unsigned int)));

    // Create stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));


   
    float* voxel_feature = (float*)malloc(voxel_feature_byte_size);
    // unsigned int coors_byte_size = voxel_feature_byte_size;
    // float * coors = (float *)malloc(coors_byte_size);
    unsigned int voxel_num = 0;

    std::string Data_File = "../data/bin/";
    std::string save_root = "../data/outputs/";

    std::vector<Bndbox> nms_pred;
    nms_pred.reserve(100);
    std::vector<Bndbox> res_;

    for (int i = 0; i <10; i++) //7481   5
    {
        std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        std::string dataFile = Data_File;
        std::stringstream ss;

        ss<< i;

        int n_zero = 6;
        std::string old_string(ss.str());
        std::string new_path = std::string(n_zero - old_string.length(),'0') + old_string;
        dataFile += new_path;
        dataFile += ".bin";

        std::cout << "<<<<<<<<<<<" <<std::endl;
        std::cout << "load file: "<< dataFile <<std::endl;

        //load points cloud
        unsigned int length = 0;
        void *data = NULL;
        std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
        loadData(dataFile.data(), &data, &length);
        buffer.reset((char *)data);

        float* points = (float*)buffer.get();
        unsigned int points_size = length/sizeof(float)/4;

        std::cout << "first point:  " << points[0] << "," << points[1] << "," << points[2] << "," << points[3] << std::endl; 

        std::cout << "find points num: "<< points_size <<std::endl;
       

            
        // auto start = std::chrono::system_clock::now();
        const clock_t begin_time = clock();
        // auto st = system_clock::now();
        
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        // checkCudaErrors(cudaMemcpyAsync(buffers[inputIndex1], points, 1 * MAX_POINTS * 4* sizeof(float), cudaMemcpyHostToDevice, stream));
        // checkCudaErrors(cudaMemcpyAsync(buffers[inputIndex2], &points_size, 1 * sizeof(unsigned int),cudaMemcpyHostToDevice, stream));

        checkCudaErrors(cudaMemcpy(buffers[inputIndex1], points, 1 * MAX_POINTS_NUM * 4* sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(buffers[inputIndex2], &points_size, 1 * sizeof(unsigned int),cudaMemcpyHostToDevice));

        context->enqueueV2(buffers, stream, nullptr);
       

        checkCudaErrors(cudaMemcpy(voxel_feature, buffers[outputIndex1], 
                voxel_feature_byte_size, cudaMemcpyDeviceToHost));
        
        // checkCudaErrors(cudaMemcpy(coors, buffers[outputIndex2], 
        //        coors_byte_size, cudaMemcpyDeviceToHost));


        // int voxel_num = 0;
        checkCudaErrors(cudaMemcpy(&voxel_num, buffers[outputIndex3], 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        std::cout << "voxel_num: " << voxel_num << std::endl;


        // // array to vector
        //  std::vector<std::complex<double>> save_data(feature_map_channel*line_num);
        //  for(int i = 0;i < line_num*feature_map_channel;i++) 
        //     save_data[i] = std::complex<double>(voxel_feature[i]);
        
        // //save it to file
        //  cnpy::npy_save("trt_points.npy",&save_data[0],{1,line_num,feature_map_channel},"w");  

        cudaStreamSynchronize(stream);

        save_result(res_,voxel_feature,voxel_num);
        nms_cpu(res_,NMS_THRESH,nms_pred);
        
        float seconds = float(clock() - begin_time) / 1000;

        // // duration<double> diff = system_clock::now() - st;
        std::cout << "doinference cost time: " << seconds <<  "ms" << std::endl;
        // std::cout << "耗时:" << diff.count() << "s" << std::endl;

        //save to txt
        std::vector<std::string> strlist;
        std::string split_name("/");
        stringsplit(dataFile,split_name,strlist);
        std::string save_path = strlist.back();
        save_path = save_path.replace(save_path.find("b"),3,"txt");
        save_path  =  save_root + save_path;
        std::cout << save_path << std::endl;
        save_txt(nms_pred,save_path,seconds);
        std::cout << "endenenenenend" << std::endl;
        
        nms_pred.clear(); 
        res_.clear();
    }

    free(voxel_feature);
    // free(coors);
    // Release stream and buffers
    cudaStreamDestroy(stream);
    
    checkCudaErrors(cudaFree(buffers[inputIndex1]));
    checkCudaErrors(cudaFree(buffers[inputIndex2]));
    checkCudaErrors(cudaFree(buffers[outputIndex1]));
    // checkCudaErrors(cudaFree(buffers[outputIndex2]));
    checkCudaErrors(cudaFree(buffers[outputIndex3]));
    
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
