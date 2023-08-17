//
// Created by nrsl on 23-4-7.
//

#ifndef _PLUGIN_HELPER_H_
#define _PLUGIN_HELPER_H_

#include <vector>
#include <chrono>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <boost/filesystem.hpp>
using namespace nvinfer1;

IPluginV2Layer* add_voxel_generator(INetworkDefinition *network,ITensor * point_data, ITensor* point_size,int max_points_num,int max_points_num_voxel_filter,
                                    int max_pillars_num, int point_feature_num, int feature_num, int max_num_points_per_voxel, 
                                    float x_min,float x_max,float y_min,float y_max,float z_min, float z_max,
                                    float voxel_size_x,float voxel_size_y,float voxel_size_z,int grid_size_x, int grid_size_y, int grid_size_z)
{

    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    float *voxel_size = (float*)malloc(3*sizeof(float));
    float *point_cloud_range = (float*)malloc(6*sizeof(float));
    int *grid_size = (int*)malloc(3*sizeof(int));
    voxel_size[0] = voxel_size_x;
    voxel_size[1] = voxel_size_y;
    voxel_size[2] = voxel_size_z;

    point_cloud_range[0] = x_min;
    point_cloud_range[1] = y_min;
    point_cloud_range[2] = z_min;
    point_cloud_range[3] = x_max;
    point_cloud_range[4] = y_max;
    point_cloud_range[5] = z_max;

    grid_size[0] = grid_size_x;
    grid_size[1] = grid_size_y;
    grid_size[2] = grid_size_z;

    auto voxelGeneratorcreator = getPluginRegistry()->getPluginCreator("Points2FeaturesPlugin", "1");
    const PluginFieldCollection* voxelGeneratorpluginData = voxelGeneratorcreator->getFieldNames();

    const PluginField* fields = voxelGeneratorpluginData->fields;
    int nbFields = voxelGeneratorpluginData->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "max_points_num"))
        {
            
            new_pluginData_list.emplace_back(PluginField("max_points_num",  &(max_points_num), PluginFieldType::kINT32, 1));
          
            
        }
        else if (!strcmp(attr_name, "max_points_num_voxel_filter"))
        {
            
            new_pluginData_list.emplace_back(PluginField("max_points_num_voxel_filter",  &(max_points_num_voxel_filter), PluginFieldType::kINT32, 1));
          
            
        }
        else if (!strcmp(attr_name, "max_pillars_num"))
        {
           
            new_pluginData_list.emplace_back(PluginField("max_pillars_num",  &(max_pillars_num), PluginFieldType::kINT32, 1)); 
            
        }
        else if (!strcmp(attr_name, "point_feature_num"))
        {
           
            new_pluginData_list.emplace_back(PluginField("point_feature_num",  &(point_feature_num), PluginFieldType::kINT32, 1)); 
            
        }
          else if (!strcmp(attr_name, "feature_num"))
        {
           
            new_pluginData_list.emplace_back(PluginField("feature_num",  &(feature_num), PluginFieldType::kINT32, 1)); 
            
        }
        else if (!strcmp(attr_name, "max_num_points_per_voxel"))
        {
           
            new_pluginData_list.emplace_back(PluginField("max_num_points_per_voxel",  &(max_num_points_per_voxel), PluginFieldType::kINT32, 1)); 
            
        }
        else if (!strcmp(attr_name, "point_cloud_range"))
        {
           
            new_pluginData_list.emplace_back(PluginField("point_cloud_range",  point_cloud_range, PluginFieldType::kFLOAT32, 1));  
           
        }
        else if (!strcmp(attr_name, "voxel_size"))
        {
            
            new_pluginData_list.emplace_back(PluginField("voxel_size",  voxel_size, PluginFieldType::kFLOAT32, 1));
            
        }
        else if (!strcmp(attr_name, "grid_size"))
        {
            
            new_pluginData_list.emplace_back(PluginField("grid_size",  grid_size, PluginFieldType::kINT32, 1));
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    
    IPluginV2 *pluginObj_voxelGenerator = voxelGeneratorcreator->createPlugin("voxelGeneratorlayer", newPluginFieldCollection);
    ITensor* inputTensors_voxelgenerator[] = {point_data,point_size};
    auto voxelGenerator = network->addPluginV2(inputTensors_voxelgenerator, 2, *pluginObj_voxelGenerator);
    pluginObj_voxelGenerator->destroy();
    free(voxel_size);
    free(point_cloud_range);
    free(grid_size);
    return voxelGenerator;
}

IPluginV2Layer* add_torch_scatter_max(INetworkDefinition *network,
                ITensor * point_features_data, ITensor* point_index_in_voxel_data, ITensor * point_num_in_voxel_data, ITensor* voxel_num_data,
                                    int max_points_num,int max_pillars_num, int feature_num)
{
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    auto torch_scatter_max_op = getPluginRegistry()->getPluginCreator("TorchScatterMaxPlugin", "1");
    const PluginFieldCollection* torchScatterMaxData = torch_scatter_max_op->getFieldNames();

    const PluginField* fields = torchScatterMaxData->fields;
    int nbFields = torchScatterMaxData->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "max_points_num"))
        {
            
            new_pluginData_list.emplace_back(PluginField("max_points_num",  &(max_points_num), PluginFieldType::kINT32, 1));
          
            
        }
        else if (!strcmp(attr_name, "max_pillars_num"))
        {
           
            new_pluginData_list.emplace_back(PluginField("max_pillars_num",  &(max_pillars_num), PluginFieldType::kINT32, 1)); 
            
        }
          else if (!strcmp(attr_name, "feature_num"))
        {
           
            new_pluginData_list.emplace_back(PluginField("feature_num",  &(feature_num), PluginFieldType::kINT32, 1)); 
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    IPluginV2 *pluginObj_torchScatterMax = torch_scatter_max_op->createPlugin("torch_scatter_max_op", newPluginFieldCollection);
    ITensor* inputTensors_torchScatterMax[] = {point_features_data,point_index_in_voxel_data,point_num_in_voxel_data,voxel_num_data};
    auto torchScatterMax_op = network->addPluginV2(inputTensors_torchScatterMax, 4, *pluginObj_torchScatterMax);
    pluginObj_torchScatterMax->destroy();
    return torchScatterMax_op;
}
           
IPluginV2Layer* add_window_partition(INetworkDefinition *network,
                ITensor * coors_data, ITensor* valid_voxel_num_data, int max_win_num, int max_voxel_num_per_win, int sparse_shape_x,
                int sparse_shape_y, int sparse_shape_z, int win_shape_x, int win_shape_y, int win_shape_z, int shift_x, int shift_y, int shift_z)
{
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    auto  op = getPluginRegistry()->getPluginCreator("WindowPartitionPlugin", "1");
    const PluginFieldCollection* opData = op->getFieldNames();

    const PluginField* fields = opData->fields;
    int nbFields = opData->nbFields;

    int *sparse_shape = (int*)malloc(3*sizeof(int));
    int *win_shape =  (int*)malloc(3*sizeof(int));
    int *shift_list =  (int*)malloc(3*sizeof(int));
    sparse_shape[0] = sparse_shape_x;
    sparse_shape[1] = sparse_shape_y;
    sparse_shape[2] = sparse_shape_z;

    win_shape[0] = win_shape_x;
    win_shape[1] = win_shape_y;
    win_shape[2] = win_shape_z;

    shift_list[0] = shift_x;
    shift_list[1] = shift_y;
    shift_list[2] = shift_z;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "max_win_num"))
        {
            
            new_pluginData_list.emplace_back(PluginField("max_win_num",  &(max_win_num), PluginFieldType::kINT32, 1));
          
            
        }
        else if (!strcmp(attr_name, "max_voxel_num_per_win"))
        {
           
            new_pluginData_list.emplace_back(PluginField("max_voxel_num_per_win",  &(max_voxel_num_per_win), PluginFieldType::kINT32, 1)); 
            
        }
          else if (!strcmp(attr_name, "sparse_shape"))
        {
           
            new_pluginData_list.emplace_back(PluginField("sparse_shape",  sparse_shape, PluginFieldType::kINT32, 1)); 
            
        }
             else if (!strcmp(attr_name, "win_shape"))
        {
           
            new_pluginData_list.emplace_back(PluginField("win_shape",  win_shape, PluginFieldType::kINT32, 1)); 
            
        }
             else if (!strcmp(attr_name, "shift_list"))
        {
           
            new_pluginData_list.emplace_back(PluginField("shift_list", shift_list, PluginFieldType::kINT32, 1)); 
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    IPluginV2 *pluginObj = op->createPlugin("window_partition_op", newPluginFieldCollection);
    ITensor* inputTensors[] = {coors_data,valid_voxel_num_data};
    auto plugin_ = network->addPluginV2(inputTensors, 2, *pluginObj);
    pluginObj->destroy();
    free(sparse_shape);
    free(win_shape);
    free(shift_list);
    return plugin_;
}

IPluginV2Layer* add_get_set_op(INetworkDefinition *network,
                ITensor * global_index_data, ITensor* coors_in_win_data, ITensor * voxel_num_in_win_data, ITensor* win_num_data,  
                int max_win_num, int max_voxel_num_per_win, int voxel_num_set, int win_shape_x, int win_shape_y, int win_shape_z)
{
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    auto  op = getPluginRegistry()->getPluginCreator("GetSetPlugin", "1");
    const PluginFieldCollection* opData = op->getFieldNames();

    const PluginField* fields = opData->fields;
    int nbFields = opData->nbFields;

    int *win_shape =  (int*)malloc(3*sizeof(int));

    win_shape[0] = win_shape_x;
    win_shape[1] = win_shape_y;
    win_shape[2] = win_shape_z;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "max_win_num"))
        {
            
            new_pluginData_list.emplace_back(PluginField("max_win_num",  &(max_win_num), PluginFieldType::kINT32, 1));
          
            
        }
        else if (!strcmp(attr_name, "max_voxel_num_per_win"))
        {
           
            new_pluginData_list.emplace_back(PluginField("max_voxel_num_per_win",  &(max_voxel_num_per_win), PluginFieldType::kINT32, 1)); 
            
        }
          else if (!strcmp(attr_name, "voxel_num_set"))
        {
           
            new_pluginData_list.emplace_back(PluginField("voxel_num_set",  &(voxel_num_set), PluginFieldType::kINT32, 1)); 
            
        }
             else if (!strcmp(attr_name, "win_shape"))
        {
           
            new_pluginData_list.emplace_back(PluginField("win_shape",  win_shape, PluginFieldType::kINT32, 1)); 
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    IPluginV2 *pluginObj = op->createPlugin("get_set_op", newPluginFieldCollection);
    ITensor* inputTensors[] = {global_index_data,coors_in_win_data,voxel_num_in_win_data,win_num_data};
    auto plugin_ = network->addPluginV2(inputTensors, 4, *pluginObj);
    pluginObj->destroy();
    free(win_shape);
    return plugin_;
}


IPluginV2Layer* add_get_value_by_index_op(INetworkDefinition *network,
                ITensor * voxel_features, ITensor* pose_features, ITensor * voxel_inds, ITensor* valid_set_num,  
                int max_win_num, int voxel_num_set, int channel_num, int axis_id)
{
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    auto  op = getPluginRegistry()->getPluginCreator("GetValueByIndexPlugin", "1");
    const PluginFieldCollection* opData = op->getFieldNames();

    const PluginField* fields = opData->fields;
    int nbFields = opData->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "max_win_num"))
        {
            
            new_pluginData_list.emplace_back(PluginField("max_win_num",  &(max_win_num), PluginFieldType::kINT32, 1));
          
            
        }
          else if (!strcmp(attr_name, "voxel_num_set"))
        {
           
            new_pluginData_list.emplace_back(PluginField("voxel_num_set",  &(voxel_num_set), PluginFieldType::kINT32, 1)); 
            
        }
             else if (!strcmp(attr_name, "channel_num"))
        {
           
            new_pluginData_list.emplace_back(PluginField("channel_num",  &(channel_num), PluginFieldType::kINT32, 1)); 
            
        }
        else if (!strcmp(attr_name, "axis_id"))
        {
           
            new_pluginData_list.emplace_back(PluginField("axis_id",  &(axis_id), PluginFieldType::kINT32, 1)); 
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    IPluginV2 *pluginObj = op->createPlugin("get_value_by_index_op", newPluginFieldCollection);
    ITensor* inputTensors[] = {voxel_features,pose_features,voxel_inds,valid_set_num};
    auto plugin_ = network->addPluginV2(inputTensors, 4, *pluginObj);
    pluginObj->destroy();
    return plugin_;
}

IPluginV2Layer* add_map_2_bev_op(INetworkDefinition *network,
                ITensor * voxel_features, ITensor* coors, ITensor* valid_voxel_num,  
                int max_pillars_num, int channel_num, int grid_size_x, int grid_size_y)
{
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    auto  op = getPluginRegistry()->getPluginCreator("Map2BevPlugin", "1");
    const PluginFieldCollection* opData = op->getFieldNames();

    const PluginField* fields = opData->fields;
    int nbFields = opData->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        // std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "max_pillars_num"))
        {
            
            new_pluginData_list.emplace_back(PluginField("max_pillars_num",  &(max_pillars_num), PluginFieldType::kINT32, 1));
          
            
        }
          else if (!strcmp(attr_name, "channel_num"))
        {
           
            new_pluginData_list.emplace_back(PluginField("channel_num",  &(channel_num), PluginFieldType::kINT32, 1)); 
            
        }
             else if (!strcmp(attr_name, "grid_size_x"))
        {
           
            new_pluginData_list.emplace_back(PluginField("grid_size_x",  &(grid_size_x), PluginFieldType::kINT32, 1)); 
            
        }
        else if (!strcmp(attr_name, "grid_size_y"))
        {
           
            new_pluginData_list.emplace_back(PluginField("grid_size_y",  &(grid_size_y), PluginFieldType::kINT32, 1)); 
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    IPluginV2 *pluginObj = op->createPlugin("map_2_bev_op", newPluginFieldCollection);
    ITensor* inputTensors[] = {voxel_features,coors,valid_voxel_num};
    auto plugin_ = network->addPluginV2(inputTensors, 3, *pluginObj);
    pluginObj->destroy();
    return plugin_;
}


IPluginV2Layer* add_map_set_feature2voxel_op(INetworkDefinition *network,
                ITensor * voxel_features, ITensor * voxel_inds, ITensor* valid_set_num,  
                int max_win_num, int voxel_num_set, int channel_num, int axis_id, int max_pillars_num)
{
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    auto  op = getPluginRegistry()->getPluginCreator("MapSetFeature2VoxelPlugin", "1");
    const PluginFieldCollection* opData = op->getFieldNames();

    const PluginField* fields = opData->fields;
    int nbFields = opData->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "max_win_num"))
        {
            
            new_pluginData_list.emplace_back(PluginField("max_win_num",  &(max_win_num), PluginFieldType::kINT32, 1));
          
            
        }
          else if (!strcmp(attr_name, "voxel_num_set"))
        {
           
            new_pluginData_list.emplace_back(PluginField("voxel_num_set",  &(voxel_num_set), PluginFieldType::kINT32, 1)); 
            
        }
             else if (!strcmp(attr_name, "channel_num"))
        {
           
            new_pluginData_list.emplace_back(PluginField("channel_num",  &(channel_num), PluginFieldType::kINT32, 1)); 
            
        }
        else if (!strcmp(attr_name, "axis_id"))
        {
           
            new_pluginData_list.emplace_back(PluginField("axis_id",  &(axis_id), PluginFieldType::kINT32, 1)); 
            
        }
          else if (!strcmp(attr_name, "max_pillars_num"))
        {
           
            new_pluginData_list.emplace_back(PluginField("max_pillars_num",  &(max_pillars_num), PluginFieldType::kINT32, 1)); 
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    IPluginV2 *pluginObj = op->createPlugin("map_set_feature2voxel_op", newPluginFieldCollection);
    ITensor* inputTensors[] = {voxel_features,voxel_inds,valid_set_num};
    auto plugin_ = network->addPluginV2(inputTensors, 3, *pluginObj);
    pluginObj->destroy();
    return plugin_;
}


IPluginV2Layer* add_layer_norm_op(INetworkDefinition *network,
                ITensor * voxel_features,ITensor* valid_voxel_num,  nvinfer1::Weights const& weights,nvinfer1::Weights const& bias,
                int max_pillars_num, int channel_num, int weights_size, float eps)
{
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    auto  op = getPluginRegistry()->getPluginCreator("LayerNormPlugin", "1");
    const PluginFieldCollection* opData = op->getFieldNames();

    const PluginField* fields = opData->fields;
    int nbFields = opData->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "max_pillars_num"))
        {
            
            new_pluginData_list.emplace_back(PluginField("max_pillars_num",  &(max_pillars_num), PluginFieldType::kINT32, 1));
          
            
        }
          else if (!strcmp(attr_name, "channel_num"))
        {
           
            new_pluginData_list.emplace_back(PluginField("channel_num",  &(channel_num), PluginFieldType::kINT32, 1)); 
            
        }
             else if (!strcmp(attr_name, "weights_size"))
        {
           
            new_pluginData_list.emplace_back(PluginField("weights_size",  &(weights_size), PluginFieldType::kINT32, 1)); 
            
        }
        else if (!strcmp(attr_name, "eps"))
        {
           
            new_pluginData_list.emplace_back(PluginField("eps",  &(eps), PluginFieldType::kFLOAT32, 1)); 
            
        }
             else if (!strcmp(attr_name, "weights"))
        {
            
            new_pluginData_list.emplace_back(PluginField("weights", weights.values, PluginFieldType::kFLOAT32, 1));  
          
        }
           else if (!strcmp(attr_name, "bias"))
        {
            
            new_pluginData_list.emplace_back(PluginField("bias", bias.values, PluginFieldType::kFLOAT32, 1));  
          
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    IPluginV2 *pluginObj = op->createPlugin("layer_norm_op", newPluginFieldCollection);
    ITensor* inputTensors[] = {voxel_features,valid_voxel_num};
    auto plugin_ = network->addPluginV2(inputTensors, 2, *pluginObj);
    pluginObj->destroy();
    return plugin_;
}


IPluginV2Layer* add_gelu_op(INetworkDefinition *network,
                ITensor * voxel_features,ITensor* valid_voxel_num,
                int max_pillars_num, int channel_num)
{
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    auto  op = getPluginRegistry()->getPluginCreator("GeluPlugin", "1");
    const PluginFieldCollection* opData = op->getFieldNames();

    const PluginField* fields = opData->fields;
    int nbFields = opData->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "max_pillars_num"))
        {
            
            new_pluginData_list.emplace_back(PluginField("max_pillars_num",  &(max_pillars_num), PluginFieldType::kINT32, 1));
          
            
        }
          else if (!strcmp(attr_name, "channel_num"))
        {
           
            new_pluginData_list.emplace_back(PluginField("channel_num",  &(channel_num), PluginFieldType::kINT32, 1)); 
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    IPluginV2 *pluginObj = op->createPlugin("gelu_op", newPluginFieldCollection);
    ITensor* inputTensors[] = {voxel_features,valid_voxel_num};
    auto plugin_ = network->addPluginV2(inputTensors, 2, *pluginObj);
    pluginObj->destroy();
    return plugin_;
}
 // topk_scores    hm_topk_1->getOutput(0)  float    1*500
    // topk_classes   topk_classes_floor_div_elementwise->getOutput(0)  int   1*500
    // xs    topk_xs_gather_reshape->getOutput(0)  int   1*500
    // ys  topk_ys_gather_reshape->getOutput(0) int 1*500
    //center  center_gather->getOutput(0)   float   1  1  500  2
    //center_z   center_z_gather->getOutput(0) float 1  1  500  1
    //angle  atan_angle->getOutput(0)  float   1  1   500  1
    // dim  dim_gather->getOutput(0) float  1  1  500  3
IPluginV2Layer* add_filter_box_by_score_op(INetworkDefinition *network,
                ITensor * topk_scores,ITensor* topk_classes,ITensor * xs,ITensor* ys,
                ITensor * center,ITensor* center_z,ITensor * angle,ITensor* dim,
                  int max_top_k, float min_x_range,float max_x_range,float min_y_range,float max_y_range,float min_z_range,float max_z_range,
                                                            float voxel_x_size,float voxel_y_size,float voxel_z_size,float score_threshold)
{
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    auto  op = getPluginRegistry()->getPluginCreator("FilterBoxByScorePlugin", "1");
    const PluginFieldCollection* opData = op->getFieldNames();

    const PluginField* fields = opData->fields;
    int nbFields = opData->nbFields;

    float *point_cloud_range = (float*)malloc(6*sizeof(float));
    float *voxel_size = (float*)malloc(3*sizeof(float));

    point_cloud_range[0] = min_x_range;
    point_cloud_range[1] = max_x_range;
    point_cloud_range[2] = min_y_range;
    point_cloud_range[3] = max_y_range;
    point_cloud_range[4] = min_z_range;
    point_cloud_range[5] = max_z_range;
    
    voxel_size[0] = voxel_x_size;
    voxel_size[1] = voxel_y_size;
    voxel_size[2] = voxel_z_size;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "max_top_k"))
        {
            
            new_pluginData_list.emplace_back(PluginField("max_top_k",  &(max_top_k), PluginFieldType::kINT32, 1));
          
            
        }
          else if (!strcmp(attr_name, "point_cloud_range"))
        {
           
            new_pluginData_list.emplace_back(PluginField("point_cloud_range",  point_cloud_range, PluginFieldType::kFLOAT32, 1)); 
            
        }
        else if (!strcmp(attr_name, "voxel_size"))
        {
           
            new_pluginData_list.emplace_back(PluginField("voxel_size",  voxel_size, PluginFieldType::kFLOAT32, 1)); 
            
        }
        else if (!strcmp(attr_name, "score_threshold"))
        {
           
            new_pluginData_list.emplace_back(PluginField("score_threshold",  &(score_threshold), PluginFieldType::kFLOAT32, 1)); 
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    IPluginV2 *pluginObj = op->createPlugin("filter_box_by_score_op", newPluginFieldCollection);
    ITensor* inputTensors[] = {topk_scores,topk_classes,xs,ys,center,center_z,angle,dim};
    auto plugin_ = network->addPluginV2(inputTensors, 8, *pluginObj);
    pluginObj->destroy();
    free(point_cloud_range);
    free(voxel_size);
    return plugin_;
}



#endif //POINT_DETECTION_UTILS_H
