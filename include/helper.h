//
// Created by nrsl on 23-4-7.
//

#ifndef _HELPER_H_
#define _HELPER_H_

#include <vector>
#include <chrono>
#include<typeinfo>
// #include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <iomanip> //设置输出格式
#include <map>
#include <sstream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <dirent.h>
#include<string>
#include "params.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

const float ThresHold = 1e-8;

int loadData(const char *file, void **data, unsigned int *length)
{
  std::fstream dataFile(file, std::ifstream::in);

  if (!dataFile.is_open())
  {
	  std::cout << "Can't open files: "<< file<<std::endl;
	  return -1;
  }

  //get length of file:
  unsigned int len = 0;
  dataFile.seekg (0, dataFile.end);
  len = dataFile.tellg();
  dataFile.seekg (0, dataFile.beg);

  //allocate memory:
  char *buffer = new char[MAX_POINTS_NUM*4*4];
  if(len>MAX_POINTS_NUM*4*4)
  {
      std::cout << "num of points: " << len << ">" << MAX_POINTS_NUM*4*4 << std::endl;
      delete [] buffer;
      dataFile.close();
      exit(-1);
  }
  // init for buffer
  for(int i=0; i<MAX_POINTS_NUM*4*4; i++)
  {
      buffer[i] = 0;
  }
  
  if(buffer==NULL) {
	  std::cout << "Can't malloc buffer."<<std::endl;
    dataFile.close();
	  exit(-1);
  }

  //read data as a block:
  dataFile.read(buffer, len);
  dataFile.close();

  *data = (void*)buffer;
  *length = len;
  return 0;  
}

// string split(使用字符串分割)
void stringsplit(const std::string& str, const std::string& splits, std::vector<std::string>& res)
{
    if(str == "") return;
    string strs = str + splits;
    int pos = strs.find(splits);
    int step = splits.size();

    while(pos != strs.npos)
    {
        std::string temp = strs.substr(0,pos);
        res.push_back(temp);
        strs = strs.substr(pos+step,strs.size());
        pos = strs.find(splits);
    }
}


//code for calculating rotated NMS come from  https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars/blob/main/src/postprocess.cpp
struct Bndbox {
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float rt;
    int id;
    float score;
    Bndbox(){};
    Bndbox(float x_, float y_, float z_, float w_, float l_, float h_, float rt_, int id_, float score_)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), rt(rt_), id(id_), score(score_) {}
};


inline float cross(const float2 p1, const float2 p2, const float2 p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

inline int check_box2d(const Bndbox box, const float2 p) {
    const float MARGIN = 1e-2;
    float center_x = box.x;
    float center_y = box.y;
    float angle_cos = cos(-box.rt);
    float angle_sin = sin(-box.rt);
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box.w / 2 + MARGIN && fabs(rot_y) < box.l / 2 + MARGIN);
}

bool intersection(const float2 p1, const float2 p0, const float2 q1, const float2 q0, float2 &ans) {

    if (( std::min(p0.x, p1.x) <= std::max(q0.x, q1.x) &&
          std::min(q0.x, q1.x) <= std::max(p0.x, p1.x) &&
          std::min(p0.y, p1.y) <= std::max(q0.y, q1.y) &&
          std::min(q0.y, q1.y) <= std::max(p0.y, p1.y) ) == 0)
        return false;


    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0))
        return false;

    float s5 = cross(q1, p1, p0);
    if (fabs(s5 - s1) > ThresHold) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    } else {
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return true;
}

inline void rotate_around_center(const float2 &center, const float angle_cos, const float angle_sin, float2 &p) {
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p = float2 {new_x, new_y};
    return;
}

inline float box_overlap(const Bndbox &box_a, const Bndbox &box_b) {
    float a_angle = box_a.rt, b_angle = box_b.rt;
    float a_dx_half = box_a.w / 2, b_dx_half = box_b.w / 2, a_dy_half = box_a.l / 2, b_dy_half = box_b.l / 2;
    float a_x1 = box_a.x - a_dx_half, a_y1 = box_a.y - a_dy_half;
    float a_x2 = box_a.x + a_dx_half, a_y2 = box_a.y + a_dy_half;
    float b_x1 = box_b.x - b_dx_half, b_y1 = box_b.y - b_dy_half;
    float b_x2 = box_b.x + b_dx_half, b_y2 = box_b.y + b_dy_half;
    float2 box_a_corners[5];
    float2 box_b_corners[5];

    float2 center_a = float2 {box_a.x, box_a.y};
    float2 center_b = float2 {box_b.x, box_b.y};

    float2 cross_points[16];
    float2 poly_center =  {0, 0};
    int cnt = 0;
    bool flag = false;

    box_a_corners[0] = float2 {a_x1, a_y1};
    box_a_corners[1] = float2 {a_x2, a_y1};
    box_a_corners[2] = float2 {a_x2, a_y2};
    box_a_corners[3] = float2 {a_x1, a_y2};

    box_b_corners[0] = float2 {b_x1, b_y1};
    box_b_corners[1] = float2 {b_x2, b_y1};
    box_b_corners[2] = float2 {b_x2, b_y2};
    box_b_corners[3] = float2 {b_x1, b_y2};

    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++) {
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                                box_b_corners[j + 1], box_b_corners[j],
                                cross_points[cnt]);
            if (flag) {
                poly_center = {poly_center.x + cross_points[cnt].x, poly_center.y + cross_points[cnt].y};
                cnt++;
            }
        }
    }

    for (int k = 0; k < 4; k++) {
        if (check_box2d(box_a, box_b_corners[k])) {
            poly_center = {poly_center.x + box_b_corners[k].x, poly_center.y + box_b_corners[k].y};
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (check_box2d(box_b, box_a_corners[k])) {
            poly_center = {poly_center.x + box_a_corners[k].x, poly_center.y + box_a_corners[k].y};
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    float2 temp;
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (atan2(cross_points[i].y - poly_center.y, cross_points[i].x - poly_center.x) >
                atan2(cross_points[i+1].y - poly_center.y, cross_points[i+1].x - poly_center.x)
                ) {
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
        float2 a = {cross_points[k].x - cross_points[0].x,
                    cross_points[k].y - cross_points[0].y};
        float2 b = {cross_points[k + 1].x - cross_points[0].x,
                    cross_points[k + 1].y - cross_points[0].y};
        area += (a.x * b.y - a.y * b.x);
    }
    return fabs(area) / 2.0;
}

int nms_cpu(std::vector<Bndbox> bndboxes, const float nms_thresh, std::vector<Bndbox> &nms_pred)
{
    std::sort(bndboxes.begin(), bndboxes.end(),
              [](Bndbox boxes1, Bndbox boxes2) { return boxes1.score > boxes2.score; });
    std::vector<int> suppressed(bndboxes.size(), 0);
    for (size_t i = 0; i < bndboxes.size(); i++) {
        if (suppressed[i] == 1) {
            continue;
        }
        nms_pred.emplace_back(bndboxes[i]);
        for (size_t j = i + 1; j < bndboxes.size(); j++) {
            if (suppressed[j] == 1) {
                continue;
            }

            float sa = bndboxes[i].w * bndboxes[i].l;
            float sb = bndboxes[j].w * bndboxes[j].l;
            float s_overlap = box_overlap(bndboxes[i], bndboxes[j]);
            float iou = s_overlap / fmaxf(sa + sb - s_overlap, ThresHold);

            if (iou >= nms_thresh) {
                suppressed[j] = 1;
            }
        }
    }
    return 0;
}


// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, nvinfer1::Weights> loadWeights_new(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, nvinfer1::Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        std::string find_str = ".in_proj_";

        input >> name >> std::dec >> size;
        wt.type = nvinfer1::DataType::kFLOAT;

        auto idx=name.find(find_str); 
        if (idx == string::npos ) // not found
        {
             // Load blob
            uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
            
            wt.count = size;
            weightMap[name] = wt;
        }
        else  // found
        {
            int split_size = size / 3;
            string split_name = "";
            for(int i=0;i<3;i++)
            {
                // std::cout << i << std::endl;
                if(i==0)
                {
                    nvinfer1::Weights wt0{nvinfer1::DataType::kFLOAT, nullptr, 0};
                     wt0.type = nvinfer1::DataType::kFLOAT;
                    split_name = name + ".query";
                    // std::cout << split_name << std::endl;

                     // Load blob
                        uint32_t* val_q = reinterpret_cast<uint32_t*>(malloc(sizeof(val_q) * split_size));
                        for (uint32_t x = 0, y = split_size; x < y; ++x)
                        {
                            input >> std::hex >> val_q[x];
                        }
                        wt0.values = val_q;
                        
                        wt0.count = split_size;
                        weightMap[split_name] = wt0;

                }
                if(i==1)
                {
                     nvinfer1::Weights wt1{nvinfer1::DataType::kFLOAT, nullptr, 0};
                     wt1.type = nvinfer1::DataType::kFLOAT;

                      split_name = name + ".key";

                        // Load blob
                        uint32_t* val_k = reinterpret_cast<uint32_t*>(malloc(sizeof(val_k) * split_size));
                        for (uint32_t x = 0, y = split_size; x < y; ++x)
                        {
                            input >> std::hex >> val_k[x];
                        }
                        wt1.values = val_k;
                        
                        wt1.count = split_size;
                        weightMap[split_name] = wt1;

                }

                if(i==2)
                {
                     nvinfer1::Weights wt2{nvinfer1::DataType::kFLOAT, nullptr, 0};
                     wt2.type = nvinfer1::DataType::kFLOAT;

                      split_name = name + ".value";

                        // Load blob
                        uint32_t* val_v = reinterpret_cast<uint32_t*>(malloc(sizeof(val_v) * split_size));
                        for (uint32_t x = 0, y = split_size; x < y; ++x)
                        {
                            input >> std::hex >> val_v[x];
                        }
                        wt2.values = val_v;
                        
                        wt2.count = split_size;
                        weightMap[split_name] = wt2;

                }

            }
        }
           
    }

    return weightMap;
}

int save_txt(std::vector<Bndbox> &nms_pred, std::string save_path,float seconds)
{
    std::ofstream out_txt_file;
    out_txt_file.open(save_path.c_str(),ios::out | ios::trunc);
    // out_txt_file << fixed;
    out_txt_file << setiosflags(ios::fixed) << setprecision(6);
    out_txt_file << seconds << std::endl;
    float max_value = 0;
    for(int i=0;i < nms_pred.size(); i++)
    {
        // std::cout << "height_i: " << height_i << std::endl;
        
            
        out_txt_file << nms_pred[i].x << ",  ";
        out_txt_file << nms_pred[i].y << ",  ";
        out_txt_file << nms_pred[i].z << ",  ";
        out_txt_file << nms_pred[i].l << ",  ";
        out_txt_file << nms_pred[i].w << ",  ";
        out_txt_file << nms_pred[i].h << ",  ";
        out_txt_file << nms_pred[i].rt << ",  ";
        out_txt_file << nms_pred[i].id << ",  ";
        out_txt_file << nms_pred[i].score << std::endl;

    }
    // std::cout << "max_value: " << max_value << std::endl;
    out_txt_file.close();
    return 0;
}

void save_result(std::vector<Bndbox> &res_,float *output,int voxel_num)
{
    
    for (int i = 0; i < voxel_num; i++) {
    auto Bb = Bndbox(output[i * 9],
                    output[i * 9 + 1], output[i * 9 + 2], output[i * 9 + 4],
                    output[i * 9 + 3], output[i * 9 + 5], output[i * 9 + 6],
                    static_cast<int>(output[i * 9 + 7]),
                    output[i * 9 + 8]);
    res_.push_back(Bb);
  }
}

#endif 
