/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-06-02T02:47:26-0300
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[38]) {
   { .size_bytes = 12289, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 64, 64, 3}}, .scale = {1, (const float[1]){0.003921568859368563}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "entrada_mapa_output" },
   { .size_bytes = 32768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 64, 64, 8}}, .scale = {1, (const float[1]){0.011654974892735481}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_19_1_Relu0_output" },
   { .size_bytes = 32768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 64, 64, 8}}, .scale = {1, (const float[1]){0.02083568647503853}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_20_1_Relu0_output" },
   { .size_bytes = 8192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 32, 32, 8}}, .scale = {1, (const float[1]){0.02083568647503853}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 32, 32, 16}}, .scale = {1, (const float[1]){0.02759747952222824}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_21_1_Relu0_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 32, 32, 16}}, .scale = {1, (const float[1]){0.04770101606845856}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_22_1_Relu0_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 16, 16, 16}}, .scale = {1, (const float[1]){0.04770101606845856}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_output" },
   { .size_bytes = 8192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 16, 16, 32}}, .scale = {1, (const float[1]){0.0679277554154396}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_23_1_Relu0_output" },
   { .size_bytes = 8192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 16, 16, 32}}, .scale = {1, (const float[1]){0.13300126791000366}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_24_1_Relu0_output" },
   { .size_bytes = 2048, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 8, 8, 32}}, .scale = {1, (const float[1]){0.13300126791000366}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 8, 8, 64}}, .scale = {1, (const float[1]){0.19942639768123627}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_25_1_Relu0_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 8, 8, 64}}, .scale = {1, (const float[1]){0.2809629738330841}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_26_1_Relu0_output" },
   { .size_bytes = 1024, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 4, 4, 64}}, .scale = {1, (const float[1]){0.2809629738330841}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_output" },
   { .size_bytes = 2048, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 4, 4, 128}}, .scale = {1, (const float[1]){0.4048124849796295}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_27_1_Relu0_output" },
   { .size_bytes = 2048, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 4, 4, 128}}, .scale = {1, (const float[1]){0.8235414028167725}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_28_1_Relu0_output" },
   { .size_bytes = 6272, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 7, 7, 128}}, .scale = {1, (const float[1]){0.8235414028167725}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 8, 8, 64}}, .scale = {1, (const float[1]){0.6743285059928894}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_transpose_4_1_Relu0_output" },
   { .size_bytes = 8192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 8, 8, 128}}, .scale = {1, (const float[1]){0.6743285059928894}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_concatenate_4_1_concat0_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 8, 8, 64}}, .scale = {1, (const float[1]){0.7520222067832947}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_29_1_Relu0_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 8, 8, 64}}, .scale = {1, (const float[1]){0.5698937177658081}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_30_1_Relu0_output" },
   { .size_bytes = 14400, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 15, 15, 64}}, .scale = {1, (const float[1]){0.5698937177658081}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output" },
   { .size_bytes = 8192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 16, 16, 32}}, .scale = {1, (const float[1]){0.6025909185409546}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_transpose_5_1_Relu0_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 16, 16, 64}}, .scale = {1, (const float[1]){0.6025909185409546}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_concatenate_5_1_concat0_output" },
   { .size_bytes = 8192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 16, 16, 32}}, .scale = {1, (const float[1]){0.5974933505058289}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_31_1_Relu0_output" },
   { .size_bytes = 8192, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 16, 16, 32}}, .scale = {1, (const float[1]){0.9845296144485474}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_32_1_Relu0_output" },
   { .size_bytes = 30752, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 31, 31, 32}}, .scale = {1, (const float[1]){0.9845296144485474}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 32, 32, 16}}, .scale = {1, (const float[1]){0.49105551838874817}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_transpose_6_1_Relu0_output" },
   { .size_bytes = 32768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 32, 32, 32}}, .scale = {1, (const float[1]){0.49105551838874817}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_concatenate_6_1_concat0_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 32, 32, 16}}, .scale = {1, (const float[1]){0.5005602240562439}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_33_1_Relu0_output" },
   { .size_bytes = 16384, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 32, 32, 16}}, .scale = {1, (const float[1]){0.5581063628196716}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_34_1_Relu0_output" },
   { .size_bytes = 63504, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 63, 63, 16}}, .scale = {1, (const float[1]){0.5581063628196716}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output" },
   { .size_bytes = 32768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 64, 64, 8}}, .scale = {1, (const float[1]){0.32563793659210205}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_transpose_7_1_Relu0_output" },
   { .size_bytes = 65536, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 64, 64, 16}}, .scale = {1, (const float[1]){0.32563793659210205}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_concatenate_7_1_concat0_output" },
   { .size_bytes = 32768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 64, 64, 8}}, .scale = {1, (const float[1]){0.2518783509731293}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_35_1_Relu0_output" },
   { .size_bytes = 32768, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 64, 64, 8}}, .scale = {1, (const float[1]){0.2936670780181885}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_36_1_Relu0_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 64, 64, 1}}, .scale = {1, (const float[1]){0.6340664625167847}}, .zeropoint = {1, (const int16_t[1]){59}}, .name = "Conv__2060_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 64, 64, 1}}, .scale = {1, (const float[1]){0.00392156932502985}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "Student_Unet_1_conv2d_37_1_Sigmoid0_output" },
   { .size_bytes = 4096, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_S8, .shape = {4, (const int32_t[4]){1, 1, 64, 64}}, .scale = {1, (const float[1]){0.00392156932502985}}, .zeropoint = {1, (const int16_t[1]){-128}}, .name = "output_QuantizeLinear_Input_to_chlast_output" }
  },
  .nodes = (const stai_node_details[37]){
    {.id = 50, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* Student_Unet_1_conv2d_19_1_Relu0 */
    {.id = 53, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* Student_Unet_1_conv2d_20_1_Relu0 */
    {.id = 56, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* Student_Unet_1_max_pooling2d_4_1_MaxPool2d0 */
    {.id = 59, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* Student_Unet_1_conv2d_21_1_Relu0 */
    {.id = 62, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* Student_Unet_1_conv2d_22_1_Relu0 */
    {.id = 65, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* Student_Unet_1_max_pooling2d_5_1_MaxPool2d0 */
    {.id = 68, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* Student_Unet_1_conv2d_23_1_Relu0 */
    {.id = 71, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* Student_Unet_1_conv2d_24_1_Relu0 */
    {.id = 74, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* Student_Unet_1_max_pooling2d_6_1_MaxPool2d0 */
    {.id = 77, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* Student_Unet_1_conv2d_25_1_Relu0 */
    {.id = 80, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* Student_Unet_1_conv2d_26_1_Relu0 */
    {.id = 83, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* Student_Unet_1_max_pooling2d_7_1_MaxPool2d0 */
    {.id = 86, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){12}}, .output_tensors = {1, (const int32_t[1]){13}} }, /* Student_Unet_1_conv2d_27_1_Relu0 */
    {.id = 89, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){13}}, .output_tensors = {1, (const int32_t[1]){14}} }, /* Student_Unet_1_conv2d_28_1_Relu0 */
    {.id = 92, .type = AI_LAYER_UPSAMPLE_TYPE, .input_tensors = {1, (const int32_t[1]){14}}, .output_tensors = {1, (const int32_t[1]){15}} }, /* Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample */
    {.id = 92, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){15}}, .output_tensors = {1, (const int32_t[1]){16}} }, /* Student_Unet_1_conv2d_transpose_4_1_Relu0 */
    {.id = 95, .type = AI_LAYER_CONCAT_TYPE, .input_tensors = {2, (const int32_t[2]){16, 11}}, .output_tensors = {1, (const int32_t[1]){17}} }, /* Student_Unet_1_concatenate_4_1_concat0 */
    {.id = 98, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){17}}, .output_tensors = {1, (const int32_t[1]){18}} }, /* Student_Unet_1_conv2d_29_1_Relu0 */
    {.id = 101, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){18}}, .output_tensors = {1, (const int32_t[1]){19}} }, /* Student_Unet_1_conv2d_30_1_Relu0 */
    {.id = 104, .type = AI_LAYER_UPSAMPLE_TYPE, .input_tensors = {1, (const int32_t[1]){19}}, .output_tensors = {1, (const int32_t[1]){20}} }, /* Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample */
    {.id = 104, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){20}}, .output_tensors = {1, (const int32_t[1]){21}} }, /* Student_Unet_1_conv2d_transpose_5_1_Relu0 */
    {.id = 107, .type = AI_LAYER_CONCAT_TYPE, .input_tensors = {2, (const int32_t[2]){21, 8}}, .output_tensors = {1, (const int32_t[1]){22}} }, /* Student_Unet_1_concatenate_5_1_concat0 */
    {.id = 110, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){22}}, .output_tensors = {1, (const int32_t[1]){23}} }, /* Student_Unet_1_conv2d_31_1_Relu0 */
    {.id = 113, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){23}}, .output_tensors = {1, (const int32_t[1]){24}} }, /* Student_Unet_1_conv2d_32_1_Relu0 */
    {.id = 116, .type = AI_LAYER_UPSAMPLE_TYPE, .input_tensors = {1, (const int32_t[1]){24}}, .output_tensors = {1, (const int32_t[1]){25}} }, /* Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample */
    {.id = 116, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){25}}, .output_tensors = {1, (const int32_t[1]){26}} }, /* Student_Unet_1_conv2d_transpose_6_1_Relu0 */
    {.id = 119, .type = AI_LAYER_CONCAT_TYPE, .input_tensors = {2, (const int32_t[2]){26, 5}}, .output_tensors = {1, (const int32_t[1]){27}} }, /* Student_Unet_1_concatenate_6_1_concat0 */
    {.id = 122, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){27}}, .output_tensors = {1, (const int32_t[1]){28}} }, /* Student_Unet_1_conv2d_33_1_Relu0 */
    {.id = 125, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){28}}, .output_tensors = {1, (const int32_t[1]){29}} }, /* Student_Unet_1_conv2d_34_1_Relu0 */
    {.id = 128, .type = AI_LAYER_UPSAMPLE_TYPE, .input_tensors = {1, (const int32_t[1]){29}}, .output_tensors = {1, (const int32_t[1]){30}} }, /* Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample */
    {.id = 128, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){30}}, .output_tensors = {1, (const int32_t[1]){31}} }, /* Student_Unet_1_conv2d_transpose_7_1_Relu0 */
    {.id = 131, .type = AI_LAYER_CONCAT_TYPE, .input_tensors = {2, (const int32_t[2]){31, 2}}, .output_tensors = {1, (const int32_t[1]){32}} }, /* Student_Unet_1_concatenate_7_1_concat0 */
    {.id = 134, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){32}}, .output_tensors = {1, (const int32_t[1]){33}} }, /* Student_Unet_1_conv2d_35_1_Relu0 */
    {.id = 137, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){33}}, .output_tensors = {1, (const int32_t[1]){34}} }, /* Student_Unet_1_conv2d_36_1_Relu0 */
    {.id = 140, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){34}}, .output_tensors = {1, (const int32_t[1]){35}} }, /* Conv__2060 */
    {.id = 143, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){35}}, .output_tensors = {1, (const int32_t[1]){36}} }, /* Student_Unet_1_conv2d_37_1_Sigmoid0 */
    {.id = 146, .type = AI_LAYER_TRANSPOSE_TYPE, .input_tensors = {1, (const int32_t[1]){36}}, .output_tensors = {1, (const int32_t[1]){37}} } /* output_QuantizeLinear_Input_to_chlast */
  },
  .n_nodes = 37
};
#endif

