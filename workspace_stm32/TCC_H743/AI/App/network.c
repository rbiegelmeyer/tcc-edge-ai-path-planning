/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-06-02T02:47:26-0300
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
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

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "network.h"
#include "network_details.h"
#include "network_data.h"
#include "stai_events.h"

#include "ai_lite_inspect.h"

#include "lite_operators.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_NETWORK_EVENT_NODE_START_CB
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_NETWORK_EVENT_NODE_STOP_CB
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_NETWORK_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_NETWORK_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_NETWORK_MODEL_SIGNATURE     "0xd6bfd17ec1303af82d7dc9bf6d0d74e6"
#define _STAI_NETWORK_DATETIME            "2026-06-02T02:47:26-0300"
#define _STAI_NETWORK_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_NETWORK_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_network_activations_1     (NULL)
#define g_network_activations_2     (NULL)




#if defined(HAVE_NETWORK_INFO)
/*****************************************************************************/
static const stai_network_info g_network_info = {
  .model_signature = _STAI_NETWORK_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_NETWORK_COMPILE_DATETIME,
  .c_model_name = STAI_NETWORK_MODEL_NAME,
  .c_model_datetime = _STAI_NETWORK_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(12, 0, 0),
  .tool_version = STAI_INIT_VERSION(4, 0, 0),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_NETWORK_MACC_NUM,
  .n_nodes = STAI_NETWORK_NODES_NUM,
  .flags = STAI_NETWORK_FLAGS,
  .n_inputs = STAI_NETWORK_IN_NUM,
  .n_outputs = STAI_NETWORK_OUT_NUM,
  .n_activations = STAI_NETWORK_ACTIVATIONS_NUM,
  .n_weights = STAI_NETWORK_WEIGHTS_NUM,
  .n_states = STAI_NETWORK_STATES_NUM,
  .inputs = (stai_tensor[STAI_NETWORK_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_IN_1_NAME,
      STAI_NETWORK_IN_1_FLAGS,
      STAI_NETWORK_IN_1_FORMAT,
      STAI_NETWORK_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 64, 64, 3),
      STAI_DECLARE_ARRAY(float, 1, 0.003921568859368563f),
      STAI_DECLARE_ARRAY(int16_t, 1, -128)),
    },
    .outputs = (stai_tensor[STAI_NETWORK_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_NETWORK_OUT_1_NAME,
      STAI_NETWORK_OUT_1_FLAGS,
      STAI_NETWORK_OUT_1_FORMAT,
      STAI_NETWORK_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 64, 1, 64),
      STAI_DECLARE_ARRAY(float, 1, 0.00392156932502985f),
      STAI_DECLARE_ARRAY(int16_t, 1, -128)),
    },
  .activations = (stai_tensor[STAI_NETWORK_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 20188),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_ACTIVATION_2_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_ACTIVATION_2_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 131072),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_NETWORK_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_NETWORK_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_NETWORK_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 488388),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_network_context* _net_ctx = (_stai_network_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_network_check(_stai_network_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_NETWORK_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_NETWORK_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_NETWORK_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_network_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_network_context* net_ctx = (_stai_network_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_network_context))

  _STAI_SET_ERROR(net_ctx, STAI_NETWORK_CONTEXT_SIZE != sizeof(_stai_network_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_network_context _network_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_NETWORK_MODEL_SIGNATURE,
      ._flags = STAI_NETWORK_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_network_activations_1,(stai_ptr)g_network_activations_2
      },
      ._weights = {
      (stai_ptr)g_network_weights_array
      },
      ._inputs = {
    NULL},
      ._outputs = {
    NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _network_context;

    _stai_network_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_NETWORK_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/



/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(entrada_mapa_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_19_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.011654974892735481f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_19_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013317640870809555f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_20_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02083568647503853f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_20_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008413145318627357f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02083568647503853f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_21_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02759747952222824f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_21_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009280918166041374f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_22_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04770101606845856f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_22_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.011530512943863869f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04770101606845856f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_23_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0679277554154396f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_23_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00908975675702095f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_24_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13300126791000366f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_24_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0108910221606493f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13300126791000366f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_25_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.19942639768123627f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_25_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00806400179862976f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_26_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2809629738330841f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_26_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00920078158378601f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2809629738330841f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_27_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.4048124849796295f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_27_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00757428677752614f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_28_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.8235414028167725f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_28_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006410157773643732f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.8235414028167725f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_transpose_4_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.6743285059928894f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_transpose_4_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005712822079658508f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_concatenate_4_1_concat0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.6743285059928894f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_29_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.7520222067832947f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_29_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008270016871392727f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_30_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5698937177658081f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_30_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008230014704167843f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5698937177658081f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_transpose_5_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.6025909185409546f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_transpose_5_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005099537316709757f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_concatenate_5_1_concat0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.6025909185409546f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_31_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5974933505058289f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_31_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013053188100457191f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_32_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.9845296144485474f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_32_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008133894763886929f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.9845296144485474f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_transpose_6_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.49105551838874817f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_transpose_6_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.005272598471492529f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_concatenate_6_1_concat0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.49105551838874817f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_33_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5005602240562439f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_33_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0216634813696146f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_34_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5581063628196716f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #48 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_34_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010488662868738174f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #49 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.5581063628196716f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #50 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_transpose_7_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.32563793659210205f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #51 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_transpose_7_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006885238457471132f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #52 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_concatenate_7_1_concat0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.32563793659210205f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #53 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_35_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2518783509731293f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #54 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_35_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.032669547945261f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #55 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_36_1_Relu0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2936670780181885f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #56 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_36_1_Relu0_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015478333458304405f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #57 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Conv__2060_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.6340664625167847f),
    AI_PACK_INTQ_ZP(59)))

/* Int quant #58 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Conv__2060_weights_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.015459747985005379f),
    AI_PACK_INTQ_ZP(0)))

/* Int quant #59 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(Student_Unet_1_conv2d_37_1_Sigmoid0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00392156932502985f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #60 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(output_QuantizeLinear_Input_to_chlast_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00392156932502985f),
    AI_PACK_INTQ_ZP(-128)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  entrada_mapa_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 12289, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_19_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_19_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 216, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_19_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_19_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 572, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_20_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_20_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_20_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_20_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1472, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_21_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_21_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_21_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_21_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2656, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_22_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_22_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_22_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_22_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5248, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_23_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_23_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_23_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_23_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5824, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_24_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_24_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_24_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_24_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6400, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_25_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_25_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_25_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_25_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6528, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_26_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_26_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_26_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_26_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7680, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_27_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_27_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 73728, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_27_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_27_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7936, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_28_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_28_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 147456, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_28_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_28_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 10240, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6272, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7424, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_concatenate_4_1_concat0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_29_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_29_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 73728, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_29_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_29_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9984, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_30_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_30_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 36864, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_30_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_30_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7680, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 14400, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6272, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_concatenate_5_1_concat0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_31_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_31_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#67 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_31_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#68 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_31_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7552, AI_STATIC)

/* Array#69 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_32_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8192, AI_STATIC)

/* Array#70 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_32_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 9216, AI_STATIC)

/* Array#71 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_32_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#72 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_32_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6400, AI_STATIC)

/* Array#73 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 30752, AI_STATIC)

/* Array#74 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#75 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2048, AI_STATIC)

/* Array#76 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#77 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4672, AI_STATIC)

/* Array#78 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_concatenate_6_1_concat0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)

/* Array#79 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_33_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#80 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_33_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#81 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_33_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#82 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_33_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6336, AI_STATIC)

/* Array#83 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_34_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 16384, AI_STATIC)

/* Array#84 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_34_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2304, AI_STATIC)

/* Array#85 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_34_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#86 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_34_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5248, AI_STATIC)

/* Array#87 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 63504, AI_STATIC)

/* Array#88 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)

/* Array#89 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#90 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#91 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1312, AI_STATIC)

/* Array#92 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_concatenate_7_1_concat0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 65536, AI_STATIC)

/* Array#93 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_35_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)

/* Array#94 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_35_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1152, AI_STATIC)

/* Array#95 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_35_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#96 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_35_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2912, AI_STATIC)

/* Array#97 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_36_1_Relu0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32768, AI_STATIC)

/* Array#98 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_36_1_Relu0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 576, AI_STATIC)

/* Array#99 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_36_1_Relu0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8, AI_STATIC)

/* Array#100 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_36_1_Relu0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1472, AI_STATIC)

/* Array#101 */
AI_ARRAY_OBJ_DECLARE(
  Conv__2060_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#102 */
AI_ARRAY_OBJ_DECLARE(
  Conv__2060_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 8, AI_STATIC)

/* Array#103 */
AI_ARRAY_OBJ_DECLARE(
  Conv__2060_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1, AI_STATIC)

/* Array#104 */
AI_ARRAY_OBJ_DECLARE(
  Conv__2060_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32, AI_STATIC)

/* Array#105 */
AI_ARRAY_OBJ_DECLARE(
  Student_Unet_1_conv2d_37_1_Sigmoid0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#106 */
AI_ARRAY_OBJ_DECLARE(
  output_QuantizeLinear_Input_to_chlast_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 4096, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_19_1_Relu0_bias, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &Student_Unet_1_conv2d_19_1_Relu0_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_19_1_Relu0_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 64, 64), AI_STRIDE_INIT(4, 1, 1, 8, 512),
  1, &Student_Unet_1_conv2d_19_1_Relu0_output_array, &Student_Unet_1_conv2d_19_1_Relu0_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_19_1_Relu0_scratch0, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 572, 1, 1), AI_STRIDE_INIT(4, 1, 1, 572, 572),
  1, &Student_Unet_1_conv2d_19_1_Relu0_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_19_1_Relu0_weights, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 3, 3, 3, 8), AI_STRIDE_INIT(4, 1, 3, 24, 72),
  1, &Student_Unet_1_conv2d_19_1_Relu0_weights_array, &Student_Unet_1_conv2d_19_1_Relu0_weights_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  entrada_mapa_output, AI_STATIC,
  105, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 64, 64), AI_STRIDE_INIT(4, 1, 1, 3, 192),
  1, &entrada_mapa_output_array, &entrada_mapa_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_20_1_Relu0_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &Student_Unet_1_conv2d_20_1_Relu0_bias_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_20_1_Relu0_output, AI_STATIC,
  13, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 64, 64), AI_STRIDE_INIT(4, 1, 1, 8, 512),
  1, &Student_Unet_1_conv2d_20_1_Relu0_output_array, &Student_Unet_1_conv2d_20_1_Relu0_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_20_1_Relu0_scratch0, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 1472, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1472, 1472),
  1, &Student_Unet_1_conv2d_20_1_Relu0_scratch0_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_20_1_Relu0_weights, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 8, 3, 3, 8), AI_STRIDE_INIT(4, 1, 8, 64, 192),
  1, &Student_Unet_1_conv2d_20_1_Relu0_weights_array, &Student_Unet_1_conv2d_20_1_Relu0_weights_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_21_1_Relu0_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &Student_Unet_1_conv2d_21_1_Relu0_bias_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_21_1_Relu0_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 32, 32), AI_STRIDE_INIT(4, 1, 1, 16, 512),
  1, &Student_Unet_1_conv2d_21_1_Relu0_output_array, &Student_Unet_1_conv2d_21_1_Relu0_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_21_1_Relu0_scratch0, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 2656, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2656, 2656),
  1, &Student_Unet_1_conv2d_21_1_Relu0_scratch0_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_21_1_Relu0_weights, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 8, 3, 3, 16), AI_STRIDE_INIT(4, 1, 8, 128, 384),
  1, &Student_Unet_1_conv2d_21_1_Relu0_weights_array, &Student_Unet_1_conv2d_21_1_Relu0_weights_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_output, AI_STATIC,
  101, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 32, 32), AI_STRIDE_INIT(4, 1, 1, 8, 256),
  1, &Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_output_array, &Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_22_1_Relu0_bias, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &Student_Unet_1_conv2d_22_1_Relu0_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_22_1_Relu0_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 32, 32), AI_STRIDE_INIT(4, 1, 1, 16, 512),
  1, &Student_Unet_1_conv2d_22_1_Relu0_output_array, &Student_Unet_1_conv2d_22_1_Relu0_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_22_1_Relu0_scratch0, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 5248, 1, 1), AI_STRIDE_INIT(4, 1, 1, 5248, 5248),
  1, &Student_Unet_1_conv2d_22_1_Relu0_scratch0_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_22_1_Relu0_weights, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 3, 16), AI_STRIDE_INIT(4, 1, 16, 256, 768),
  1, &Student_Unet_1_conv2d_22_1_Relu0_weights_array, &Student_Unet_1_conv2d_22_1_Relu0_weights_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_23_1_Relu0_bias, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &Student_Unet_1_conv2d_23_1_Relu0_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_23_1_Relu0_output, AI_STATIC,
  25, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 1, 1, 32, 512),
  1, &Student_Unet_1_conv2d_23_1_Relu0_output_array, &Student_Unet_1_conv2d_23_1_Relu0_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_23_1_Relu0_scratch0, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 5824, 1, 1), AI_STRIDE_INIT(4, 1, 1, 5824, 5824),
  1, &Student_Unet_1_conv2d_23_1_Relu0_scratch0_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_23_1_Relu0_weights, AI_STATIC,
  27, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 1, 16, 512, 1536),
  1, &Student_Unet_1_conv2d_23_1_Relu0_weights_array, &Student_Unet_1_conv2d_23_1_Relu0_weights_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_output, AI_STATIC,
  102, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 1, 1, 16, 256),
  1, &Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_output_array, &Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_24_1_Relu0_bias, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &Student_Unet_1_conv2d_24_1_Relu0_bias_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_24_1_Relu0_output, AI_STATIC,
  29, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 1, 1, 32, 512),
  1, &Student_Unet_1_conv2d_24_1_Relu0_output_array, &Student_Unet_1_conv2d_24_1_Relu0_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_24_1_Relu0_scratch0, AI_STATIC,
  30, 0x0,
  AI_SHAPE_INIT(4, 1, 6400, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6400, 6400),
  1, &Student_Unet_1_conv2d_24_1_Relu0_scratch0_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_24_1_Relu0_weights, AI_STATIC,
  31, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 3, 32), AI_STRIDE_INIT(4, 1, 32, 1024, 3072),
  1, &Student_Unet_1_conv2d_24_1_Relu0_weights_array, &Student_Unet_1_conv2d_24_1_Relu0_weights_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_25_1_Relu0_bias, AI_STATIC,
  32, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &Student_Unet_1_conv2d_25_1_Relu0_bias_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_25_1_Relu0_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 8, 8), AI_STRIDE_INIT(4, 1, 1, 64, 512),
  1, &Student_Unet_1_conv2d_25_1_Relu0_output_array, &Student_Unet_1_conv2d_25_1_Relu0_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_25_1_Relu0_scratch0, AI_STATIC,
  34, 0x0,
  AI_SHAPE_INIT(4, 1, 6528, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6528, 6528),
  1, &Student_Unet_1_conv2d_25_1_Relu0_scratch0_array, NULL)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_25_1_Relu0_weights, AI_STATIC,
  35, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 3, 64), AI_STRIDE_INIT(4, 1, 32, 2048, 6144),
  1, &Student_Unet_1_conv2d_25_1_Relu0_weights_array, &Student_Unet_1_conv2d_25_1_Relu0_weights_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_output, AI_STATIC,
  103, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 8, 8), AI_STRIDE_INIT(4, 1, 1, 32, 256),
  1, &Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_output_array, &Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_output_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_26_1_Relu0_bias, AI_STATIC,
  36, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &Student_Unet_1_conv2d_26_1_Relu0_bias_array, NULL)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_26_1_Relu0_output, AI_STATIC,
  37, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 8, 8), AI_STRIDE_INIT(4, 1, 1, 64, 512),
  1, &Student_Unet_1_conv2d_26_1_Relu0_output_array, &Student_Unet_1_conv2d_26_1_Relu0_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_26_1_Relu0_scratch0, AI_STATIC,
  38, 0x0,
  AI_SHAPE_INIT(4, 1, 7680, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7680, 7680),
  1, &Student_Unet_1_conv2d_26_1_Relu0_scratch0_array, NULL)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_26_1_Relu0_weights, AI_STATIC,
  39, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 1, 64, 4096, 12288),
  1, &Student_Unet_1_conv2d_26_1_Relu0_weights_array, &Student_Unet_1_conv2d_26_1_Relu0_weights_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_27_1_Relu0_bias, AI_STATIC,
  40, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &Student_Unet_1_conv2d_27_1_Relu0_bias_array, NULL)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_27_1_Relu0_output, AI_STATIC,
  41, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 4, 4), AI_STRIDE_INIT(4, 1, 1, 128, 512),
  1, &Student_Unet_1_conv2d_27_1_Relu0_output_array, &Student_Unet_1_conv2d_27_1_Relu0_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_27_1_Relu0_scratch0, AI_STATIC,
  42, 0x0,
  AI_SHAPE_INIT(4, 1, 7936, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7936, 7936),
  1, &Student_Unet_1_conv2d_27_1_Relu0_scratch0_array, NULL)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_27_1_Relu0_weights, AI_STATIC,
  43, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 3, 128), AI_STRIDE_INIT(4, 1, 64, 8192, 24576),
  1, &Student_Unet_1_conv2d_27_1_Relu0_weights_array, &Student_Unet_1_conv2d_27_1_Relu0_weights_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_output, AI_STATIC,
  104, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 4, 4), AI_STRIDE_INIT(4, 1, 1, 64, 256),
  1, &Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_output_array, &Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_28_1_Relu0_bias, AI_STATIC,
  44, 0x0,
  AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &Student_Unet_1_conv2d_28_1_Relu0_bias_array, NULL)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_28_1_Relu0_output, AI_STATIC,
  45, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 4, 4), AI_STRIDE_INIT(4, 1, 1, 128, 512),
  1, &Student_Unet_1_conv2d_28_1_Relu0_output_array, &Student_Unet_1_conv2d_28_1_Relu0_output_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_28_1_Relu0_scratch0, AI_STATIC,
  46, 0x0,
  AI_SHAPE_INIT(4, 1, 10240, 1, 1), AI_STRIDE_INIT(4, 1, 1, 10240, 10240),
  1, &Student_Unet_1_conv2d_28_1_Relu0_scratch0_array, NULL)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_28_1_Relu0_weights, AI_STATIC,
  47, 0x1,
  AI_SHAPE_INIT(4, 128, 3, 3, 128), AI_STRIDE_INIT(4, 1, 128, 16384, 49152),
  1, &Student_Unet_1_conv2d_28_1_Relu0_weights_array, &Student_Unet_1_conv2d_28_1_Relu0_weights_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output, AI_STATIC,
  84, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 7, 7), AI_STRIDE_INIT(4, 1, 1, 128, 896),
  1, &Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output_array, &Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_bias, AI_STATIC,
  81, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &Student_Unet_1_conv2d_transpose_4_1_Relu0_bias_array, NULL)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_output, AI_STATIC,
  82, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 8, 8), AI_STRIDE_INIT(4, 1, 1, 64, 512),
  1, &Student_Unet_1_conv2d_transpose_4_1_Relu0_output_array, &Student_Unet_1_conv2d_transpose_4_1_Relu0_output_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_scratch0, AI_STATIC,
  83, 0x0,
  AI_SHAPE_INIT(4, 1, 7424, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7424, 7424),
  1, &Student_Unet_1_conv2d_transpose_4_1_Relu0_scratch0_array, NULL)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_weights, AI_STATIC,
  85, 0x1,
  AI_SHAPE_INIT(4, 128, 2, 2, 64), AI_STRIDE_INIT(4, 1, 128, 8192, 16384),
  1, &Student_Unet_1_conv2d_transpose_4_1_Relu0_weights_array, &Student_Unet_1_conv2d_transpose_4_1_Relu0_weights_array_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_concatenate_4_1_concat0_output, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 128, 8, 8), AI_STRIDE_INIT(4, 1, 1, 128, 1024),
  1, &Student_Unet_1_concatenate_4_1_concat0_output_array, &Student_Unet_1_concatenate_4_1_concat0_output_array_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_29_1_Relu0_bias, AI_STATIC,
  48, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &Student_Unet_1_conv2d_29_1_Relu0_bias_array, NULL)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_29_1_Relu0_output, AI_STATIC,
  49, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 8, 8), AI_STRIDE_INIT(4, 1, 1, 64, 512),
  1, &Student_Unet_1_conv2d_29_1_Relu0_output_array, &Student_Unet_1_conv2d_29_1_Relu0_output_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_29_1_Relu0_scratch0, AI_STATIC,
  50, 0x0,
  AI_SHAPE_INIT(4, 1, 9984, 1, 1), AI_STRIDE_INIT(4, 1, 1, 9984, 9984),
  1, &Student_Unet_1_conv2d_29_1_Relu0_scratch0_array, NULL)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_29_1_Relu0_weights, AI_STATIC,
  51, 0x1,
  AI_SHAPE_INIT(4, 128, 3, 3, 64), AI_STRIDE_INIT(4, 1, 128, 8192, 24576),
  1, &Student_Unet_1_conv2d_29_1_Relu0_weights_array, &Student_Unet_1_conv2d_29_1_Relu0_weights_array_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_30_1_Relu0_bias, AI_STATIC,
  52, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &Student_Unet_1_conv2d_30_1_Relu0_bias_array, NULL)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_30_1_Relu0_output, AI_STATIC,
  53, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 8, 8), AI_STRIDE_INIT(4, 1, 1, 64, 512),
  1, &Student_Unet_1_conv2d_30_1_Relu0_output_array, &Student_Unet_1_conv2d_30_1_Relu0_output_array_intq)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_30_1_Relu0_scratch0, AI_STATIC,
  54, 0x0,
  AI_SHAPE_INIT(4, 1, 7680, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7680, 7680),
  1, &Student_Unet_1_conv2d_30_1_Relu0_scratch0_array, NULL)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_30_1_Relu0_weights, AI_STATIC,
  55, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 3, 64), AI_STRIDE_INIT(4, 1, 64, 4096, 12288),
  1, &Student_Unet_1_conv2d_30_1_Relu0_weights_array, &Student_Unet_1_conv2d_30_1_Relu0_weights_array_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output, AI_STATIC,
  89, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 15, 15), AI_STRIDE_INIT(4, 1, 1, 64, 960),
  1, &Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output_array, &Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output_array_intq)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_bias, AI_STATIC,
  86, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &Student_Unet_1_conv2d_transpose_5_1_Relu0_bias_array, NULL)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_output, AI_STATIC,
  87, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 1, 1, 32, 512),
  1, &Student_Unet_1_conv2d_transpose_5_1_Relu0_output_array, &Student_Unet_1_conv2d_transpose_5_1_Relu0_output_array_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_scratch0, AI_STATIC,
  88, 0x0,
  AI_SHAPE_INIT(4, 1, 6272, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6272, 6272),
  1, &Student_Unet_1_conv2d_transpose_5_1_Relu0_scratch0_array, NULL)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_weights, AI_STATIC,
  90, 0x1,
  AI_SHAPE_INIT(4, 64, 2, 2, 32), AI_STRIDE_INIT(4, 1, 64, 2048, 4096),
  1, &Student_Unet_1_conv2d_transpose_5_1_Relu0_weights_array, &Student_Unet_1_conv2d_transpose_5_1_Relu0_weights_array_intq)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_concatenate_5_1_concat0_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 16, 16), AI_STRIDE_INIT(4, 1, 1, 64, 1024),
  1, &Student_Unet_1_concatenate_5_1_concat0_output_array, &Student_Unet_1_concatenate_5_1_concat0_output_array_intq)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_31_1_Relu0_bias, AI_STATIC,
  56, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &Student_Unet_1_conv2d_31_1_Relu0_bias_array, NULL)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_31_1_Relu0_output, AI_STATIC,
  57, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 1, 1, 32, 512),
  1, &Student_Unet_1_conv2d_31_1_Relu0_output_array, &Student_Unet_1_conv2d_31_1_Relu0_output_array_intq)

/* Tensor #67 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_31_1_Relu0_scratch0, AI_STATIC,
  58, 0x0,
  AI_SHAPE_INIT(4, 1, 7552, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7552, 7552),
  1, &Student_Unet_1_conv2d_31_1_Relu0_scratch0_array, NULL)

/* Tensor #68 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_31_1_Relu0_weights, AI_STATIC,
  59, 0x1,
  AI_SHAPE_INIT(4, 64, 3, 3, 32), AI_STRIDE_INIT(4, 1, 64, 2048, 6144),
  1, &Student_Unet_1_conv2d_31_1_Relu0_weights_array, &Student_Unet_1_conv2d_31_1_Relu0_weights_array_intq)

/* Tensor #69 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_32_1_Relu0_bias, AI_STATIC,
  60, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &Student_Unet_1_conv2d_32_1_Relu0_bias_array, NULL)

/* Tensor #70 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_32_1_Relu0_output, AI_STATIC,
  61, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 16, 16), AI_STRIDE_INIT(4, 1, 1, 32, 512),
  1, &Student_Unet_1_conv2d_32_1_Relu0_output_array, &Student_Unet_1_conv2d_32_1_Relu0_output_array_intq)

/* Tensor #71 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_32_1_Relu0_scratch0, AI_STATIC,
  62, 0x0,
  AI_SHAPE_INIT(4, 1, 6400, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6400, 6400),
  1, &Student_Unet_1_conv2d_32_1_Relu0_scratch0_array, NULL)

/* Tensor #72 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_32_1_Relu0_weights, AI_STATIC,
  63, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 3, 32), AI_STRIDE_INIT(4, 1, 32, 1024, 3072),
  1, &Student_Unet_1_conv2d_32_1_Relu0_weights_array, &Student_Unet_1_conv2d_32_1_Relu0_weights_array_intq)

/* Tensor #73 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output, AI_STATIC,
  94, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 31, 31), AI_STRIDE_INIT(4, 1, 1, 32, 992),
  1, &Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output_array, &Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output_array_intq)

/* Tensor #74 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_bias, AI_STATIC,
  91, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &Student_Unet_1_conv2d_transpose_6_1_Relu0_bias_array, NULL)

/* Tensor #75 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_output, AI_STATIC,
  92, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 32, 32), AI_STRIDE_INIT(4, 1, 1, 16, 512),
  1, &Student_Unet_1_conv2d_transpose_6_1_Relu0_output_array, &Student_Unet_1_conv2d_transpose_6_1_Relu0_output_array_intq)

/* Tensor #76 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_scratch0, AI_STATIC,
  93, 0x0,
  AI_SHAPE_INIT(4, 1, 4672, 1, 1), AI_STRIDE_INIT(4, 1, 1, 4672, 4672),
  1, &Student_Unet_1_conv2d_transpose_6_1_Relu0_scratch0_array, NULL)

/* Tensor #77 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_weights, AI_STATIC,
  95, 0x1,
  AI_SHAPE_INIT(4, 32, 2, 2, 16), AI_STRIDE_INIT(4, 1, 32, 512, 1024),
  1, &Student_Unet_1_conv2d_transpose_6_1_Relu0_weights_array, &Student_Unet_1_conv2d_transpose_6_1_Relu0_weights_array_intq)

/* Tensor #78 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_concatenate_6_1_concat0_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 32, 32), AI_STRIDE_INIT(4, 1, 1, 32, 1024),
  1, &Student_Unet_1_concatenate_6_1_concat0_output_array, &Student_Unet_1_concatenate_6_1_concat0_output_array_intq)

/* Tensor #79 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_33_1_Relu0_bias, AI_STATIC,
  64, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &Student_Unet_1_conv2d_33_1_Relu0_bias_array, NULL)

/* Tensor #80 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_33_1_Relu0_output, AI_STATIC,
  65, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 32, 32), AI_STRIDE_INIT(4, 1, 1, 16, 512),
  1, &Student_Unet_1_conv2d_33_1_Relu0_output_array, &Student_Unet_1_conv2d_33_1_Relu0_output_array_intq)

/* Tensor #81 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_33_1_Relu0_scratch0, AI_STATIC,
  66, 0x0,
  AI_SHAPE_INIT(4, 1, 6336, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6336, 6336),
  1, &Student_Unet_1_conv2d_33_1_Relu0_scratch0_array, NULL)

/* Tensor #82 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_33_1_Relu0_weights, AI_STATIC,
  67, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 3, 16), AI_STRIDE_INIT(4, 1, 32, 512, 1536),
  1, &Student_Unet_1_conv2d_33_1_Relu0_weights_array, &Student_Unet_1_conv2d_33_1_Relu0_weights_array_intq)

/* Tensor #83 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_34_1_Relu0_bias, AI_STATIC,
  68, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &Student_Unet_1_conv2d_34_1_Relu0_bias_array, NULL)

/* Tensor #84 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_34_1_Relu0_output, AI_STATIC,
  69, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 32, 32), AI_STRIDE_INIT(4, 1, 1, 16, 512),
  1, &Student_Unet_1_conv2d_34_1_Relu0_output_array, &Student_Unet_1_conv2d_34_1_Relu0_output_array_intq)

/* Tensor #85 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_34_1_Relu0_scratch0, AI_STATIC,
  70, 0x0,
  AI_SHAPE_INIT(4, 1, 5248, 1, 1), AI_STRIDE_INIT(4, 1, 1, 5248, 5248),
  1, &Student_Unet_1_conv2d_34_1_Relu0_scratch0_array, NULL)

/* Tensor #86 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_34_1_Relu0_weights, AI_STATIC,
  71, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 3, 16), AI_STRIDE_INIT(4, 1, 16, 256, 768),
  1, &Student_Unet_1_conv2d_34_1_Relu0_weights_array, &Student_Unet_1_conv2d_34_1_Relu0_weights_array_intq)

/* Tensor #87 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output, AI_STATIC,
  99, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 63, 63), AI_STRIDE_INIT(4, 1, 1, 16, 1008),
  1, &Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output_array, &Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output_array_intq)

/* Tensor #88 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_bias, AI_STATIC,
  96, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &Student_Unet_1_conv2d_transpose_7_1_Relu0_bias_array, NULL)

/* Tensor #89 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_output, AI_STATIC,
  97, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 64, 64), AI_STRIDE_INIT(4, 1, 1, 8, 512),
  1, &Student_Unet_1_conv2d_transpose_7_1_Relu0_output_array, &Student_Unet_1_conv2d_transpose_7_1_Relu0_output_array_intq)

/* Tensor #90 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_scratch0, AI_STATIC,
  98, 0x0,
  AI_SHAPE_INIT(4, 1, 1312, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1312, 1312),
  1, &Student_Unet_1_conv2d_transpose_7_1_Relu0_scratch0_array, NULL)

/* Tensor #91 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_weights, AI_STATIC,
  100, 0x1,
  AI_SHAPE_INIT(4, 16, 2, 2, 8), AI_STRIDE_INIT(4, 1, 16, 128, 256),
  1, &Student_Unet_1_conv2d_transpose_7_1_Relu0_weights_array, &Student_Unet_1_conv2d_transpose_7_1_Relu0_weights_array_intq)

/* Tensor #92 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_concatenate_7_1_concat0_output, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 64, 64), AI_STRIDE_INIT(4, 1, 1, 16, 1024),
  1, &Student_Unet_1_concatenate_7_1_concat0_output_array, &Student_Unet_1_concatenate_7_1_concat0_output_array_intq)

/* Tensor #93 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_35_1_Relu0_bias, AI_STATIC,
  72, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &Student_Unet_1_conv2d_35_1_Relu0_bias_array, NULL)

/* Tensor #94 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_35_1_Relu0_output, AI_STATIC,
  73, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 64, 64), AI_STRIDE_INIT(4, 1, 1, 8, 512),
  1, &Student_Unet_1_conv2d_35_1_Relu0_output_array, &Student_Unet_1_conv2d_35_1_Relu0_output_array_intq)

/* Tensor #95 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_35_1_Relu0_scratch0, AI_STATIC,
  74, 0x0,
  AI_SHAPE_INIT(4, 1, 2912, 1, 1), AI_STRIDE_INIT(4, 1, 1, 2912, 2912),
  1, &Student_Unet_1_conv2d_35_1_Relu0_scratch0_array, NULL)

/* Tensor #96 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_35_1_Relu0_weights, AI_STATIC,
  75, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 3, 8), AI_STRIDE_INIT(4, 1, 16, 128, 384),
  1, &Student_Unet_1_conv2d_35_1_Relu0_weights_array, &Student_Unet_1_conv2d_35_1_Relu0_weights_array_intq)

/* Tensor #97 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_36_1_Relu0_bias, AI_STATIC,
  76, 0x0,
  AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &Student_Unet_1_conv2d_36_1_Relu0_bias_array, NULL)

/* Tensor #98 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_36_1_Relu0_output, AI_STATIC,
  77, 0x1,
  AI_SHAPE_INIT(4, 1, 8, 64, 64), AI_STRIDE_INIT(4, 1, 1, 8, 512),
  1, &Student_Unet_1_conv2d_36_1_Relu0_output_array, &Student_Unet_1_conv2d_36_1_Relu0_output_array_intq)

/* Tensor #99 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_36_1_Relu0_scratch0, AI_STATIC,
  78, 0x0,
  AI_SHAPE_INIT(4, 1, 1472, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1472, 1472),
  1, &Student_Unet_1_conv2d_36_1_Relu0_scratch0_array, NULL)

/* Tensor #100 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_36_1_Relu0_weights, AI_STATIC,
  79, 0x1,
  AI_SHAPE_INIT(4, 8, 3, 3, 8), AI_STRIDE_INIT(4, 1, 8, 64, 192),
  1, &Student_Unet_1_conv2d_36_1_Relu0_weights_array, &Student_Unet_1_conv2d_36_1_Relu0_weights_array_intq)

/* Tensor #101 */
AI_TENSOR_OBJ_DECLARE(
  Conv__2060_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &Conv__2060_bias_array, NULL)

/* Tensor #102 */
AI_TENSOR_OBJ_DECLARE(
  Conv__2060_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 64, 64), AI_STRIDE_INIT(4, 1, 1, 1, 64),
  1, &Conv__2060_output_array, &Conv__2060_output_array_intq)

/* Tensor #103 */
AI_TENSOR_OBJ_DECLARE(
  Conv__2060_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &Conv__2060_scratch0_array, NULL)

/* Tensor #104 */
AI_TENSOR_OBJ_DECLARE(
  Conv__2060_weights, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 8, 1, 1, 1), AI_STRIDE_INIT(4, 1, 8, 8, 8),
  1, &Conv__2060_weights_array, &Conv__2060_weights_array_intq)

/* Tensor #105 */
AI_TENSOR_OBJ_DECLARE(
  Student_Unet_1_conv2d_37_1_Sigmoid0_output, AI_STATIC,
  80, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 64, 64), AI_STRIDE_INIT(4, 1, 1, 1, 64),
  1, &Student_Unet_1_conv2d_37_1_Sigmoid0_output_array, &Student_Unet_1_conv2d_37_1_Sigmoid0_output_array_intq)

/* Tensor #106 */
AI_TENSOR_OBJ_DECLARE(
  output_QuantizeLinear_Input_to_chlast_output, AI_STATIC,
  106, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 64, 1), AI_STRIDE_INIT(4, 1, 1, 64, 4096),
  1, &output_QuantizeLinear_Input_to_chlast_output_array, &output_QuantizeLinear_Input_to_chlast_output_array_intq)


AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_19_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &entrada_mapa_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_19_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_19_1_Relu0_weights, &Student_Unet_1_conv2d_19_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_19_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_19_1_Relu0_layer, 50,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_19_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_19_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_20_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_19_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_20_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_20_1_Relu0_weights, &Student_Unet_1_conv2d_20_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_20_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_20_1_Relu0_layer, 53,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_20_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_20_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_21_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_21_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_21_1_Relu0_weights, &Student_Unet_1_conv2d_21_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_21_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_21_1_Relu0_layer, 59,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_21_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_21_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_22_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_21_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_22_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_22_1_Relu0_weights, &Student_Unet_1_conv2d_22_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_22_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_22_1_Relu0_layer, 62,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_22_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_22_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_23_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_23_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_23_1_Relu0_weights, &Student_Unet_1_conv2d_23_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_23_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_23_1_Relu0_layer, 68,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_23_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_23_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_24_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_23_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_24_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_24_1_Relu0_weights, &Student_Unet_1_conv2d_24_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_24_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_24_1_Relu0_layer, 71,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_24_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_24_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_25_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_25_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_25_1_Relu0_weights, &Student_Unet_1_conv2d_25_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_25_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_25_1_Relu0_layer, 77,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_25_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_25_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_26_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_25_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_26_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_26_1_Relu0_weights, &Student_Unet_1_conv2d_26_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_26_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_26_1_Relu0_layer, 80,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_26_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_26_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_27_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_27_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_27_1_Relu0_weights, &Student_Unet_1_conv2d_27_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_27_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_27_1_Relu0_layer, 86,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_27_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_27_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_28_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_27_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_28_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_28_1_Relu0_weights, &Student_Unet_1_conv2d_28_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_28_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_28_1_Relu0_layer, 89,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_28_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_28_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_scales_data[] = { 2, 2, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_scales, AI_ARRAY_FORMAT_FLOAT,
    Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_scales_data, Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_28_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_layer, 92,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_zeros_is8os8,
  &Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_chain,
  NULL, &Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_layer, AI_STATIC, 
  .scales = &Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_ZEROS, 
  .nearest_mode = AI_ROUND_PREFER_CEIL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_4_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_transpose_4_1_Relu0_weights, &Student_Unet_1_conv2d_transpose_4_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_4_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_4_1_Relu0_layer, 92,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_transpose_4_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_transpose_4_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_concatenate_4_1_concat0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &Student_Unet_1_conv2d_transpose_4_1_Relu0_output, &Student_Unet_1_conv2d_26_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_concatenate_4_1_concat0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_concatenate_4_1_concat0_layer, 95,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &Student_Unet_1_concatenate_4_1_concat0_chain,
  NULL, &Student_Unet_1_concatenate_4_1_concat0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_29_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_concatenate_4_1_concat0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_29_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_29_1_Relu0_weights, &Student_Unet_1_conv2d_29_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_29_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_29_1_Relu0_layer, 98,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_29_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_29_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_30_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_29_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_30_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_30_1_Relu0_weights, &Student_Unet_1_conv2d_30_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_30_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_30_1_Relu0_layer, 101,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_30_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_30_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_scales_data[] = { 2, 2, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_scales, AI_ARRAY_FORMAT_FLOAT,
    Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_scales_data, Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_30_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_layer, 104,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_zeros_is8os8,
  &Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_chain,
  NULL, &Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_layer, AI_STATIC, 
  .scales = &Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_ZEROS, 
  .nearest_mode = AI_ROUND_PREFER_CEIL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_5_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_transpose_5_1_Relu0_weights, &Student_Unet_1_conv2d_transpose_5_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_5_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_5_1_Relu0_layer, 104,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_transpose_5_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_transpose_5_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_concatenate_5_1_concat0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &Student_Unet_1_conv2d_transpose_5_1_Relu0_output, &Student_Unet_1_conv2d_24_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_concatenate_5_1_concat0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_concatenate_5_1_concat0_layer, 107,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &Student_Unet_1_concatenate_5_1_concat0_chain,
  NULL, &Student_Unet_1_concatenate_5_1_concat0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_31_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_concatenate_5_1_concat0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_31_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_31_1_Relu0_weights, &Student_Unet_1_conv2d_31_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_31_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_31_1_Relu0_layer, 110,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_31_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_31_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_32_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_31_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_32_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_32_1_Relu0_weights, &Student_Unet_1_conv2d_32_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_32_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_32_1_Relu0_layer, 113,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_32_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_32_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_scales_data[] = { 2, 2, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_scales, AI_ARRAY_FORMAT_FLOAT,
    Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_scales_data, Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_32_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_layer, 116,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_zeros_is8os8,
  &Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_chain,
  NULL, &Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_layer, AI_STATIC, 
  .scales = &Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_ZEROS, 
  .nearest_mode = AI_ROUND_PREFER_CEIL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_6_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_transpose_6_1_Relu0_weights, &Student_Unet_1_conv2d_transpose_6_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_6_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_6_1_Relu0_layer, 116,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_transpose_6_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_transpose_6_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_concatenate_6_1_concat0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &Student_Unet_1_conv2d_transpose_6_1_Relu0_output, &Student_Unet_1_conv2d_22_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_concatenate_6_1_concat0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_concatenate_6_1_concat0_layer, 119,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &Student_Unet_1_concatenate_6_1_concat0_chain,
  NULL, &Student_Unet_1_concatenate_6_1_concat0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_33_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_concatenate_6_1_concat0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_33_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_33_1_Relu0_weights, &Student_Unet_1_conv2d_33_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_33_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_33_1_Relu0_layer, 122,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_33_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_33_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_34_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_33_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_34_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_34_1_Relu0_weights, &Student_Unet_1_conv2d_34_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_34_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_34_1_Relu0_layer, 125,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_34_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_34_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_float Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_scales_data[] = { 2, 2, 1.0, 1.0 };
AI_ARRAY_OBJ_DECLARE(
    Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_scales, AI_ARRAY_FORMAT_FLOAT,
    Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_scales_data, Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_scales_data, 4, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_34_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_layer, 128,
  UPSAMPLE_TYPE, 0x0, NULL,
  upsample, forward_upsample_zeros_is8os8,
  &Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_chain,
  NULL, &Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_layer, AI_STATIC, 
  .scales = &Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_scales, 
  .center = false, 
  .mode = AI_UPSAMPLE_ZEROS, 
  .nearest_mode = AI_ROUND_PREFER_CEIL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_7_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_transpose_7_1_Relu0_weights, &Student_Unet_1_conv2d_transpose_7_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_transpose_7_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_transpose_7_1_Relu0_layer, 128,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_transpose_7_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_transpose_7_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_concatenate_7_1_concat0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &Student_Unet_1_conv2d_transpose_7_1_Relu0_output, &Student_Unet_1_conv2d_20_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_concatenate_7_1_concat0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_concatenate_7_1_concat0_layer, 131,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &Student_Unet_1_concatenate_7_1_concat0_chain,
  NULL, &Student_Unet_1_concatenate_7_1_concat0_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_35_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_concatenate_7_1_concat0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_35_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_35_1_Relu0_weights, &Student_Unet_1_conv2d_35_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_35_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_35_1_Relu0_layer, 134,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_35_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_35_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_36_1_Relu0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_35_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_36_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Student_Unet_1_conv2d_36_1_Relu0_weights, &Student_Unet_1_conv2d_36_1_Relu0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_36_1_Relu0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_36_1_Relu0_layer, 137,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Student_Unet_1_conv2d_36_1_Relu0_chain,
  NULL, &Student_Unet_1_conv2d_36_1_Relu0_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_SAME, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  Conv__2060_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_36_1_Relu0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__2060_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &Conv__2060_weights, &Conv__2060_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__2060_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  Conv__2060_layer, 140,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d_integer_SSSA,
  &Conv__2060_chain,
  NULL, &Conv__2060_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


AI_STATIC_CONST ai_i8 Student_Unet_1_conv2d_37_1_Sigmoid0_nl_params_data[] = { -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -127, -126, -125, -122, -118, -109, -95, -72, -40, -1, 39, 71, 94, 108, 117, 121, 124, 125, 126, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
AI_ARRAY_OBJ_DECLARE(
    Student_Unet_1_conv2d_37_1_Sigmoid0_nl_params, AI_ARRAY_FORMAT_S8,
    Student_Unet_1_conv2d_37_1_Sigmoid0_nl_params_data, Student_Unet_1_conv2d_37_1_Sigmoid0_nl_params_data, 256, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  Student_Unet_1_conv2d_37_1_Sigmoid0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Conv__2060_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_37_1_Sigmoid0_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  Student_Unet_1_conv2d_37_1_Sigmoid0_layer, 143,
  NL_TYPE, 0x0, NULL,
  nl, forward_nl_integer,
  &Student_Unet_1_conv2d_37_1_Sigmoid0_chain,
  NULL, &Student_Unet_1_conv2d_37_1_Sigmoid0_layer, AI_STATIC, 
  .nl_params = &Student_Unet_1_conv2d_37_1_Sigmoid0_nl_params, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  output_QuantizeLinear_Input_to_chlast_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &Student_Unet_1_conv2d_37_1_Sigmoid0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &output_QuantizeLinear_Input_to_chlast_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  output_QuantizeLinear_Input_to_chlast_layer, 146,
  TRANSPOSE_TYPE, 0x0, NULL,
  transpose, forward_transpose,
  &output_QuantizeLinear_Input_to_chlast_chain,
  NULL, &output_QuantizeLinear_Input_to_chlast_layer, AI_STATIC, 
  .out_mapping = AI_SHAPE_INIT(6, AI_SHAPE_IN_CHANNEL, AI_SHAPE_WIDTH, AI_SHAPE_HEIGHT, AI_SHAPE_CHANNEL, AI_SHAPE_DEPTH, AI_SHAPE_EXTENSION), 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_19_1_Relu0(_stai_network_context* net_ctx)
{
  entrada_mapa_output_array.data = AI_PTR(net_ctx->_inputs[0] + 0);
  entrada_mapa_output_array.data_start = AI_PTR(net_ctx->_inputs[0] + 0);
  Student_Unet_1_conv2d_19_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 0);
  Student_Unet_1_conv2d_19_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 0);
  Student_Unet_1_conv2d_19_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 216);
  Student_Unet_1_conv2d_19_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 216);
  Student_Unet_1_conv2d_19_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_19_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_19_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 1040);
  Student_Unet_1_conv2d_19_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 1040);
  _STAI_NETWORK_EVENT_NODE_START_CB(50, 1, { entrada_mapa_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_19_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(50, 1, { Student_Unet_1_conv2d_19_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_20_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_19_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 1040);
  Student_Unet_1_conv2d_19_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 1040);
  Student_Unet_1_conv2d_20_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 248);
  Student_Unet_1_conv2d_20_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 248);
  Student_Unet_1_conv2d_20_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 824);
  Student_Unet_1_conv2d_20_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 824);
  Student_Unet_1_conv2d_20_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 572);
  Student_Unet_1_conv2d_20_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 572);
  Student_Unet_1_conv2d_20_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  Student_Unet_1_conv2d_20_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(53, 1, { Student_Unet_1_conv2d_19_1_Relu0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_20_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(53, 1, { Student_Unet_1_conv2d_20_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_21_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_21_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 856);
  Student_Unet_1_conv2d_21_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 856);
  Student_Unet_1_conv2d_21_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 2008);
  Student_Unet_1_conv2d_21_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 2008);
  Student_Unet_1_conv2d_21_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 2044);
  Student_Unet_1_conv2d_21_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 2044);
  Student_Unet_1_conv2d_21_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 40960);
  Student_Unet_1_conv2d_21_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 40960);
  _STAI_NETWORK_EVENT_NODE_START_CB(59, 1, { Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_21_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(59, 1, { Student_Unet_1_conv2d_21_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_22_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_21_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 40960);
  Student_Unet_1_conv2d_21_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 40960);
  Student_Unet_1_conv2d_22_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 2072);
  Student_Unet_1_conv2d_22_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 2072);
  Student_Unet_1_conv2d_22_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 4376);
  Student_Unet_1_conv2d_22_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 4376);
  Student_Unet_1_conv2d_22_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 4700);
  Student_Unet_1_conv2d_22_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 4700);
  Student_Unet_1_conv2d_22_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 39904);
  Student_Unet_1_conv2d_22_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 39904);
  _STAI_NETWORK_EVENT_NODE_START_CB(62, 1, { Student_Unet_1_conv2d_21_1_Relu0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_22_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(62, 1, { Student_Unet_1_conv2d_22_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_23_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_23_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 4440);
  Student_Unet_1_conv2d_23_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 4440);
  Student_Unet_1_conv2d_23_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 9048);
  Student_Unet_1_conv2d_23_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 9048);
  Student_Unet_1_conv2d_23_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_23_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_23_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_23_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 56288);
  _STAI_NETWORK_EVENT_NODE_START_CB(68, 1, { Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_23_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(68, 1, { Student_Unet_1_conv2d_23_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_24_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_23_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_23_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_24_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 9176);
  Student_Unet_1_conv2d_24_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 9176);
  Student_Unet_1_conv2d_24_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 18392);
  Student_Unet_1_conv2d_24_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 18392);
  Student_Unet_1_conv2d_24_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_24_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_24_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 64480);
  Student_Unet_1_conv2d_24_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 64480);
  _STAI_NETWORK_EVENT_NODE_START_CB(71, 1, { Student_Unet_1_conv2d_23_1_Relu0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_24_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(71, 1, { Student_Unet_1_conv2d_24_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_25_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_25_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 18520);
  Student_Unet_1_conv2d_25_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 18520);
  Student_Unet_1_conv2d_25_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 36952);
  Student_Unet_1_conv2d_25_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 36952);
  Student_Unet_1_conv2d_25_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_25_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_25_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 34816);
  Student_Unet_1_conv2d_25_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 34816);
  _STAI_NETWORK_EVENT_NODE_START_CB(77, 1, { Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_25_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(77, 1, { Student_Unet_1_conv2d_25_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_26_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_25_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 34816);
  Student_Unet_1_conv2d_25_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 34816);
  Student_Unet_1_conv2d_26_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 37208);
  Student_Unet_1_conv2d_26_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 37208);
  Student_Unet_1_conv2d_26_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 74072);
  Student_Unet_1_conv2d_26_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 74072);
  Student_Unet_1_conv2d_26_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_26_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_26_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_26_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 56288);
  _STAI_NETWORK_EVENT_NODE_START_CB(80, 1, { Student_Unet_1_conv2d_25_1_Relu0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_26_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(80, 1, { Student_Unet_1_conv2d_26_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_27_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_27_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 74328);
  Student_Unet_1_conv2d_27_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 74328);
  Student_Unet_1_conv2d_27_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 148056);
  Student_Unet_1_conv2d_27_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 148056);
  Student_Unet_1_conv2d_27_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_27_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_27_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 33792);
  Student_Unet_1_conv2d_27_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 33792);
  _STAI_NETWORK_EVENT_NODE_START_CB(86, 1, { Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_27_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(86, 1, { Student_Unet_1_conv2d_27_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_28_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_27_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 33792);
  Student_Unet_1_conv2d_27_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 33792);
  Student_Unet_1_conv2d_28_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 148568);
  Student_Unet_1_conv2d_28_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 148568);
  Student_Unet_1_conv2d_28_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 296024);
  Student_Unet_1_conv2d_28_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 296024);
  Student_Unet_1_conv2d_28_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 9948);
  Student_Unet_1_conv2d_28_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 9948);
  Student_Unet_1_conv2d_28_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 35840);
  Student_Unet_1_conv2d_28_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 35840);
  _STAI_NETWORK_EVENT_NODE_START_CB(89, 1, { Student_Unet_1_conv2d_27_1_Relu0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_28_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(89, 1, { Student_Unet_1_conv2d_28_1_Relu0_output.data->data});
}
void forward_lite_upsample_zeros_is8os8_Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_28_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 35840);
  Student_Unet_1_conv2d_28_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 35840);
  Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output_array.data = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 72672);
  _STAI_NETWORK_EVENT_NODE_START_CB(92, 1, { Student_Unet_1_conv2d_28_1_Relu0_output.data->data});
  forward_upsample_zeros_is8os8(&Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(92, 1, { Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_transpose_4_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output_array.data = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_conv2d_transpose_4_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 296536);
  Student_Unet_1_conv2d_transpose_4_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 296536);
  Student_Unet_1_conv2d_transpose_4_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 329304);
  Student_Unet_1_conv2d_transpose_4_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 329304);
  Student_Unet_1_conv2d_transpose_4_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_transpose_4_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_transpose_4_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_transpose_4_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  _STAI_NETWORK_EVENT_NODE_START_CB(92, 1, { Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_transpose_4_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(92, 1, { Student_Unet_1_conv2d_transpose_4_1_Relu0_output.data->data});
}
void forward_lite_concat_Student_Unet_1_concatenate_4_1_concat0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_transpose_4_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_transpose_4_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_26_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_26_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_concatenate_4_1_concat0_output_array.data = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_concatenate_4_1_concat0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 72672);
  _STAI_NETWORK_EVENT_NODE_START_CB(95, 2, { Student_Unet_1_conv2d_transpose_4_1_Relu0_output.data->data,Student_Unet_1_conv2d_26_1_Relu0_output.data->data});
  forward_concat(&Student_Unet_1_concatenate_4_1_concat0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(95, 1, { Student_Unet_1_concatenate_4_1_concat0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_29_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_concatenate_4_1_concat0_output_array.data = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_concatenate_4_1_concat0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_conv2d_29_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 329560);
  Student_Unet_1_conv2d_29_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 329560);
  Student_Unet_1_conv2d_29_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 403288);
  Student_Unet_1_conv2d_29_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 403288);
  Student_Unet_1_conv2d_29_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_29_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_29_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_29_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  _STAI_NETWORK_EVENT_NODE_START_CB(98, 1, { Student_Unet_1_concatenate_4_1_concat0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_29_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(98, 1, { Student_Unet_1_conv2d_29_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_30_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_29_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_29_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_30_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 403544);
  Student_Unet_1_conv2d_30_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 403544);
  Student_Unet_1_conv2d_30_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 440408);
  Student_Unet_1_conv2d_30_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 440408);
  Student_Unet_1_conv2d_30_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_30_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_30_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_30_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 56288);
  _STAI_NETWORK_EVENT_NODE_START_CB(101, 1, { Student_Unet_1_conv2d_29_1_Relu0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_30_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(101, 1, { Student_Unet_1_conv2d_30_1_Relu0_output.data->data});
}
void forward_lite_upsample_zeros_is8os8_Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_30_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_30_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output_array.data = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 72672);
  _STAI_NETWORK_EVENT_NODE_START_CB(104, 1, { Student_Unet_1_conv2d_30_1_Relu0_output.data->data});
  forward_upsample_zeros_is8os8(&Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(104, 1, { Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_transpose_5_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output_array.data = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_conv2d_transpose_5_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 440664);
  Student_Unet_1_conv2d_transpose_5_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 440664);
  Student_Unet_1_conv2d_transpose_5_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 448856);
  Student_Unet_1_conv2d_transpose_5_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 448856);
  Student_Unet_1_conv2d_transpose_5_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_transpose_5_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_transpose_5_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_transpose_5_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 56288);
  _STAI_NETWORK_EVENT_NODE_START_CB(104, 1, { Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_transpose_5_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(104, 1, { Student_Unet_1_conv2d_transpose_5_1_Relu0_output.data->data});
}
void forward_lite_concat_Student_Unet_1_concatenate_5_1_concat0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_transpose_5_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_transpose_5_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_24_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 64480);
  Student_Unet_1_conv2d_24_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 64480);
  Student_Unet_1_concatenate_5_1_concat0_output_array.data = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_concatenate_5_1_concat0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 72672);
  _STAI_NETWORK_EVENT_NODE_START_CB(107, 2, { Student_Unet_1_conv2d_transpose_5_1_Relu0_output.data->data,Student_Unet_1_conv2d_24_1_Relu0_output.data->data});
  forward_concat(&Student_Unet_1_concatenate_5_1_concat0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(107, 1, { Student_Unet_1_concatenate_5_1_concat0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_31_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_concatenate_5_1_concat0_output_array.data = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_concatenate_5_1_concat0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_conv2d_31_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 448984);
  Student_Unet_1_conv2d_31_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 448984);
  Student_Unet_1_conv2d_31_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 467416);
  Student_Unet_1_conv2d_31_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 467416);
  Student_Unet_1_conv2d_31_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_31_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_31_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_31_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 56288);
  _STAI_NETWORK_EVENT_NODE_START_CB(110, 1, { Student_Unet_1_concatenate_5_1_concat0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_31_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(110, 1, { Student_Unet_1_conv2d_31_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_32_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_31_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_31_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_32_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 467544);
  Student_Unet_1_conv2d_32_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 467544);
  Student_Unet_1_conv2d_32_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 476760);
  Student_Unet_1_conv2d_32_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 476760);
  Student_Unet_1_conv2d_32_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_32_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_32_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 64480);
  Student_Unet_1_conv2d_32_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 64480);
  _STAI_NETWORK_EVENT_NODE_START_CB(113, 1, { Student_Unet_1_conv2d_31_1_Relu0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_32_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(113, 1, { Student_Unet_1_conv2d_32_1_Relu0_output.data->data});
}
void forward_lite_upsample_zeros_is8os8_Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_32_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 64480);
  Student_Unet_1_conv2d_32_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 64480);
  Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output_array.data = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 72672);
  _STAI_NETWORK_EVENT_NODE_START_CB(116, 1, { Student_Unet_1_conv2d_32_1_Relu0_output.data->data});
  forward_upsample_zeros_is8os8(&Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(116, 1, { Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_transpose_6_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output_array.data = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_conv2d_transpose_6_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 476888);
  Student_Unet_1_conv2d_transpose_6_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 476888);
  Student_Unet_1_conv2d_transpose_6_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 478936);
  Student_Unet_1_conv2d_transpose_6_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 478936);
  Student_Unet_1_conv2d_transpose_6_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_transpose_6_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_transpose_6_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_transpose_6_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 56288);
  _STAI_NETWORK_EVENT_NODE_START_CB(116, 1, { Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_transpose_6_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(116, 1, { Student_Unet_1_conv2d_transpose_6_1_Relu0_output.data->data});
}
void forward_lite_concat_Student_Unet_1_concatenate_6_1_concat0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_transpose_6_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_transpose_6_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 56288);
  Student_Unet_1_conv2d_22_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 39904);
  Student_Unet_1_conv2d_22_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 39904);
  Student_Unet_1_concatenate_6_1_concat0_output_array.data = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_concatenate_6_1_concat0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 72672);
  _STAI_NETWORK_EVENT_NODE_START_CB(119, 2, { Student_Unet_1_conv2d_transpose_6_1_Relu0_output.data->data,Student_Unet_1_conv2d_22_1_Relu0_output.data->data});
  forward_concat(&Student_Unet_1_concatenate_6_1_concat0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(119, 1, { Student_Unet_1_concatenate_6_1_concat0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_33_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_concatenate_6_1_concat0_output_array.data = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_concatenate_6_1_concat0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 72672);
  Student_Unet_1_conv2d_33_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 479000);
  Student_Unet_1_conv2d_33_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 479000);
  Student_Unet_1_conv2d_33_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 483608);
  Student_Unet_1_conv2d_33_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 483608);
  Student_Unet_1_conv2d_33_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_33_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_33_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_33_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  _STAI_NETWORK_EVENT_NODE_START_CB(122, 1, { Student_Unet_1_concatenate_6_1_concat0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_33_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(122, 1, { Student_Unet_1_conv2d_33_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_34_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_33_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_33_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_34_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 483672);
  Student_Unet_1_conv2d_34_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 483672);
  Student_Unet_1_conv2d_34_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 485976);
  Student_Unet_1_conv2d_34_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 485976);
  Student_Unet_1_conv2d_34_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_34_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_34_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 49152);
  Student_Unet_1_conv2d_34_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 49152);
  _STAI_NETWORK_EVENT_NODE_START_CB(125, 1, { Student_Unet_1_conv2d_33_1_Relu0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_34_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(125, 1, { Student_Unet_1_conv2d_34_1_Relu0_output.data->data});
}
void forward_lite_upsample_zeros_is8os8_Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_34_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 49152);
  Student_Unet_1_conv2d_34_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 49152);
  Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output_array.data = AI_PTR(net_ctx->_activations[1] + 65536);
  Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 65536);
  _STAI_NETWORK_EVENT_NODE_START_CB(128, 1, { Student_Unet_1_conv2d_34_1_Relu0_output.data->data});
  forward_upsample_zeros_is8os8(&Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(128, 1, { Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_transpose_7_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output_array.data = AI_PTR(net_ctx->_activations[1] + 65536);
  Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 65536);
  Student_Unet_1_conv2d_transpose_7_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 486040);
  Student_Unet_1_conv2d_transpose_7_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 486040);
  Student_Unet_1_conv2d_transpose_7_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 486552);
  Student_Unet_1_conv2d_transpose_7_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 486552);
  Student_Unet_1_conv2d_transpose_7_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_transpose_7_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_transpose_7_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_transpose_7_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  _STAI_NETWORK_EVENT_NODE_START_CB(128, 1, { Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_transpose_7_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(128, 1, { Student_Unet_1_conv2d_transpose_7_1_Relu0_output.data->data});
}
void forward_lite_concat_Student_Unet_1_concatenate_7_1_concat0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_transpose_7_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_transpose_7_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_20_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  Student_Unet_1_conv2d_20_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  Student_Unet_1_concatenate_7_1_concat0_output_array.data = AI_PTR(net_ctx->_activations[1] + 65536);
  Student_Unet_1_concatenate_7_1_concat0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 65536);
  _STAI_NETWORK_EVENT_NODE_START_CB(131, 2, { Student_Unet_1_conv2d_transpose_7_1_Relu0_output.data->data,Student_Unet_1_conv2d_20_1_Relu0_output.data->data});
  forward_concat(&Student_Unet_1_concatenate_7_1_concat0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(131, 1, { Student_Unet_1_concatenate_7_1_concat0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_35_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_concatenate_7_1_concat0_output_array.data = AI_PTR(net_ctx->_activations[1] + 65536);
  Student_Unet_1_concatenate_7_1_concat0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 65536);
  Student_Unet_1_conv2d_35_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 486584);
  Student_Unet_1_conv2d_35_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 486584);
  Student_Unet_1_conv2d_35_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 487736);
  Student_Unet_1_conv2d_35_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 487736);
  Student_Unet_1_conv2d_35_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_35_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_35_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  Student_Unet_1_conv2d_35_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(134, 1, { Student_Unet_1_concatenate_7_1_concat0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_35_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(134, 1, { Student_Unet_1_conv2d_35_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_36_1_Relu0(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_35_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  Student_Unet_1_conv2d_35_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  Student_Unet_1_conv2d_36_1_Relu0_weights_array.data = AI_PTR(net_ctx->_weights[0] + 487768);
  Student_Unet_1_conv2d_36_1_Relu0_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 487768);
  Student_Unet_1_conv2d_36_1_Relu0_bias_array.data = AI_PTR(net_ctx->_weights[0] + 488344);
  Student_Unet_1_conv2d_36_1_Relu0_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 488344);
  Student_Unet_1_conv2d_36_1_Relu0_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_36_1_Relu0_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Student_Unet_1_conv2d_36_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_36_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  _STAI_NETWORK_EVENT_NODE_START_CB(137, 1, { Student_Unet_1_conv2d_35_1_Relu0_output.data->data});
  forward_conv2d_integer_SSSA(&Student_Unet_1_conv2d_36_1_Relu0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(137, 1, { Student_Unet_1_conv2d_36_1_Relu0_output.data->data});
}
void forward_lite_conv2d_integer_SSSA_Conv__2060(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_36_1_Relu0_output_array.data = AI_PTR(net_ctx->_activations[1] + 32768);
  Student_Unet_1_conv2d_36_1_Relu0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 32768);
  Conv__2060_weights_array.data = AI_PTR(net_ctx->_weights[0] + 488376);
  Conv__2060_weights_array.data_start = AI_PTR(net_ctx->_weights[0] + 488376);
  Conv__2060_bias_array.data = AI_PTR(net_ctx->_weights[0] + 488384);
  Conv__2060_bias_array.data_start = AI_PTR(net_ctx->_weights[0] + 488384);
  Conv__2060_scratch0_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  Conv__2060_scratch0_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  Conv__2060_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  Conv__2060_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(140, 1, { Student_Unet_1_conv2d_36_1_Relu0_output.data->data});
  forward_conv2d_integer_SSSA(&Conv__2060_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(140, 1, { Conv__2060_output.data->data});
}
void forward_lite_nl_integer_Student_Unet_1_conv2d_37_1_Sigmoid0(_stai_network_context* net_ctx)
{
  Conv__2060_output_array.data = AI_PTR(net_ctx->_activations[1] + 0);
  Conv__2060_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 0);
  Student_Unet_1_conv2d_37_1_Sigmoid0_output_array.data = AI_PTR(net_ctx->_activations[1] + 4096);
  Student_Unet_1_conv2d_37_1_Sigmoid0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 4096);
  _STAI_NETWORK_EVENT_NODE_START_CB(143, 1, { Conv__2060_output.data->data});
  forward_nl_integer(&Student_Unet_1_conv2d_37_1_Sigmoid0_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(143, 1, { Student_Unet_1_conv2d_37_1_Sigmoid0_output.data->data});
}
void forward_lite_transpose_output_QuantizeLinear_Input_to_chlast(_stai_network_context* net_ctx)
{
  Student_Unet_1_conv2d_37_1_Sigmoid0_output_array.data = AI_PTR(net_ctx->_activations[1] + 4096);
  Student_Unet_1_conv2d_37_1_Sigmoid0_output_array.data_start = AI_PTR(net_ctx->_activations[1] + 4096);
  output_QuantizeLinear_Input_to_chlast_output_array.data = AI_PTR(net_ctx->_outputs[0] + 0);
  output_QuantizeLinear_Input_to_chlast_output_array.data_start = AI_PTR(net_ctx->_outputs[0] + 0);
  _STAI_NETWORK_EVENT_NODE_START_CB(146, 1, { Student_Unet_1_conv2d_37_1_Sigmoid0_output.data->data});
  forward_transpose(&output_QuantizeLinear_Input_to_chlast_layer);
  _STAI_NETWORK_EVENT_NODE_STOP_CB(146, 1, { output_QuantizeLinear_Input_to_chlast_output.data->data});
}

/*****************************************************************************/




static const ai_u16 Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_in_0_shape_w_const_u16 = 64;
static const ai_u16 Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_in_0_shape_h_const_u16 = 64;
static const ai_u16 Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_in_0_shape_ch_const_u16 = 8;
static const ai_u16 Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_l_pool_size_1_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_l_pool_size_0_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_l_legacy_pool_pad_1_const_u16 = 0;
static const ai_u16 Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_l_legacy_pool_pad_0_const_u16 = 0;
static const ai_u16 Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_l_pool_stride_1_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_l_pool_stride_0_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_out_0_shape_w_const_u16 = 32;
static const ai_u16 Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_out_0_shape_h_const_u16 = 32;
static const ai_i8 Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_out_0_fmt_zero_const_s8 = -128;



static const ai_u16 Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_in_0_shape_h_const_u16 = 32;
static const ai_u16 Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_l_pool_size_1_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_l_pool_size_0_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_l_legacy_pool_pad_1_const_u16 = 0;
static const ai_u16 Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_l_legacy_pool_pad_0_const_u16 = 0;
static const ai_u16 Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_l_pool_stride_1_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_l_pool_stride_0_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_out_0_shape_w_const_u16 = 16;
static const ai_u16 Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_out_0_shape_h_const_u16 = 16;
static const ai_i8 Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_out_0_fmt_zero_const_s8 = -128;



static const ai_u16 Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_in_0_shape_h_const_u16 = 16;
static const ai_u16 Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_l_pool_size_1_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_l_pool_size_0_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_l_legacy_pool_pad_1_const_u16 = 0;
static const ai_u16 Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_l_legacy_pool_pad_0_const_u16 = 0;
static const ai_u16 Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_l_pool_stride_1_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_l_pool_stride_0_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_out_0_shape_w_const_u16 = 8;
static const ai_u16 Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_out_0_shape_h_const_u16 = 8;
static const ai_i8 Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_out_0_fmt_zero_const_s8 = -128;



static const ai_u16 Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_in_0_shape_ch_const_u16 = 64;
static const ai_u16 Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_l_pool_size_1_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_l_pool_size_0_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_l_legacy_pool_pad_1_const_u16 = 0;
static const ai_u16 Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_l_legacy_pool_pad_0_const_u16 = 0;
static const ai_u16 Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_l_pool_stride_1_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_l_pool_stride_0_const_u16 = 2;
static const ai_u16 Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_out_0_shape_w_const_u16 = 4;
static const ai_u16 Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_out_0_shape_h_const_u16 = 4;
static const ai_i8 Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_out_0_fmt_zero_const_s8 = -128;

























STAI_API_ENTRY
stai_return_code stai_network_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_19_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_19_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_19_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_20_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_20_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_20_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_max_pooling2d_4_1_MaxPool2d0 */
  {
      const ai_i8* Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 0);
    ai_i8* Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 32768);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(56, 1, {(stai_ptr) Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_in_0_ptr_const_s8});
    
  forward_lite_maxpool_is8os8_scalepos(Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_in_0_ptr_const_s8, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_out_0_ptr_s8, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_in_0_shape_w_const_u16, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_in_0_shape_h_const_u16, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_in_0_shape_ch_const_u16, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_l_pool_size_1_const_u16, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_l_pool_size_0_const_u16, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_l_legacy_pool_pad_1_const_u16, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_l_legacy_pool_pad_0_const_u16, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_l_pool_stride_1_const_u16, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_l_pool_stride_0_const_u16, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_out_0_shape_w_const_u16, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_out_0_shape_h_const_u16, 1.0f, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_in_0_fmt_zero_const_s8, Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(56, 1, {(stai_ptr) Student_Unet_1_max_pooling2d_4_1_MaxPool2d0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_max_pooling2d_4_1_MaxPool2d0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_21_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_21_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_21_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_22_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_22_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_22_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_max_pooling2d_5_1_MaxPool2d0 */
  {
      const ai_i8* Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 39904);
    ai_i8* Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 32768);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(65, 1, {(stai_ptr) Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_in_0_ptr_const_s8});
    
  forward_lite_maxpool_is8os8_scalepos(Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_in_0_ptr_const_s8, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_out_0_ptr_s8, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_in_0_shape_w_const_u16, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_in_0_shape_h_const_u16, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_in_0_shape_ch_const_u16, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_l_pool_size_1_const_u16, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_l_pool_size_0_const_u16, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_l_legacy_pool_pad_1_const_u16, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_l_legacy_pool_pad_0_const_u16, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_l_pool_stride_1_const_u16, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_l_pool_stride_0_const_u16, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_out_0_shape_w_const_u16, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_out_0_shape_h_const_u16, 1.0f, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_in_0_fmt_zero_const_s8, Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(65, 1, {(stai_ptr) Student_Unet_1_max_pooling2d_5_1_MaxPool2d0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_max_pooling2d_5_1_MaxPool2d0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_23_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_23_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_23_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_24_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_24_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_24_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_max_pooling2d_6_1_MaxPool2d0 */
  {
      const ai_i8* Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 64480);
    ai_i8* Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 32768);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(74, 1, {(stai_ptr) Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_in_0_ptr_const_s8});
    
  forward_lite_maxpool_is8os8_scalepos(Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_in_0_ptr_const_s8, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_out_0_ptr_s8, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_in_0_shape_w_const_u16, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_in_0_shape_h_const_u16, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_in_0_shape_ch_const_u16, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_l_pool_size_1_const_u16, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_l_pool_size_0_const_u16, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_l_legacy_pool_pad_1_const_u16, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_l_legacy_pool_pad_0_const_u16, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_l_pool_stride_1_const_u16, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_l_pool_stride_0_const_u16, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_out_0_shape_w_const_u16, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_out_0_shape_h_const_u16, 1.0f, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_in_0_fmt_zero_const_s8, Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(74, 1, {(stai_ptr) Student_Unet_1_max_pooling2d_6_1_MaxPool2d0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_max_pooling2d_6_1_MaxPool2d0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_25_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_25_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_25_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_26_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_26_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_26_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_max_pooling2d_7_1_MaxPool2d0 */
  {
      const ai_i8* Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[1] + 56288);
    ai_i8* Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[1] + 32768);
  
  _STAI_NETWORK_EVENT_NODE_START_CB(83, 1, {(stai_ptr) Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_in_0_ptr_const_s8});
    
  forward_lite_maxpool_is8os8_scalepos(Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_in_0_ptr_const_s8, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_out_0_ptr_s8, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_in_0_shape_w_const_u16, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_in_0_shape_h_const_u16, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_in_0_shape_ch_const_u16, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_l_pool_size_1_const_u16, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_l_pool_size_0_const_u16, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_l_legacy_pool_pad_1_const_u16, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_l_legacy_pool_pad_0_const_u16, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_l_pool_stride_1_const_u16, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_l_pool_stride_0_const_u16, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_out_0_shape_w_const_u16, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_out_0_shape_h_const_u16, 1.0f, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_in_0_fmt_zero_const_s8, Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_out_0_fmt_zero_const_s8);
    
  _STAI_NETWORK_EVENT_NODE_STOP_CB(83, 1, {(stai_ptr) Student_Unet_1_max_pooling2d_7_1_MaxPool2d0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_max_pooling2d_7_1_MaxPool2d0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_27_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_27_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_27_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_28_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_28_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_28_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample */
  {
    
  forward_lite_upsample_zeros_is8os8_Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_transpose_4_1_Relu0_upsample */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_transpose_4_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_transpose_4_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_transpose_4_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_concatenate_4_1_concat0 */
  {
    
  forward_lite_concat_Student_Unet_1_concatenate_4_1_concat0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_concatenate_4_1_concat0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_29_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_29_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_29_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_30_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_30_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_30_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample */
  {
    
  forward_lite_upsample_zeros_is8os8_Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_transpose_5_1_Relu0_upsample */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_transpose_5_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_transpose_5_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_transpose_5_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_concatenate_5_1_concat0 */
  {
    
  forward_lite_concat_Student_Unet_1_concatenate_5_1_concat0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_concatenate_5_1_concat0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_31_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_31_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_31_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_32_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_32_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_32_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample */
  {
    
  forward_lite_upsample_zeros_is8os8_Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_transpose_6_1_Relu0_upsample */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_transpose_6_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_transpose_6_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_transpose_6_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_concatenate_6_1_concat0 */
  {
    
  forward_lite_concat_Student_Unet_1_concatenate_6_1_concat0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_concatenate_6_1_concat0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_33_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_33_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_33_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_34_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_34_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_34_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample */
  {
    
  forward_lite_upsample_zeros_is8os8_Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_transpose_7_1_Relu0_upsample */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_transpose_7_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_transpose_7_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_transpose_7_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_concatenate_7_1_concat0 */
  {
    
  forward_lite_concat_Student_Unet_1_concatenate_7_1_concat0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_concatenate_7_1_concat0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_35_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_35_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_35_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_36_1_Relu0 */
  {
    
  forward_lite_conv2d_integer_SSSA_Student_Unet_1_conv2d_36_1_Relu0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_36_1_Relu0 */
  /* LITE_KERNEL_SECTION BEGIN Conv__2060 */
  {
    
  forward_lite_conv2d_integer_SSSA_Conv__2060(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Conv__2060 */
  /* LITE_KERNEL_SECTION BEGIN Student_Unet_1_conv2d_37_1_Sigmoid0 */
  {
    
  forward_lite_nl_integer_Student_Unet_1_conv2d_37_1_Sigmoid0(net_ctx);
  }
  /* LITE_KERNEL_SECTION END Student_Unet_1_conv2d_37_1_Sigmoid0 */
  /* LITE_KERNEL_SECTION BEGIN output_QuantizeLinear_Input_to_chlast */
  {
    
  forward_lite_transpose_output_QuantizeLinear_Input_to_chlast(net_ctx);
  }
  /* LITE_KERNEL_SECTION END output_QuantizeLinear_Input_to_chlast */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_network_get_context_size()
{
  return (stai_size)STAI_NETWORK_CONTEXT_SIZE;
}

#if defined(HAVE_NETWORK_INFO)
STAI_API_ENTRY
stai_return_code stai_network_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_network_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_network_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_NETWORK_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_NETWORK_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_NETWORK_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_NETWORK_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_NETWORK_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_NETWORK_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_NETWORK_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_NETWORK_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_NETWORK_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_network_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_NETWORK_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_network_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_NETWORK_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_NETWORK_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[1] + 22560;

  net_ctx->_outputs[0] = activations[1] + 0;
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_NETWORK_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_NETWORK_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_NETWORK_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_NETWORK_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_NETWORK_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_NETWORK_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_NETWORK_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_NETWORK_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_network_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_network_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_network_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_network_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_NETWORK_EVENT_NODE_START_CB
#undef _STAI_NETWORK_EVENT_NODE_STOP_CB
#undef _STAI_NETWORK_MODEL_SIGNATURE
#undef _STAI_NETWORK_DATETIME
#undef _STAI_NETWORK_COMPILE_DATETIME

