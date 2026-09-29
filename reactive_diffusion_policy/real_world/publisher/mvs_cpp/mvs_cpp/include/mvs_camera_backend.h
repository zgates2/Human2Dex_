#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MVSOpenConfig {
  uint32_t roi_width;
  uint32_t roi_height;
  uint32_t offset_x;
  uint32_t offset_y;
  uint32_t crop_x;
  uint32_t crop_y;
  uint32_t crop_width;
  uint32_t crop_height;
  uint32_t output_width;
  uint32_t output_height;
  uint32_t image_node_num;
  uint32_t frame_pool_size;
  int32_t rotate_180;
  int32_t acquisition_frame_rate_enable;
  float acquisition_frame_rate_fps;
  int32_t exposure_auto_mode;
  float exposure_time_us;
  int32_t gain_auto_mode;
  int32_t balance_white_auto_mode;
  int32_t black_level_enable;
  int32_t black_level;
  int32_t brightness;
} MVSOpenConfig;

typedef struct MVSFrameInfo {
  uint32_t width;
  uint32_t height;
  uint64_t frame_num;
  uint64_t timestamp_ns;
} MVSFrameInfo;

typedef struct MVSFrameView {
  uint32_t width;
  uint32_t height;
  uint32_t stride_bytes;
  uint32_t channels;
  uint32_t slot_index;
  uint64_t frame_num;
  uint64_t timestamp_ns;
  uint8_t* data;
  size_t data_len;
} MVSFrameView;

typedef struct MVSCameraRuntimeConfig {
  uint32_t width;
  uint32_t height;
  uint32_t offset_x;
  uint32_t offset_y;
  int32_t acquisition_frame_rate_enable;
  float acquisition_frame_rate_fps;
  int32_t exposure_auto_mode;
  float exposure_time_us;
  int32_t gain_auto_mode;
  int32_t balance_white_auto_mode;
  int32_t black_level_enable;
  int32_t black_level;
  int32_t brightness;
} MVSCameraRuntimeConfig;

int mvs_enumerate_serials(
    char* serials_buf,
    size_t serials_buf_len,
    size_t* required_len,
    char* err_buf,
    size_t err_buf_len);

int mvs_open_camera(
    const char* serial,
    const MVSOpenConfig* config,
    void** out_handle,
    char* err_buf,
    size_t err_buf_len);

int mvs_close_camera(
    void* handle,
    int32_t* out_handle_destroyed,
    char* err_buf,
    size_t err_buf_len);

int mvs_destroy_camera(
    void* handle,
    char* err_buf,
    size_t err_buf_len);

int mvs_get_frame_shape(
    void* handle,
    uint32_t* width,
    uint32_t* height,
    char* err_buf,
    size_t err_buf_len);

int mvs_get_runtime_config(
    void* handle,
    MVSCameraRuntimeConfig* out_config,
    char* err_buf,
    size_t err_buf_len);

int mvs_grab_frame_rgb(
    void* handle,
    uint32_t timeout_ms,
    uint8_t* out_buffer,
    size_t out_buffer_len,
    MVSFrameInfo* out_frame_info,
    char* err_buf,
    size_t err_buf_len);

int mvs_acquire_frame_rgb_view(
    void* handle,
    uint32_t timeout_ms,
    MVSFrameView* out_frame_view,
    char* err_buf,
    size_t err_buf_len);

int mvs_release_frame(
    void* handle,
    uint32_t slot_index,
    char* err_buf,
    size_t err_buf_len);

#ifdef __cplusplus
}
#endif
