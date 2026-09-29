#include <MvCameraControl.h>
#include "../include/mvs_camera_backend.h"

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kBackendOk = 0;
constexpr int kBackendBufferTooSmall = 1;

struct FrameSlot {
  std::vector<uint8_t> data;
  bool in_use = false;
};

struct CameraHandle {
  void* handle = nullptr;
  unsigned int width = 0;
  unsigned int height = 0;
  unsigned int sensor_width = 0;
  unsigned int sensor_height = 0;
  unsigned int crop_x = 0;
  unsigned int crop_y = 0;
  unsigned int crop_width = 0;
  unsigned int crop_height = 0;
  bool rotate_180 = false;
  bool processing_required = false;
  std::mutex grab_mutex;
  std::mutex frame_mutex;
  std::vector<FrameSlot> frame_slots;
  std::vector<uint8_t> rgb_workspace;
};

std::mutex g_sdk_mutex;
bool g_sdk_initialized = false;
size_t g_sdk_session_count = 0;
size_t g_sdk_open_camera_count = 0;

void initialize_frame_slots(CameraHandle* camera, size_t pool_size);
void set_error(char* err_buf, size_t err_buf_len, const std::string& message);
std::string format_mv_error(const char* action, int ret_code);

void maybe_finalize_sdk_locked() {
  if (!g_sdk_initialized) {
    return;
  }
  if (g_sdk_session_count != 0 || g_sdk_open_camera_count != 0) {
    return;
  }
  MV_CC_Finalize();
  g_sdk_initialized = false;
}

int acquire_sdk_session(char* err_buf, size_t err_buf_len) {
  std::lock_guard<std::mutex> lock(g_sdk_mutex);
  if (!g_sdk_initialized) {
    const int ret = MV_CC_Initialize();
    if (ret != MV_OK) {
      set_error(err_buf, err_buf_len, format_mv_error("MV_CC_Initialize", ret));
      return ret;
    }
    g_sdk_initialized = true;
  }
  ++g_sdk_session_count;
  return MV_OK;
}

void release_sdk_session() {
  std::lock_guard<std::mutex> lock(g_sdk_mutex);
  if (g_sdk_session_count > 0) {
    --g_sdk_session_count;
  }
  maybe_finalize_sdk_locked();
}

void register_open_camera() {
  std::lock_guard<std::mutex> lock(g_sdk_mutex);
  ++g_sdk_open_camera_count;
}

void unregister_open_camera() {
  std::lock_guard<std::mutex> lock(g_sdk_mutex);
  if (g_sdk_open_camera_count > 0) {
    --g_sdk_open_camera_count;
  }
  maybe_finalize_sdk_locked();
}

class SdkSessionGuard {
 public:
  int acquire(char* err_buf, size_t err_buf_len) {
    const int ret = acquire_sdk_session(err_buf, err_buf_len);
    active_ = (ret == MV_OK);
    return ret;
  }

  ~SdkSessionGuard() {
    if (active_) {
      release_sdk_session();
    }
  }

 private:
  bool active_ = false;
};

void set_error(char* err_buf, size_t err_buf_len, const std::string& message) {
  if (err_buf == nullptr || err_buf_len == 0) {
    return;
  }
  std::snprintf(err_buf, err_buf_len, "%s", message.c_str());
}

std::string format_mv_error(const char* action, int ret_code) {
  std::ostringstream oss;
  oss << action << " failed with code 0x" << std::hex
      << static_cast<unsigned int>(ret_code);
  return oss.str();
}

std::string serial_from_chars(const unsigned char* chars, size_t max_len) {
  std::string serial;
  serial.reserve(max_len);
  for (size_t i = 0; i < max_len; ++i) {
    if (chars[i] == 0) {
      break;
    }
    serial.push_back(static_cast<char>(chars[i]));
  }
  return serial;
}

std::string extract_device_serial(const MV_CC_DEVICE_INFO& device_info) {
  switch (device_info.nTLayerType) {
    case MV_GIGE_DEVICE:
      return serial_from_chars(
          device_info.SpecialInfo.stGigEInfo.chSerialNumber,
          sizeof(device_info.SpecialInfo.stGigEInfo.chSerialNumber));
    case MV_USB_DEVICE:
      return serial_from_chars(
          device_info.SpecialInfo.stUsb3VInfo.chSerialNumber,
          sizeof(device_info.SpecialInfo.stUsb3VInfo.chSerialNumber));
    case MV_GENTL_CAMERALINK_DEVICE:
      return serial_from_chars(
          device_info.SpecialInfo.stCMLInfo.chSerialNumber,
          sizeof(device_info.SpecialInfo.stCMLInfo.chSerialNumber));
    case MV_GENTL_CXP_DEVICE:
      return serial_from_chars(
          device_info.SpecialInfo.stCXPInfo.chSerialNumber,
          sizeof(device_info.SpecialInfo.stCXPInfo.chSerialNumber));
    case MV_GENTL_XOF_DEVICE:
      return serial_from_chars(
          device_info.SpecialInfo.stXoFInfo.chSerialNumber,
          sizeof(device_info.SpecialInfo.stXoFInfo.chSerialNumber));
    default:
      return {};
  }
}

int maybe_set_enum_value(
    void* handle,
    const char* key,
    unsigned int value,
    bool required,
    char* err_buf,
    size_t err_buf_len) {
  const int ret = MV_CC_SetEnumValue(handle, key, value);
  if (ret != MV_OK && required) {
    set_error(err_buf, err_buf_len, format_mv_error(key, ret));
    return ret;
  }
  return MV_OK;
}

int maybe_set_int_value(
    void* handle,
    const char* key,
    unsigned int value,
    bool required,
    char* err_buf,
    size_t err_buf_len) {
  const int ret = MV_CC_SetIntValue(handle, key, value);
  if (ret != MV_OK && required) {
    set_error(err_buf, err_buf_len, format_mv_error(key, ret));
    return ret;
  }
  return MV_OK;
}

int maybe_set_float_value(
    void* handle,
    const char* key,
    float value,
    bool required,
    char* err_buf,
    size_t err_buf_len) {
  const int ret = MV_CC_SetFloatValue(handle, key, value);
  if (ret != MV_OK && required) {
    set_error(err_buf, err_buf_len, format_mv_error(key, ret));
    return ret;
  }
  return MV_OK;
}

int maybe_set_bool_value(
    void* handle,
    const char* key,
    bool value,
    bool required,
    char* err_buf,
    size_t err_buf_len) {
  const int ret = MV_CC_SetBoolValue(handle, key, value);
  if (ret != MV_OK && required) {
    set_error(err_buf, err_buf_len, format_mv_error(key, ret));
    return ret;
  }
  return MV_OK;
}

int query_int_value(void* handle, const char* key, unsigned int* out_value) {
  MVCC_INTVALUE int_value = {};
  const int ret = MV_CC_GetIntValue(handle, key, &int_value);
  if (ret != MV_OK) {
    return ret;
  }
  *out_value = int_value.nCurValue;
  return MV_OK;
}

int query_enum_value(void* handle, const char* key, unsigned int* out_value) {
  MVCC_ENUMVALUE enum_value = {};
  const int ret = MV_CC_GetEnumValue(handle, key, &enum_value);
  if (ret != MV_OK) {
    return ret;
  }
  *out_value = enum_value.nCurValue;
  return MV_OK;
}

int query_float_value(void* handle, const char* key, float* out_value) {
  MVCC_FLOATVALUE float_value = {};
  const int ret = MV_CC_GetFloatValue(handle, key, &float_value);
  if (ret != MV_OK) {
    return ret;
  }
  *out_value = float_value.fCurValue;
  return MV_OK;
}

int query_bool_value(void* handle, const char* key, bool* out_value) {
  bool value = false;
  const int ret = MV_CC_GetBoolValue(handle, key, &value);
  if (ret != MV_OK) {
    return ret;
  }
  *out_value = value;
  return MV_OK;
}

int query_int_info(void* handle, const char* key, MVCC_INTVALUE* out_value) {
  MVCC_INTVALUE int_value = {};
  const int ret = MV_CC_GetIntValue(handle, key, &int_value);
  if (ret != MV_OK) {
    return ret;
  }
  *out_value = int_value;
  return MV_OK;
}

std::string format_mismatch_error(
    const char* key,
    const std::string& expected,
    const std::string& actual) {
  std::ostringstream oss;
  oss << key << " verification mismatch (expected=" << expected
      << ", actual=" << actual << ")";
  return oss.str();
}

int verify_enum_value(
    void* handle,
    const char* key,
    unsigned int expected,
    char* err_buf,
    size_t err_buf_len) {
  unsigned int actual = 0;
  const int ret = query_enum_value(handle, key, &actual);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error(key, ret));
    return ret;
  }
  if (actual != expected) {
    set_error(
        err_buf,
        err_buf_len,
        format_mismatch_error(key, std::to_string(expected), std::to_string(actual)));
    return MV_E_PARAMETER;
  }
  return MV_OK;
}

int verify_int_value(
    void* handle,
    const char* key,
    unsigned int expected,
    char* err_buf,
    size_t err_buf_len) {
  unsigned int actual = 0;
  const int ret = query_int_value(handle, key, &actual);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error(key, ret));
    return ret;
  }
  if (actual != expected) {
    set_error(
        err_buf,
        err_buf_len,
        format_mismatch_error(key, std::to_string(expected), std::to_string(actual)));
    return MV_E_PARAMETER;
  }
  return MV_OK;
}

int verify_bool_value(
    void* handle,
    const char* key,
    bool expected,
    char* err_buf,
    size_t err_buf_len) {
  bool actual = false;
  const int ret = query_bool_value(handle, key, &actual);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error(key, ret));
    return ret;
  }
  if (actual != expected) {
    set_error(
        err_buf,
        err_buf_len,
        format_mismatch_error(key, expected ? "1" : "0", actual ? "1" : "0"));
    return MV_E_PARAMETER;
  }
  return MV_OK;
}

int verify_float_value(
    void* handle,
    const char* key,
    float expected,
    float tolerance,
    char* err_buf,
    size_t err_buf_len) {
  float actual = 0.0f;
  const int ret = query_float_value(handle, key, &actual);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error(key, ret));
    return ret;
  }
  if (std::fabs(actual - expected) > tolerance) {
    std::ostringstream expected_oss;
    expected_oss << expected;
    std::ostringstream actual_oss;
    actual_oss << actual;
    set_error(
        err_buf,
        err_buf_len,
        format_mismatch_error(key, expected_oss.str(), actual_oss.str()));
    return MV_E_PARAMETER;
  }
  return MV_OK;
}

unsigned long long now_timestamp_ns() {
  using clock = std::chrono::system_clock;
  return static_cast<unsigned long long>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          clock::now().time_since_epoch())
          .count());
}

void rotate_rgb_image_180(uint8_t* data, size_t width, size_t height) {
  if (data == nullptr || width == 0 || height == 0) {
    return;
  }

  const size_t pixels = width * height;
  for (size_t i = 0; i < pixels / 2; ++i) {
    const size_t j = pixels - 1 - i;
    const size_t lhs = i * 3;
    const size_t rhs = j * 3;
    std::swap(data[lhs + 0], data[rhs + 0]);
    std::swap(data[lhs + 1], data[rhs + 1]);
    std::swap(data[lhs + 2], data[rhs + 2]);
  }
}

size_t rgb_byte_size(uint32_t width, uint32_t height) {
  return static_cast<size_t>(width) * static_cast<size_t>(height) * 3U;
}

void copy_cropped_rgb(
    const uint8_t* src,
    uint32_t src_width,
    uint32_t crop_x,
    uint32_t crop_y,
    uint32_t crop_width,
    uint32_t crop_height,
    uint8_t* dst) {
  const size_t src_stride = static_cast<size_t>(src_width) * 3U;
  const size_t dst_stride = static_cast<size_t>(crop_width) * 3U;
  for (uint32_t row = 0; row < crop_height; ++row) {
    const uint8_t* src_row =
        src + (static_cast<size_t>(crop_y + row) * src_stride) +
        (static_cast<size_t>(crop_x) * 3U);
    uint8_t* dst_row = dst + (static_cast<size_t>(row) * dst_stride);
    std::memcpy(dst_row, src_row, dst_stride);
  }
}

void crop_resize_rgb_bilinear(
    const uint8_t* src,
    uint32_t src_width,
    uint32_t src_height,
    uint32_t crop_x,
    uint32_t crop_y,
    uint32_t crop_width,
    uint32_t crop_height,
    uint8_t* dst,
    uint32_t dst_width,
    uint32_t dst_height) {
  if (dst_width == 0 || dst_height == 0) {
    return;
  }
  if (crop_width == 0 || crop_height == 0) {
    return;
  }
  if (dst_width == crop_width && dst_height == crop_height) {
    copy_cropped_rgb(src, src_width, crop_x, crop_y, crop_width, crop_height, dst);
    return;
  }

  const float scale_x = static_cast<float>(crop_width) / static_cast<float>(dst_width);
  const float scale_y = static_cast<float>(crop_height) / static_cast<float>(dst_height);

  for (uint32_t dy = 0; dy < dst_height; ++dy) {
    const float src_y =
        static_cast<float>(crop_y) + ((static_cast<float>(dy) + 0.5f) * scale_y) - 0.5f;
    const float clamped_y =
        std::max(static_cast<float>(crop_y),
                 std::min(src_y, static_cast<float>(crop_y + crop_height - 1)));
    const uint32_t y0 = static_cast<uint32_t>(std::floor(clamped_y));
    const uint32_t y1 = std::min(y0 + 1, crop_y + crop_height - 1);
    const float wy = clamped_y - static_cast<float>(y0);
    const float wy0 = 1.0f - wy;

    for (uint32_t dx = 0; dx < dst_width; ++dx) {
      const float src_x =
          static_cast<float>(crop_x) + ((static_cast<float>(dx) + 0.5f) * scale_x) - 0.5f;
      const float clamped_x =
          std::max(static_cast<float>(crop_x),
                   std::min(src_x, static_cast<float>(crop_x + crop_width - 1)));
      const uint32_t x0 = static_cast<uint32_t>(std::floor(clamped_x));
      const uint32_t x1 = std::min(x0 + 1, crop_x + crop_width - 1);
      const float wx = clamped_x - static_cast<float>(x0);
      const float wx0 = 1.0f - wx;

      const size_t idx00 = (static_cast<size_t>(y0) * src_width + x0) * 3U;
      const size_t idx01 = (static_cast<size_t>(y0) * src_width + x1) * 3U;
      const size_t idx10 = (static_cast<size_t>(y1) * src_width + x0) * 3U;
      const size_t idx11 = (static_cast<size_t>(y1) * src_width + x1) * 3U;
      uint8_t* dst_pixel =
          dst + ((static_cast<size_t>(dy) * static_cast<size_t>(dst_width) + dx) * 3U);

      for (size_t channel = 0; channel < 3; ++channel) {
        const float top =
            (static_cast<float>(src[idx00 + channel]) * wx0) +
            (static_cast<float>(src[idx01 + channel]) * wx);
        const float bottom =
            (static_cast<float>(src[idx10 + channel]) * wx0) +
            (static_cast<float>(src[idx11 + channel]) * wx);
        const float value = (top * wy0) + (bottom * wy);
        dst_pixel[channel] = static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 255.0f)));
      }
    }
  }
}

int configure_frame_processing(
    CameraHandle* camera,
    const MVSOpenConfig* config,
    char* err_buf,
    size_t err_buf_len) {
  const uint32_t sensor_width = camera->sensor_width;
  const uint32_t sensor_height = camera->sensor_height;
  if (sensor_width == 0 || sensor_height == 0) {
    set_error(err_buf, err_buf_len, "Invalid sensor frame size");
    return MV_E_PARAMETER;
  }

  const uint32_t crop_x = config != nullptr ? config->crop_x : 0U;
  const uint32_t crop_y = config != nullptr ? config->crop_y : 0U;
  const uint32_t crop_width =
      (config != nullptr && config->crop_width > 0) ? config->crop_width : sensor_width;
  const uint32_t crop_height =
      (config != nullptr && config->crop_height > 0) ? config->crop_height : sensor_height;
  if (crop_width == 0 || crop_height == 0) {
    set_error(err_buf, err_buf_len, "Crop size must be positive");
    return MV_E_PARAMETER;
  }
  if (crop_x >= sensor_width || crop_y >= sensor_height ||
      crop_x + crop_width > sensor_width || crop_y + crop_height > sensor_height) {
    set_error(err_buf, err_buf_len, "Crop rectangle exceeds the configured sensor frame");
    return MV_E_PARAMETER;
  }

  const uint32_t output_width =
      (config != nullptr && config->output_width > 0) ? config->output_width : crop_width;
  const uint32_t output_height =
      (config != nullptr && config->output_height > 0) ? config->output_height : crop_height;
  if (output_width == 0 || output_height == 0) {
    set_error(err_buf, err_buf_len, "Output size must be positive");
    return MV_E_PARAMETER;
  }

  camera->crop_x = crop_x;
  camera->crop_y = crop_y;
  camera->crop_width = crop_width;
  camera->crop_height = crop_height;
  camera->width = output_width;
  camera->height = output_height;
  camera->processing_required =
      crop_x != 0 || crop_y != 0 || crop_width != sensor_width ||
      crop_height != sensor_height || output_width != crop_width ||
      output_height != crop_height;
  if (camera->processing_required) {
    camera->rgb_workspace.assign(rgb_byte_size(sensor_width, sensor_height), 0);
  } else {
    camera->rgb_workspace.clear();
  }

  initialize_frame_slots(camera, config != nullptr && config->frame_pool_size > 0
                                     ? config->frame_pool_size
                                     : 4U);
  return MV_OK;
}

int transform_rgb_output(
    CameraHandle* camera,
    const uint8_t* rgb_source,
    uint32_t src_width,
    uint32_t src_height,
    uint8_t* out_buffer,
    size_t out_buffer_len,
    char* err_buf,
    size_t err_buf_len) {
  const size_t output_bytes = rgb_byte_size(camera->width, camera->height);
  if (out_buffer_len < output_bytes) {
    set_error(err_buf, err_buf_len, "RGB output buffer is too small");
    return MV_E_NOENOUGH_BUF;
  }

  if (!camera->processing_required) {
    std::memcpy(out_buffer, rgb_source, output_bytes);
    return MV_OK;
  }

  if (camera->crop_x + camera->crop_width > src_width ||
      camera->crop_y + camera->crop_height > src_height) {
    set_error(err_buf, err_buf_len, "Incoming frame is smaller than the configured crop rectangle");
    return MV_E_PARAMETER;
  }

  crop_resize_rgb_bilinear(
      rgb_source,
      src_width,
      src_height,
      camera->crop_x,
      camera->crop_y,
      camera->crop_width,
      camera->crop_height,
      out_buffer,
      camera->width,
      camera->height);
  return MV_OK;
}

void initialize_frame_slots(CameraHandle* camera, size_t pool_size) {
  const size_t slot_bytes = rgb_byte_size(camera->width, camera->height);
  camera->frame_slots.clear();
  camera->frame_slots.resize(pool_size);
  for (auto& slot : camera->frame_slots) {
    slot.data.resize(slot_bytes);
    slot.in_use = false;
  }
}

int acquire_frame_slot(CameraHandle* camera, uint32_t* out_slot_index, char* err_buf, size_t err_buf_len) {
  std::lock_guard<std::mutex> lock(camera->frame_mutex);
  for (size_t i = 0; i < camera->frame_slots.size(); ++i) {
    if (!camera->frame_slots[i].in_use) {
      camera->frame_slots[i].in_use = true;
      *out_slot_index = static_cast<uint32_t>(i);
      return MV_OK;
    }
  }

  set_error(
      err_buf,
      err_buf_len,
      "No free frame slot available. Increase frame_pool_size or release frames sooner.");
  return MV_E_BUFOVER;
}

int release_frame_slot(CameraHandle* camera, uint32_t slot_index, char* err_buf, size_t err_buf_len) {
  std::lock_guard<std::mutex> lock(camera->frame_mutex);
  if (slot_index >= camera->frame_slots.size()) {
    set_error(err_buf, err_buf_len, "Invalid frame slot index");
    return MV_E_PARAMETER;
  }

  camera->frame_slots[slot_index].in_use = false;
  return MV_OK;
}

int set_and_verify_enum_value(
    void* handle,
    const char* key,
    unsigned int value,
    char* err_buf,
    size_t err_buf_len) {
  const int ret = maybe_set_enum_value(handle, key, value, true, err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }
  return verify_enum_value(handle, key, value, err_buf, err_buf_len);
}

int set_and_verify_int_value(
    void* handle,
    const char* key,
    unsigned int value,
    char* err_buf,
    size_t err_buf_len) {
  const int ret = maybe_set_int_value(handle, key, value, true, err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }
  return verify_int_value(handle, key, value, err_buf, err_buf_len);
}

int set_and_verify_bool_value(
    void* handle,
    const char* key,
    bool value,
    char* err_buf,
    size_t err_buf_len) {
  const int ret = maybe_set_bool_value(handle, key, value, true, err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }
  return verify_bool_value(handle, key, value, err_buf, err_buf_len);
}

int set_and_verify_float_value(
    void* handle,
    const char* key,
    float value,
    float tolerance,
    char* err_buf,
    size_t err_buf_len) {
  const int ret = maybe_set_float_value(handle, key, value, true, err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }
  return verify_float_value(handle, key, value, tolerance, err_buf, err_buf_len);
}

int set_and_verify_frame_rate(
    void* handle,
    const MVSOpenConfig* config,
    char* err_buf,
    size_t err_buf_len) {
  const bool enabled = config->acquisition_frame_rate_enable != 0;
  int ret = set_and_verify_bool_value(
      handle, "AcquisitionFrameRateEnable", enabled, err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }
  if (!enabled) {
    return MV_OK;
  }

  if (config->acquisition_frame_rate_fps <= 0.0f) {
    set_error(
        err_buf,
        err_buf_len,
        "AcquisitionFrameRate must be > 0 when AcquisitionFrameRateEnable is enabled");
    return MV_E_PARAMETER;
  }

  const float tolerance =
      std::max(0.1f, std::fabs(config->acquisition_frame_rate_fps) * 0.01f);
  return set_and_verify_float_value(
      handle,
      "AcquisitionFrameRate",
      config->acquisition_frame_rate_fps,
      tolerance,
      err_buf,
      err_buf_len);
}

int reset_or_apply_roi(
    CameraHandle* camera,
    const MVSOpenConfig* config,
    char* err_buf,
    size_t err_buf_len) {
  const bool apply_roi = config != nullptr;
  if (!apply_roi) {
    return MV_OK;
  }

  int ret = set_and_verify_int_value(camera->handle, "OffsetX", 0, err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }
  ret = set_and_verify_int_value(camera->handle, "OffsetY", 0, err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }

  MVCC_INTVALUE width_info = {};
  ret = query_int_info(camera->handle, "Width", &width_info);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("Width", ret));
    return ret;
  }

  MVCC_INTVALUE height_info = {};
  ret = query_int_info(camera->handle, "Height", &height_info);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("Height", ret));
    return ret;
  }

  const unsigned int target_width =
      config->roi_width > 0 ? config->roi_width : width_info.nMax;
  const unsigned int target_height =
      config->roi_height > 0 ? config->roi_height : height_info.nMax;

  ret = set_and_verify_int_value(camera->handle, "Width", target_width, err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }
  ret = set_and_verify_int_value(camera->handle, "Height", target_height, err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }

  ret = set_and_verify_int_value(camera->handle, "OffsetX", config->offset_x, err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }
  return set_and_verify_int_value(camera->handle, "OffsetY", config->offset_y, err_buf, err_buf_len);
}

int cleanup_open_failure(
    CameraHandle* camera,
    bool close_device,
    char* err_buf,
    size_t err_buf_len) {
  if (camera == nullptr) {
    return MV_OK;
  }

  int first_error = MV_OK;
  auto capture_error = [&](const char* action, int ret_code) {
    if (ret_code == MV_OK || ret_code == MV_E_CALLORDER || first_error != MV_OK) {
      return;
    }
    first_error = ret_code;
    set_error(err_buf, err_buf_len, format_mv_error(action, ret_code));
  };

  if (camera->handle != nullptr) {
    if (close_device) {
      const int close_ret = MV_CC_CloseDevice(camera->handle);
      capture_error("MV_CC_CloseDevice", close_ret);
    }

    const int destroy_ret = MV_CC_DestroyHandle(camera->handle);
    if (destroy_ret == MV_OK) {
      camera->handle = nullptr;
    }
    capture_error("MV_CC_DestroyHandle", destroy_ret);
  }

  delete camera;
  return first_error;
}

int configure_camera(
    CameraHandle* camera,
    const MVSOpenConfig* config,
    char* err_buf,
    size_t err_buf_len) {
  int ret = set_and_verify_enum_value(
      camera->handle, "TriggerMode", MV_TRIGGER_MODE_OFF, err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }

  if (config != nullptr) {
    if (config->image_node_num > 0) {
      ret = MV_CC_SetImageNodeNum(camera->handle, config->image_node_num);
      if (ret != MV_OK) {
        set_error(err_buf, err_buf_len, format_mv_error("MV_CC_SetImageNodeNum", ret));
        return ret;
      }
    }

    ret = reset_or_apply_roi(camera, config, err_buf, err_buf_len);
    if (ret != MV_OK) {
      return ret;
    }

    ret = set_and_verify_frame_rate(camera->handle, config, err_buf, err_buf_len);
    if (ret != MV_OK) {
      return ret;
    }

    ret = set_and_verify_enum_value(
        camera->handle,
        "ExposureAuto",
        static_cast<unsigned int>(config->exposure_auto_mode),
        err_buf,
        err_buf_len);
    if (ret != MV_OK) {
      return ret;
    }

    if (config->exposure_auto_mode == 0) {
      const float tolerance = std::max(1.0f, std::fabs(config->exposure_time_us) * 0.01f);
      ret = set_and_verify_float_value(
          camera->handle,
          "ExposureTime",
          config->exposure_time_us,
          tolerance,
          err_buf,
          err_buf_len);
      if (ret != MV_OK) {
        return ret;
      }
    }

    ret = set_and_verify_enum_value(
        camera->handle,
        "GainAuto",
        static_cast<unsigned int>(config->gain_auto_mode),
        err_buf,
        err_buf_len);
    if (ret != MV_OK) {
      return ret;
    }

    ret = set_and_verify_enum_value(
        camera->handle,
        "BalanceWhiteAuto",
        static_cast<unsigned int>(config->balance_white_auto_mode),
        err_buf,
        err_buf_len);
    if (ret != MV_OK) {
      return ret;
    }

    ret = set_and_verify_bool_value(
        camera->handle,
        "BlackLevelEnable",
        config->black_level_enable != 0,
        err_buf,
        err_buf_len);
    if (ret != MV_OK) {
      return ret;
    }

    if (config->black_level_enable != 0) {
      ret = set_and_verify_int_value(
          camera->handle,
          "BlackLevel",
          static_cast<unsigned int>(config->black_level),
          err_buf,
          err_buf_len);
      if (ret != MV_OK) {
        return ret;
      }
    }

    ret = set_and_verify_int_value(
        camera->handle,
        "Brightness",
        static_cast<unsigned int>(config->brightness),
        err_buf,
        err_buf_len);
    if (ret != MV_OK) {
      return ret;
    }
  }

  ret = query_int_value(camera->handle, "Width", &camera->width);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("Width", ret));
    return ret;
  }
  camera->sensor_width = camera->width;

  ret = query_int_value(camera->handle, "Height", &camera->height);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("Height", ret));
    return ret;
  }
  camera->sensor_height = camera->height;

  camera->rotate_180 = config != nullptr && config->rotate_180 != 0;
  return configure_frame_processing(camera, config, err_buf, err_buf_len);
}

int open_device_for_serial(
    const char* serial,
    const MVSOpenConfig* config,
    CameraHandle** out_camera,
    char* err_buf,
    size_t err_buf_len) {
  SdkSessionGuard sdk_session;
  int ret = sdk_session.acquire(err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }

  MV_CC_DEVICE_INFO_LIST device_list = {};
  const unsigned int layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE |
                                  MV_GENTL_CAMERALINK_DEVICE |
                                  MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE;
  ret = MV_CC_EnumDevices(layer_type, &device_list);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("MV_CC_EnumDevices", ret));
    return ret;
  }

  if (device_list.nDeviceNum == 0) {
    set_error(err_buf, err_buf_len, "No MVS devices found");
    return MV_E_NODATA;
  }

  MV_CC_DEVICE_INFO* target_device = nullptr;
  for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
    MV_CC_DEVICE_INFO* device_info = device_list.pDeviceInfo[i];
    if (device_info == nullptr) {
      continue;
    }
    if (extract_device_serial(*device_info) == serial) {
      target_device = device_info;
      break;
    }
  }

  if (target_device == nullptr) {
    set_error(err_buf, err_buf_len, std::string("Camera not found: ") + serial);
    return MV_E_PARAMETER;
  }

  auto* camera = new CameraHandle();
  ret = MV_CC_CreateHandle(&camera->handle, target_device);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("MV_CC_CreateHandle", ret));
    delete camera;
    return ret;
  }

  ret = MV_CC_OpenDevice(camera->handle, MV_ACCESS_Exclusive, 0);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("MV_CC_OpenDevice", ret));
    cleanup_open_failure(camera, false, nullptr, 0);
    return ret;
  }

  if (target_device->nTLayerType == MV_GIGE_DEVICE) {
    const int packet_size = MV_CC_GetOptimalPacketSize(camera->handle);
    if (packet_size > 0) {
      MV_CC_SetIntValue(
          camera->handle, "GevSCPSPacketSize", static_cast<unsigned int>(packet_size));
    }
  }

  ret = configure_camera(camera, config, err_buf, err_buf_len);
  if (ret != MV_OK) {
    cleanup_open_failure(camera, true, nullptr, 0);
    return ret;
  }

  ret = MV_CC_StartGrabbing(camera->handle);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("MV_CC_StartGrabbing", ret));
    cleanup_open_failure(camera, true, nullptr, 0);
    return ret;
  }

  register_open_camera();
  *out_camera = camera;
  return MV_OK;
}

}  // namespace

extern "C" {

int mvs_enumerate_serials(
    char* serials_buf,
    size_t serials_buf_len,
    size_t* required_len,
    char* err_buf,
    size_t err_buf_len) {
  SdkSessionGuard sdk_session;
  const int init_ret = sdk_session.acquire(err_buf, err_buf_len);
  if (init_ret != MV_OK) {
    return init_ret;
  }

  MV_CC_DEVICE_INFO_LIST device_list = {};
  const unsigned int layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE |
                                  MV_GENTL_CAMERALINK_DEVICE |
                                  MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE;
  const int ret = MV_CC_EnumDevices(layer_type, &device_list);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("MV_CC_EnumDevices", ret));
    return ret;
  }

  std::ostringstream serials;
  for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
    MV_CC_DEVICE_INFO* device_info = device_list.pDeviceInfo[i];
    if (device_info == nullptr) {
      continue;
    }
    const std::string serial = extract_device_serial(*device_info);
    if (serial.empty()) {
      continue;
    }
    serials << serial << '\n';
  }

  const std::string payload = serials.str();
  if (required_len != nullptr) {
    *required_len = payload.size() + 1;
  }
  if (serials_buf == nullptr || serials_buf_len < payload.size() + 1) {
    return kBackendBufferTooSmall;
  }

  std::memcpy(serials_buf, payload.c_str(), payload.size() + 1);
  return kBackendOk;
}

int mvs_open_camera(
    const char* serial,
    const MVSOpenConfig* config,
    void** out_handle,
    char* err_buf,
    size_t err_buf_len) {
  if (serial == nullptr || out_handle == nullptr) {
    set_error(err_buf, err_buf_len, "serial or out_handle is null");
    return MV_E_PARAMETER;
  }

  CameraHandle* camera = nullptr;
  const int ret = open_device_for_serial(serial, config, &camera, err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }

  *out_handle = camera;
  return MV_OK;
}

int mvs_close_camera(
    void* handle,
    int32_t* out_handle_destroyed,
    char* err_buf,
    size_t err_buf_len) {
  if (out_handle_destroyed != nullptr) {
    *out_handle_destroyed = 0;
  }
  if (handle == nullptr) {
    if (out_handle_destroyed != nullptr) {
      *out_handle_destroyed = 1;
    }
    return MV_OK;
  }

  auto* camera = static_cast<CameraHandle*>(handle);
  if (camera->handle == nullptr) {
    if (out_handle_destroyed != nullptr) {
      *out_handle_destroyed = 1;
    }
    return MV_OK;
  }

  SdkSessionGuard sdk_session;
  const int init_ret = sdk_session.acquire(err_buf, err_buf_len);
  if (init_ret != MV_OK) {
    return init_ret;
  }

  int first_error = MV_OK;
  auto capture_error = [&](const char* action, int ret_code) {
    if (ret_code == MV_OK || ret_code == MV_E_CALLORDER || first_error != MV_OK) {
      return;
    }
    first_error = ret_code;
    set_error(err_buf, err_buf_len, format_mv_error(action, ret_code));
  };

  int ret = MV_CC_ClearImageBuffer(camera->handle);
  capture_error("MV_CC_ClearImageBuffer", ret);

  ret = MV_CC_StopGrabbing(camera->handle);
  capture_error("MV_CC_StopGrabbing", ret);

  ret = MV_CC_CloseDevice(camera->handle);
  capture_error("MV_CC_CloseDevice", ret);

  ret = MV_CC_DestroyHandle(camera->handle);
  const bool handle_destroyed = (ret == MV_OK);
  capture_error("MV_CC_DestroyHandle", ret);

  if (handle_destroyed) {
    camera->handle = nullptr;
    unregister_open_camera();
    if (out_handle_destroyed != nullptr) {
      *out_handle_destroyed = 1;
    }
  }

  return first_error;
}

int mvs_destroy_camera(
    void* handle,
    char* err_buf,
    size_t err_buf_len) {
  if (handle == nullptr) {
    return MV_OK;
  }

  auto* camera = static_cast<CameraHandle*>(handle);
  if (camera->handle != nullptr) {
    set_error(err_buf, err_buf_len, "Cannot destroy camera before closing the SDK handle");
    return MV_E_CALLORDER;
  }

  {
    std::lock_guard<std::mutex> lock(camera->frame_mutex);
    for (const auto& slot : camera->frame_slots) {
      if (slot.in_use) {
        set_error(err_buf, err_buf_len, "Cannot destroy camera while frame views are still in use");
        return MV_E_CALLORDER;
      }
    }
  }

  delete camera;
  return MV_OK;
}

int mvs_get_frame_shape(
    void* handle,
    unsigned int* width,
    unsigned int* height,
    char* err_buf,
    size_t err_buf_len) {
  if (handle == nullptr || width == nullptr || height == nullptr) {
    set_error(err_buf, err_buf_len, "handle or output pointer is null");
    return MV_E_PARAMETER;
  }

  auto* camera = static_cast<CameraHandle*>(handle);
  *width = camera->width;
  *height = camera->height;
  return MV_OK;
}

int mvs_get_runtime_config(
    void* handle,
    MVSCameraRuntimeConfig* out_config,
    char* err_buf,
    size_t err_buf_len) {
  if (handle == nullptr || out_config == nullptr) {
    set_error(err_buf, err_buf_len, "handle or out_config is null");
    return MV_E_PARAMETER;
  }

  auto* camera = static_cast<CameraHandle*>(handle);
  if (camera->handle == nullptr) {
    set_error(err_buf, err_buf_len, "camera is already closed");
    return MV_E_CALLORDER;
  }
  MVSCameraRuntimeConfig config = {};
  unsigned int uint_value = 0;
  float float_value = 0.0f;
  bool bool_value = false;
  int ret = query_int_value(camera->handle, "Width", &uint_value);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("Width", ret));
    return ret;
  }
  config.width = uint_value;

  ret = query_int_value(camera->handle, "Height", &uint_value);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("Height", ret));
    return ret;
  }
  config.height = uint_value;

  ret = query_int_value(camera->handle, "OffsetX", &uint_value);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("OffsetX", ret));
    return ret;
  }
  config.offset_x = uint_value;

  ret = query_int_value(camera->handle, "OffsetY", &uint_value);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("OffsetY", ret));
    return ret;
  }
  config.offset_y = uint_value;

  ret = query_bool_value(camera->handle, "AcquisitionFrameRateEnable", &bool_value);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("AcquisitionFrameRateEnable", ret));
    return ret;
  }
  config.acquisition_frame_rate_enable = bool_value ? 1 : 0;

  ret = query_float_value(camera->handle, "AcquisitionFrameRate", &float_value);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("AcquisitionFrameRate", ret));
    return ret;
  }
  config.acquisition_frame_rate_fps = float_value;

  ret = query_enum_value(camera->handle, "ExposureAuto", &uint_value);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("ExposureAuto", ret));
    return ret;
  }
  config.exposure_auto_mode = static_cast<int32_t>(uint_value);

  ret = query_float_value(camera->handle, "ExposureTime", &float_value);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("ExposureTime", ret));
    return ret;
  }
  config.exposure_time_us = float_value;

  ret = query_enum_value(camera->handle, "GainAuto", &uint_value);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("GainAuto", ret));
    return ret;
  }
  config.gain_auto_mode = static_cast<int32_t>(uint_value);

  ret = query_enum_value(camera->handle, "BalanceWhiteAuto", &uint_value);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("BalanceWhiteAuto", ret));
    return ret;
  }
  config.balance_white_auto_mode = static_cast<int32_t>(uint_value);

  ret = query_bool_value(camera->handle, "BlackLevelEnable", &bool_value);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("BlackLevelEnable", ret));
    return ret;
  }
  config.black_level_enable = bool_value ? 1 : 0;

  ret = query_int_value(camera->handle, "BlackLevel", &uint_value);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("BlackLevel", ret));
    return ret;
  }
  config.black_level = static_cast<int32_t>(uint_value);

  ret = query_int_value(camera->handle, "Brightness", &uint_value);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("Brightness", ret));
    return ret;
  }
  config.brightness = static_cast<int32_t>(uint_value);

  *out_config = config;
  return MV_OK;
}

int mvs_grab_frame_rgb(
    void* handle,
    unsigned int timeout_ms,
    unsigned char* out_buffer,
    size_t out_buffer_len,
    MVSFrameInfo* out_frame_info,
    char* err_buf,
    size_t err_buf_len) {
  if (handle == nullptr || out_buffer == nullptr || out_frame_info == nullptr) {
    set_error(err_buf, err_buf_len, "handle, out_buffer, or out_frame_info is null");
    return MV_E_PARAMETER;
  }

  auto* camera = static_cast<CameraHandle*>(handle);
  if (camera->handle == nullptr) {
    set_error(err_buf, err_buf_len, "camera is already closed");
    return MV_E_CALLORDER;
  }
  std::lock_guard<std::mutex> grab_lock(camera->grab_mutex);
  MV_FRAME_OUT frame_out = {};
  int ret = MV_CC_GetImageBuffer(camera->handle, &frame_out, timeout_ms);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("MV_CC_GetImageBuffer", ret));
    return ret;
  }

  const size_t required_size = rgb_byte_size(camera->width, camera->height);
  if (out_buffer_len < required_size) {
    MV_CC_FreeImageBuffer(camera->handle, &frame_out);
    set_error(err_buf, err_buf_len, "RGB output buffer is too small");
    return MV_E_NOENOUGH_BUF;
  }

  MV_CC_PIXEL_CONVERT_PARAM convert_param = {};
  convert_param.nWidth = frame_out.stFrameInfo.nWidth;
  convert_param.nHeight = frame_out.stFrameInfo.nHeight;
  convert_param.pSrcData = frame_out.pBufAddr;
  convert_param.nSrcDataLen = frame_out.stFrameInfo.nFrameLen;
  convert_param.enSrcPixelType = frame_out.stFrameInfo.enPixelType;
  convert_param.enDstPixelType = PixelType_Gvsp_RGB8_Packed;
  uint8_t* convert_dst = out_buffer;
  size_t convert_dst_len = out_buffer_len;
  if (camera->processing_required) {
    convert_dst = camera->rgb_workspace.data();
    convert_dst_len = camera->rgb_workspace.size();
  }
  convert_param.pDstBuffer = convert_dst;
  convert_param.nDstBufferSize = static_cast<unsigned int>(convert_dst_len);

  ret = MV_CC_ConvertPixelType(camera->handle, &convert_param);
  if (ret != MV_OK) {
    MV_CC_FreeImageBuffer(camera->handle, &frame_out);
    set_error(err_buf, err_buf_len, format_mv_error("MV_CC_ConvertPixelType", ret));
    return ret;
  }

  if (camera->rotate_180) {
    rotate_rgb_image_180(convert_dst, frame_out.stFrameInfo.nWidth, frame_out.stFrameInfo.nHeight);
  }

  ret = transform_rgb_output(
      camera,
      convert_dst,
      frame_out.stFrameInfo.nWidth,
      frame_out.stFrameInfo.nHeight,
      out_buffer,
      out_buffer_len,
      err_buf,
      err_buf_len);
  if (ret != MV_OK) {
    MV_CC_FreeImageBuffer(camera->handle, &frame_out);
    return ret;
  }

  out_frame_info->width = camera->width;
  out_frame_info->height = camera->height;
  out_frame_info->frame_num = frame_out.stFrameInfo.nFrameNum;
  out_frame_info->timestamp_ns = now_timestamp_ns();

  ret = MV_CC_FreeImageBuffer(camera->handle, &frame_out);
  if (ret != MV_OK) {
    set_error(err_buf, err_buf_len, format_mv_error("MV_CC_FreeImageBuffer", ret));
    return ret;
  }

  return MV_OK;
}

int mvs_acquire_frame_rgb_view(
    void* handle,
    unsigned int timeout_ms,
    MVSFrameView* out_frame_view,
    char* err_buf,
    size_t err_buf_len) {
  if (handle == nullptr || out_frame_view == nullptr) {
    set_error(err_buf, err_buf_len, "handle or out_frame_view is null");
    return MV_E_PARAMETER;
  }

  auto* camera = static_cast<CameraHandle*>(handle);
  if (camera->handle == nullptr) {
    set_error(err_buf, err_buf_len, "camera is already closed");
    return MV_E_CALLORDER;
  }
  std::lock_guard<std::mutex> grab_lock(camera->grab_mutex);
  uint32_t slot_index = 0;
  int ret = acquire_frame_slot(camera, &slot_index, err_buf, err_buf_len);
  if (ret != MV_OK) {
    return ret;
  }

  auto release_reserved_slot = [&]() {
    release_frame_slot(camera, slot_index, nullptr, 0);
  };

  MV_FRAME_OUT frame_out = {};
  ret = MV_CC_GetImageBuffer(camera->handle, &frame_out, timeout_ms);
  if (ret != MV_OK) {
    release_reserved_slot();
    set_error(err_buf, err_buf_len, format_mv_error("MV_CC_GetImageBuffer", ret));
    return ret;
  }

  auto& slot = camera->frame_slots[slot_index];
  const size_t required_size = rgb_byte_size(camera->width, camera->height);
  if (slot.data.size() < required_size) {
    MV_CC_FreeImageBuffer(camera->handle, &frame_out);
    release_reserved_slot();
    set_error(
        err_buf,
        err_buf_len,
        "Frame slot is smaller than the incoming frame. Reopen camera with a larger pool slot size.");
    return MV_E_NOENOUGH_BUF;
  }

  MV_CC_PIXEL_CONVERT_PARAM convert_param = {};
  convert_param.nWidth = frame_out.stFrameInfo.nWidth;
  convert_param.nHeight = frame_out.stFrameInfo.nHeight;
  convert_param.pSrcData = frame_out.pBufAddr;
  convert_param.nSrcDataLen = frame_out.stFrameInfo.nFrameLen;
  convert_param.enSrcPixelType = frame_out.stFrameInfo.enPixelType;
  convert_param.enDstPixelType = PixelType_Gvsp_RGB8_Packed;
  uint8_t* convert_dst = slot.data.data();
  size_t convert_dst_len = slot.data.size();
  if (camera->processing_required) {
    convert_dst = camera->rgb_workspace.data();
    convert_dst_len = camera->rgb_workspace.size();
  }
  convert_param.pDstBuffer = convert_dst;
  convert_param.nDstBufferSize = static_cast<unsigned int>(convert_dst_len);

  ret = MV_CC_ConvertPixelType(camera->handle, &convert_param);
  if (ret != MV_OK) {
    MV_CC_FreeImageBuffer(camera->handle, &frame_out);
    release_reserved_slot();
    set_error(err_buf, err_buf_len, format_mv_error("MV_CC_ConvertPixelType", ret));
    return ret;
  }

  if (camera->rotate_180) {
    rotate_rgb_image_180(convert_dst, frame_out.stFrameInfo.nWidth, frame_out.stFrameInfo.nHeight);
  }

  ret = transform_rgb_output(
      camera,
      convert_dst,
      frame_out.stFrameInfo.nWidth,
      frame_out.stFrameInfo.nHeight,
      slot.data.data(),
      slot.data.size(),
      err_buf,
      err_buf_len);
  if (ret != MV_OK) {
    MV_CC_FreeImageBuffer(camera->handle, &frame_out);
    release_reserved_slot();
    return ret;
  }

  out_frame_view->width = camera->width;
  out_frame_view->height = camera->height;
  out_frame_view->stride_bytes = camera->width * 3U;
  out_frame_view->channels = 3U;
  out_frame_view->slot_index = slot_index;
  out_frame_view->frame_num = frame_out.stFrameInfo.nFrameNum;
  out_frame_view->timestamp_ns = now_timestamp_ns();
  out_frame_view->data = slot.data.data();
  out_frame_view->data_len = required_size;

  ret = MV_CC_FreeImageBuffer(camera->handle, &frame_out);
  if (ret != MV_OK) {
    release_reserved_slot();
    set_error(err_buf, err_buf_len, format_mv_error("MV_CC_FreeImageBuffer", ret));
    return ret;
  }

  return MV_OK;
}

int mvs_release_frame(
    void* handle,
    uint32_t slot_index,
    char* err_buf,
    size_t err_buf_len) {
  if (handle == nullptr) {
    set_error(err_buf, err_buf_len, "handle is null");
    return MV_E_PARAMETER;
  }

  auto* camera = static_cast<CameraHandle*>(handle);
  return release_frame_slot(camera, slot_index, err_buf, err_buf_len);
}

}  // extern "C"
