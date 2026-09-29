#include "../include/mvs_camera_backend.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

namespace {

constexpr size_t kErrorBufferBytes = 2048;
constexpr uint32_t kMvENoData = 0x80000007u;

uint32_t normalize_ret_code(int ret_code) {
  return static_cast<uint32_t>(ret_code);
}

struct BackendError : std::runtime_error {
  explicit BackendError(std::string message)
      : std::runtime_error(std::move(message)) {}
};

struct NoDataError : BackendError {
  explicit NoDataError(std::string message)
      : BackendError(std::move(message)) {}
};

std::string format_error_message(const char* action, int ret_code, const char* err_buf) {
  if (err_buf != nullptr && err_buf[0] != '\0') {
    return std::string(err_buf);
  }

  std::ostringstream oss;
  oss << action << " failed with code 0x" << std::hex
      << normalize_ret_code(ret_code);
  return oss.str();
}

[[noreturn]] void throw_backend_error(
    const char* action,
    int ret_code,
    const char* err_buf) {
  const std::string message = format_error_message(action, ret_code, err_buf);
  if (normalize_ret_code(ret_code) == kMvENoData) {
    throw NoDataError(message);
  }
  throw BackendError(message);
}

struct DeviceOptions {
  uint32_t roi_width = 0;
  uint32_t roi_height = 0;
  uint32_t offset_x = 0;
  uint32_t offset_y = 0;
  uint32_t crop_x = 0;
  uint32_t crop_y = 0;
  uint32_t crop_width = 0;
  uint32_t crop_height = 0;
  uint32_t output_width = 0;
  uint32_t output_height = 0;
  uint32_t image_node_num = 1;
  uint32_t frame_pool_size = 4;
  bool rotate_180 = true;
  bool acquisition_frame_rate_enable = true;
  float acquisition_frame_rate_fps = 30.0f;
  int32_t exposure_auto_mode = 0;
  float exposure_time_us = 20000.0f;
  int32_t gain_auto_mode = 2;
  int32_t balance_white_auto_mode = 2;
  bool black_level_enable = true;
  int32_t black_level = 240;
  int32_t brightness = 40;
};

struct RuntimeConfig {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t offset_x = 0;
  uint32_t offset_y = 0;
  bool acquisition_frame_rate_enable = false;
  float acquisition_frame_rate_fps = 0.0f;
  int32_t exposure_auto_mode = 0;
  float exposure_time_us = 0.0f;
  int32_t gain_auto_mode = 0;
  int32_t balance_white_auto_mode = 0;
  bool black_level_enable = false;
  int32_t black_level = 0;
  int32_t brightness = 0;
};

class DeviceState : public std::enable_shared_from_this<DeviceState> {
 public:
  DeviceState(std::string serial, DeviceOptions options)
      : serial_(std::move(serial)), options_(options) {
    MVSOpenConfig config{};
    config.roi_width = options_.roi_width;
    config.roi_height = options_.roi_height;
    config.offset_x = options_.offset_x;
    config.offset_y = options_.offset_y;
    config.crop_x = options_.crop_x;
    config.crop_y = options_.crop_y;
    config.crop_width = options_.crop_width;
    config.crop_height = options_.crop_height;
    config.output_width = options_.output_width;
    config.output_height = options_.output_height;
    config.image_node_num = options_.image_node_num;
    config.frame_pool_size = options_.frame_pool_size;
    config.rotate_180 = options_.rotate_180 ? 1 : 0;
    config.acquisition_frame_rate_enable = options_.acquisition_frame_rate_enable ? 1 : 0;
    config.acquisition_frame_rate_fps = options_.acquisition_frame_rate_fps;
    config.exposure_auto_mode = options_.exposure_auto_mode;
    config.exposure_time_us = options_.exposure_time_us;
    config.gain_auto_mode = options_.gain_auto_mode;
    config.balance_white_auto_mode = options_.balance_white_auto_mode;
    config.black_level_enable = options_.black_level_enable ? 1 : 0;
    config.black_level = options_.black_level;
    config.brightness = options_.brightness;

    std::array<char, kErrorBufferBytes> err_buf{};
    const int ret =
        mvs_open_camera(serial_.c_str(), &config, &handle_, err_buf.data(), err_buf.size());
    if (ret != 0) {
      handle_ = nullptr;
      throw_backend_error("mvs_open_camera", ret, err_buf.data());
    }
    device_closed_ = false;
  }

  ~DeviceState() noexcept {
    close_noexcept(true);
  }

  MVSFrameView acquire_frame(uint32_t timeout_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (handle_ == nullptr || device_closed_) {
      throw BackendError("MVS nanobind device is closed");
    }
    if (close_requested_) {
      throw BackendError("MVS nanobind device is closing");
    }

    std::array<char, kErrorBufferBytes> err_buf{};
    MVSFrameView view{};
    const int ret =
        mvs_acquire_frame_rgb_view(handle_, timeout_ms, &view, err_buf.data(), err_buf.size());
    if (ret != 0) {
      throw_backend_error("mvs_acquire_frame_rgb_view", ret, err_buf.data());
    }

    ++outstanding_frames_;
    return view;
  }

  void release_frame(uint32_t slot_index, bool throw_on_error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (outstanding_frames_ > 0) {
      --outstanding_frames_;
    }

    if (handle_ != nullptr) {
      std::array<char, kErrorBufferBytes> err_buf{};
      const int ret = mvs_release_frame(handle_, slot_index, err_buf.data(), err_buf.size());
      if (ret != 0 && throw_on_error) {
        throw_backend_error("mvs_release_frame", ret, err_buf.data());
      }
    }

    if (close_requested_) {
      finalize_locked(throw_on_error, false);
    }
  }

  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    close_requested_ = true;
    finalize_locked(true, false);
  }

  void close_noexcept(bool from_destructor = false) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    close_requested_ = true;
    try {
      finalize_locked(false, from_destructor);
    } catch (...) {
    }
  }

  bool is_open() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return handle_ != nullptr && !device_closed_;
  }

  std::pair<uint32_t, uint32_t> frame_shape() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (handle_ == nullptr) {
      throw BackendError("MVS nanobind device is closed");
    }

    std::array<char, kErrorBufferBytes> err_buf{};
    uint32_t width = 0;
    uint32_t height = 0;
    const int ret = mvs_get_frame_shape(handle_, &width, &height, err_buf.data(), err_buf.size());
    if (ret != 0) {
      throw_backend_error("mvs_get_frame_shape", ret, err_buf.data());
    }
    return {width, height};
  }

  RuntimeConfig runtime_config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (handle_ == nullptr || device_closed_) {
      throw BackendError("MVS nanobind device is closed");
    }

    std::array<char, kErrorBufferBytes> err_buf{};
    MVSCameraRuntimeConfig runtime_config{};
    const int ret =
        mvs_get_runtime_config(handle_, &runtime_config, err_buf.data(), err_buf.size());
    if (ret != 0) {
      throw_backend_error("mvs_get_runtime_config", ret, err_buf.data());
    }

    RuntimeConfig result{};
    result.width = runtime_config.width;
    result.height = runtime_config.height;
    result.offset_x = runtime_config.offset_x;
    result.offset_y = runtime_config.offset_y;
    result.acquisition_frame_rate_enable =
        runtime_config.acquisition_frame_rate_enable != 0;
    result.acquisition_frame_rate_fps = runtime_config.acquisition_frame_rate_fps;
    result.exposure_auto_mode = runtime_config.exposure_auto_mode;
    result.exposure_time_us = runtime_config.exposure_time_us;
    result.gain_auto_mode = runtime_config.gain_auto_mode;
    result.balance_white_auto_mode = runtime_config.balance_white_auto_mode;
    result.black_level_enable = runtime_config.black_level_enable != 0;
    result.black_level = runtime_config.black_level;
    result.brightness = runtime_config.brightness;
    return result;
  }

 private:
  void finalize_locked(bool throw_on_error, bool force_close) {
    if (handle_ == nullptr) {
      return;
    }

    std::array<char, kErrorBufferBytes> close_err_buf{};
    int close_ret = 0;
    bool close_attempted = false;
    if (!device_closed_ && (close_requested_ || force_close)) {
      int32_t handle_destroyed = 0;
      close_ret =
          mvs_close_camera(handle_, &handle_destroyed, close_err_buf.data(), close_err_buf.size());
      close_attempted = true;
      if (handle_destroyed != 0 || close_ret == 0) {
        device_closed_ = true;
      }
    }

    std::array<char, kErrorBufferBytes> destroy_err_buf{};
    int destroy_ret = 0;
    bool destroy_attempted = false;
    if (device_closed_ && outstanding_frames_ == 0 && handle_ != nullptr) {
      destroy_ret = mvs_destroy_camera(handle_, destroy_err_buf.data(), destroy_err_buf.size());
      destroy_attempted = true;
      if (destroy_ret == 0) {
        handle_ = nullptr;
      }
    }

    if (throw_on_error) {
      if (close_attempted && close_ret != 0) {
        throw_backend_error("mvs_close_camera", close_ret, close_err_buf.data());
      }
      if (destroy_attempted && destroy_ret != 0) {
        throw_backend_error("mvs_destroy_camera", destroy_ret, destroy_err_buf.data());
      }
    }
  }

  std::string serial_;
  DeviceOptions options_;
  mutable std::mutex mutex_;
  void* handle_ = nullptr;
  size_t outstanding_frames_ = 0;
  bool device_closed_ = false;
  bool close_requested_ = false;
};

class FrameHandle : public std::enable_shared_from_this<FrameHandle> {
 public:
  FrameHandle(std::shared_ptr<DeviceState> device, MVSFrameView view)
      : device_(std::move(device)), view_(view) {}

  ~FrameHandle() noexcept {
    release_noexcept();
  }

  nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, 3>, nb::device::cpu> as_array() {
    ensure_alive();
    nb::object owner = nb::find(this);
    if (!owner.is_valid()) {
      owner = nb::cast(shared_from_this());
    }
    return nb::ndarray<nb::numpy, uint8_t, nb::shape<-1, -1, 3>, nb::device::cpu>(
        view_.data,
        {view_.height, view_.width, view_.channels},
        owner,
        {static_cast<int64_t>(view_.stride_bytes), static_cast<int64_t>(view_.channels), 1});
  }

  void release() {
    if (released_) {
      return;
    }
    if (device_ != nullptr) {
      device_->release_frame(view_.slot_index, true);
      device_.reset();
    }
    released_ = true;
  }

  uint64_t frame_id() const {
    return view_.frame_num;
  }

  uint64_t timestamp_ns() const {
    return view_.timestamp_ns;
  }

  uint32_t width() const {
    return view_.width;
  }

  uint32_t height() const {
    return view_.height;
  }

  uint32_t stride_bytes() const {
    return view_.stride_bytes;
  }

  uint32_t slot_index() const {
    return view_.slot_index;
  }

  bool released() const {
    return released_;
  }

 private:
  void ensure_alive() const {
    if (released_ || device_ == nullptr || view_.data == nullptr) {
      throw BackendError("FrameHandle has already been released");
    }
  }

  void release_noexcept() noexcept {
    if (released_) {
      return;
    }
    try {
      if (device_ != nullptr) {
        device_->release_frame(view_.slot_index, false);
      }
    } catch (...) {
    }
    device_.reset();
    released_ = true;
  }

  std::shared_ptr<DeviceState> device_;
  MVSFrameView view_{};
  bool released_ = false;
};

class Device {
 public:
  Device(
      const std::string& serial,
      uint32_t roi_width,
      uint32_t roi_height,
      uint32_t offset_x,
      uint32_t offset_y,
      uint32_t crop_x,
      uint32_t crop_y,
      uint32_t crop_width,
      uint32_t crop_height,
      uint32_t output_width,
      uint32_t output_height,
      uint32_t image_node_num,
      uint32_t frame_pool_size,
      bool rotate_180,
      bool acquisition_frame_rate_enable,
      float acquisition_frame_rate_fps,
      int32_t exposure_auto_mode,
      float exposure_time_us,
      int32_t gain_auto_mode,
      int32_t balance_white_auto_mode,
      bool black_level_enable,
      int32_t black_level,
      int32_t brightness) {
    DeviceOptions options;
    options.roi_width = roi_width;
    options.roi_height = roi_height;
    options.offset_x = offset_x;
    options.offset_y = offset_y;
    options.crop_x = crop_x;
    options.crop_y = crop_y;
    options.crop_width = crop_width;
    options.crop_height = crop_height;
    options.output_width = output_width;
    options.output_height = output_height;
    options.image_node_num = image_node_num;
    options.frame_pool_size = frame_pool_size;
    options.rotate_180 = rotate_180;
    options.acquisition_frame_rate_enable = acquisition_frame_rate_enable;
    options.acquisition_frame_rate_fps = acquisition_frame_rate_fps;
    options.exposure_auto_mode = exposure_auto_mode;
    options.exposure_time_us = exposure_time_us;
    options.gain_auto_mode = gain_auto_mode;
    options.balance_white_auto_mode = balance_white_auto_mode;
    options.black_level_enable = black_level_enable;
    options.black_level = black_level;
    options.brightness = brightness;

    state_ = std::make_shared<DeviceState>(serial, options);
  }

  std::shared_ptr<FrameHandle> read_frame(uint32_t timeout_ms) {
    MVSFrameView view = state_->acquire_frame(timeout_ms);
    try {
      return std::make_shared<FrameHandle>(state_, view);
    } catch (...) {
      state_->release_frame(view.slot_index, false);
      throw;
    }
  }

  void close() {
    state_->close();
  }

  bool is_open() const {
    return state_->is_open();
  }

  std::pair<uint32_t, uint32_t> frame_shape() const {
    return state_->frame_shape();
  }

  RuntimeConfig runtime_config() const {
    return state_->runtime_config();
  }

 private:
  std::shared_ptr<DeviceState> state_;
};

std::vector<std::string> enumerate_serials() {
  std::array<char, kErrorBufferBytes> err_buf{};
  size_t required = 0;
  std::vector<char> payload(1024);
  int ret = mvs_enumerate_serials(
      payload.data(), payload.size(), &required, err_buf.data(), err_buf.size());
  if (ret == 1 && required > payload.size()) {
    payload.assign(required, '\0');
    ret = mvs_enumerate_serials(
        payload.data(), payload.size(), &required, err_buf.data(), err_buf.size());
  }
  if (ret != 0) {
    throw_backend_error("mvs_enumerate_serials", ret, err_buf.data());
  }

  std::vector<std::string> serials;
  std::string current;
  for (char ch : payload) {
    if (ch == '\0') {
      break;
    }
    if (ch == '\n') {
      if (!current.empty()) {
        serials.push_back(current);
        current.clear();
      }
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty()) {
    serials.push_back(current);
  }
  return serials;
}

}  // namespace

NB_MODULE(_mvs_camera, m) {
  m.doc() = "Nanobind-backed MVS camera module";

  nb::exception<BackendError>(m, "BackendError", PyExc_RuntimeError);
  nb::exception<NoDataError>(m, "NoDataError", PyExc_RuntimeError);

  nb::class_<RuntimeConfig>(m, "RuntimeConfig")
      .def(nb::init<>())
      .def_rw("width", &RuntimeConfig::width)
      .def_rw("height", &RuntimeConfig::height)
      .def_rw("offset_x", &RuntimeConfig::offset_x)
      .def_rw("offset_y", &RuntimeConfig::offset_y)
      .def_rw("acquisition_frame_rate_enable", &RuntimeConfig::acquisition_frame_rate_enable)
      .def_rw("acquisition_frame_rate_fps", &RuntimeConfig::acquisition_frame_rate_fps)
      .def_rw("exposure_auto_mode", &RuntimeConfig::exposure_auto_mode)
      .def_rw("exposure_time_us", &RuntimeConfig::exposure_time_us)
      .def_rw("gain_auto_mode", &RuntimeConfig::gain_auto_mode)
      .def_rw("balance_white_auto_mode", &RuntimeConfig::balance_white_auto_mode)
      .def_rw("black_level_enable", &RuntimeConfig::black_level_enable)
      .def_rw("black_level", &RuntimeConfig::black_level)
      .def_rw("brightness", &RuntimeConfig::brightness);

  nb::class_<FrameHandle>(m, "FrameHandle")
      .def("as_array", &FrameHandle::as_array)
      .def("release", &FrameHandle::release)
      .def("__enter__", [](const std::shared_ptr<FrameHandle>& self) { return self; })
      .def(
          "__exit__",
          [](FrameHandle& self, nb::handle, nb::handle, nb::handle) {
            self.release();
            return false;
          })
      .def_prop_ro("frame_id", &FrameHandle::frame_id)
      .def_prop_ro("timestamp_ns", &FrameHandle::timestamp_ns)
      .def_prop_ro("width", &FrameHandle::width)
      .def_prop_ro("height", &FrameHandle::height)
      .def_prop_ro("stride_bytes", &FrameHandle::stride_bytes)
      .def_prop_ro("slot_index", &FrameHandle::slot_index)
      .def_prop_ro("released", &FrameHandle::released);

  nb::class_<Device>(m, "Device")
      .def(
          nb::init<
              const std::string&,
              uint32_t,
              uint32_t,
              uint32_t,
              uint32_t,
              uint32_t,
              uint32_t,
              uint32_t,
              uint32_t,
              uint32_t,
              uint32_t,
              uint32_t,
              uint32_t,
              bool,
              bool,
              float,
              int32_t,
              float,
              int32_t,
              int32_t,
              bool,
              int32_t,
              int32_t>(),
          "serial"_a,
          "roi_width"_a = 0,
          "roi_height"_a = 0,
          "offset_x"_a = 0,
          "offset_y"_a = 0,
          "crop_x"_a = 0,
          "crop_y"_a = 0,
          "crop_width"_a = 0,
          "crop_height"_a = 0,
          "output_width"_a = 0,
          "output_height"_a = 0,
          "image_node_num"_a = 1,
          "frame_pool_size"_a = 4,
          "rotate_180"_a = true,
          "acquisition_frame_rate_enable"_a = true,
          "acquisition_frame_rate_fps"_a = 30.0f,
          "exposure_auto_mode"_a = 0,
          "exposure_time_us"_a = 20000.0f,
          "gain_auto_mode"_a = 2,
          "balance_white_auto_mode"_a = 2,
          "black_level_enable"_a = true,
          "black_level"_a = 240,
          "brightness"_a = 40)
      .def(
          "read_frame",
          [](Device& self, uint32_t timeout_ms) {
            nb::gil_scoped_release guard;
            return self.read_frame(timeout_ms);
          },
          "timeout_ms"_a = 1000)
      .def(
          "close",
          [](Device& self) {
            nb::gil_scoped_release guard;
            self.close();
          })
      .def(
          "get_frame_shape",
          [](const Device& self) {
            auto [width, height] = self.frame_shape();
            return nb::make_tuple(width, height);
          })
      .def("get_runtime_config", &Device::runtime_config)
      .def_prop_ro("is_open", &Device::is_open);

  m.def(
      "enumerate_serials",
      []() {
        nb::gil_scoped_release guard;
        return enumerate_serials();
      });
}
