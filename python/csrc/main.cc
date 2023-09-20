#include <pybind11/pybind11.h>

void init_vk_backend_for_triton(pybind11::module &m);

PYBIND11_MODULE(libvk_backend_for_triton, m) {
  m.doc() = "Python bindings to the C++ VK XPU Backend for Triton API";
  init_vk_backend_for_triton(m);
}
