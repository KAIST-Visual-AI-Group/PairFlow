#include <torch/extension.h>
#include "df_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fill_bits", &fill_bits);
  m.def("count_same_tokens", &count_same_tokens);
  // m.def("sum_same_tokens", &sum_same_tokens);
  m.def("sum_same_tokens_large", &sum_same_tokens_large);
  m.def("calculate_val_tn1", &calculate_val_tn1);
  m.def("calculate_val_te1", &calculate_val_te1);
  m.def("normalize", &normalize);
  m.def("get_thres", &get_thres);
};
