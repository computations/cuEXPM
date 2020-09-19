#include "expm.cuh"

int main() {
  constexpr size_t n = 2;
  CudaRateMatrix rm(n);
  thrust::host_vector<double> a(4);
  a[0] = 1;
  a[1] = 2;
  a[2] = 3;
  a[3] = 4;
  rm.set_matrix(a);
  rm.expm(1.0);
}
