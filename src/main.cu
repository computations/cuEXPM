#include "expm.cuh"
#include <limits>

int main() {
  constexpr size_t n = 1 << 4;
  CudaRateMatrix rm(n);
  thrust::host_vector<double> a(n * n);

  auto gen_f = [] __host__() {
    return (double)rand() / (double)std::numeric_limits<int>::max();
  };
  thrust::generate(a.begin(), a.end(), gen_f);

  rm.set_matrix(a);
  rm.expm(1.0);
}
