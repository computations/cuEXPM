#ifndef _EXPM_CUH_
#define _EXPM_CUH_

#include <cublas.h>
#include <cublasLt.h>
#include <cusolverDn.h>
#include <thrust/device_malloc.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class CudaRateMatrix {
public:
  CudaRateMatrix(size_t n);
  ~CudaRateMatrix();

  void expm(double t) { expm_ss(t); }

  void set_matrix(const thrust::host_vector<double> &a) {
    if (a.size() != _matrix_size) {
      throw std::runtime_error{"matrix is the incorrect size"};
    }
    _A_host = a;
  }

private:
  void expm_ss(double t);

  void init_cublas();
  void init_cublaslt();
  void init_cusolver();

  void init_I();
  void init_eA();
  void set_identity(thrust::device_vector<double> &I);

  cublasHandle_t _blas_handle;

  cusolverDnHandle_t _solve_handle;

  cublasLtHandle_t _lthandle;
  cublasLtMatmulDesc_t _operation;
  cublasLtMatmulPreference_t _preference;
  cublasLtMatmulHeuristicResult_t _heuristics;
  cublasLtMatrixLayout_t _layout;
  const cublasLtOrder_t _row_order = CUBLASLT_ORDER_ROW;
  const cublasOperation_t _transpose = CUBLAS_OP_N;

  thrust::device_ptr<uint8_t> _workspace;
  thrust::device_ptr<int> _info;

  thrust::host_vector<double> _A_host;
  thrust::host_vector<double> _eA_host;

  thrust::device_vector<double> _eA_dev;
  thrust::device_vector<double> _A_dev;
  thrust::device_vector<double> _X_dev;
  thrust::device_vector<double> _N_dev;
  thrust::device_vector<double> _D_dev;
  thrust::device_vector<double> _I_dev;
  thrust::device_vector<int> _P_dev;

  size_t _workspace_size;
  size_t _matrix_size;
  size_t _row_size;
};
#endif
