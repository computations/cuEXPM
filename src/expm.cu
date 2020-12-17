#include "expm.cuh"
#include <cublasLt.h>
#include <stdexcept>
#include <thrust/device_free.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <iostream>

inline void checkCudaStatus(cudaError_t status) {
  if (status != cudaSuccess) {
    printf("cuda API failed with status %d: %s\n", status,
           cudaGetErrorString(status));
    throw std::logic_error("cuda API failed");
  }
}

inline void checkCublasStatus(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("cuBLAS API failed with status %d\n", status);
    throw std::logic_error("cuBLAS API failed");
  }
}

inline void checkCuSolverStatus(cusolverStatus_t status) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    printf("cuSOLVER API failed with status %d\n", status);
    throw std::logic_error("cuSolver API failed");
  }
}

inline void print_matrix(const thrust::host_vector<double> &a,
                         size_t row_size) {
  std::cout << "[ ";
  for (size_t i = 0; i < row_size; i++) {
    if (i != 0) {
      std::cout << " ";
    }
    for (size_t j = 0; j < row_size; j++) {
      std::cout << a[i * row_size + j];
      if (j != row_size - 1) {
        std::cout << " ";
      } else {
        if (i == row_size - 1) {
          std::cout << "]";
        }
        std::cout << "\n";
      }
    }
  }
  std::cout << "\n";
}


template <typename T>
struct identity_matrix_functor : public thrust::unary_function<T, T> {
  const size_t _row_size;

  __host__ __device__ identity_matrix_functor(size_t row_size)
      : _row_size{row_size} {}

  __host__ __device__ T operator()(T a) {
    size_t i = static_cast<size_t>(a + 0.5);
    size_t r = i / _row_size;
    size_t c = i % _row_size;
    return 1.0 * (r == c);
  }
};

template <typename T>
struct scale_functor : public thrust::unary_function<T, T> {
  const T _scale;

  __host__ __device__ scale_functor(T scale) : _scale{scale} {}

  __host__ __device__ T operator()(T x) { return x * _scale; }
};

template <typename T> struct inf_norm {
  __host__ __device__ bool operator()(const T &lhs, const T &rhs) {
    return fabs(lhs) < fabs(rhs);
  }
};

template <typename T> struct saxpy : public thrust::binary_function<T, T, T> {
  const T _c;
  __host__ __device__ saxpy(T c) : _c{c} {}

  __host__ __device__ T operator()(const T &x, const T &y) const {
    return x + _c * y;
  }
};

CudaRateMatrix::CudaRateMatrix(size_t n) {
  _row_size = n;
  _matrix_size = _row_size * _row_size;
  _workspace_size = 8096;

  _A_host.resize(_matrix_size, 0.0);
  _eA_host.resize(_matrix_size, 0.0);

  _eA_dev.resize(_matrix_size, 0.0);
  _A_dev.resize(_matrix_size, 0.0);
  _X_dev.resize(_matrix_size, 0.0);
  _N_dev.resize(_matrix_size, 0.0);
  _D_dev.resize(_matrix_size, 0.0);
  _I_dev.resize(_matrix_size, 0.0);
  _P_dev.resize(_row_size);

  init_I();

  _workspace = thrust::device_malloc<uint8_t>(_workspace_size);
  _info = thrust::device_malloc<int>(1);

  init_cublas();
  init_cublaslt();
  init_cusolver();
}

CudaRateMatrix::~CudaRateMatrix() { thrust::device_free(_workspace); }

void CudaRateMatrix::set_identity(thrust::device_vector<double> &I) {
  thrust::sequence(I.begin(), I.end(), 0);
  thrust::transform(I.begin(), I.end(), I.begin(),
                    identity_matrix_functor<double>(_row_size));
}

void CudaRateMatrix::init_I() {
  for (size_t i = 0; i < _matrix_size; i += _row_size + 1) {
    _I_dev[i] = 1.0;
  }
}

void CudaRateMatrix::init_cublas() {
  checkCublasStatus(cublasCreate_v2(&_blas_handle));
}

void CudaRateMatrix::init_cublaslt() {
  checkCublasStatus(cublasLtCreate(&_lthandle));
  checkCublasStatus(
      cublasLtMatmulDescCreate(&_operation, CUBLAS_COMPUTE_64F, CUDA_R_64F));
  checkCublasStatus(
      cublasLtMatmulDescSetAttribute(_operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                     &_transpose, sizeof(_transpose)));
  checkCublasStatus(
      cublasLtMatmulDescSetAttribute(_operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                     &_transpose, sizeof(_transpose)));

  checkCublasStatus(cublasLtMatrixLayoutCreate(
      &_layout,   // layout structure
      CUDA_R_64F, // Datatype
      _row_size,  // rows
      _row_size,  // cols
      _row_size   //"leading dim". the number of elements to
                  // skip to get to the next col. except in
                  // our case we specify the row size,
                  // because we are row major
      ));

  checkCublasStatus(cublasLtMatrixLayoutSetAttribute( // set the matrix layout
      _layout,                                        // which layout to set
      CUBLASLT_MATRIX_LAYOUT_ORDER,                   // what we are doing
      &_row_order,       // we are setting to a row major order
      sizeof(_row_order) // Size of the attribute
      ));

  checkCublasStatus(cublasLtMatmulPreferenceCreate(&_preference));
  checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(
      _preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &_workspace_size,
      sizeof(_workspace_size)));

  int returned_results = 0;

  checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(
      _lthandle, _operation, _layout, _layout, _layout, _layout, _preference, 1,
      &_heuristics, &returned_results));

  if (returned_results == 0) {
    throw std::runtime_error{"failed to find an algorithm"};
  }
}

void CudaRateMatrix::init_cusolver() {
  checkCuSolverStatus(cusolverDnCreate(&_solve_handle));
  int required_work_size = 0;
  checkCuSolverStatus(cusolverDnDgetrf_bufferSize(
      _solve_handle, _row_size, _row_size,
      thrust::raw_pointer_cast(_D_dev.data()), _row_size, &required_work_size));
  if (required_work_size > _workspace_size) {
    _workspace_size = required_work_size;
    thrust::device_free(_workspace);
    _workspace = thrust::device_malloc<uint8_t>(_workspace_size);
  }
}

void CudaRateMatrix::expm_ss(double t) {
  _A_dev = _A_host;
  thrust::transform(_A_dev.begin(), _A_dev.end(), _A_dev.begin(),
                    scale_functor<double>(t));
  int scale =
      *thrust::max_element(_A_dev.begin(), _A_dev.end(), inf_norm<double>());

  scale = std::max(0, 1 + scale);

  thrust::transform(_A_dev.begin(), _A_dev.end(), _A_dev.begin(),
                    scale_functor<double>(1 / std::pow(2.0, scale)));

  constexpr int q = 3;
  double c = 0.5;
  double sign = -1.0;

  thrust::transform(_I_dev.begin(), _I_dev.end(), _A_dev.begin(),
                    _N_dev.begin(), saxpy<double>(c));
  thrust::transform(_I_dev.begin(), _I_dev.end(), _A_dev.begin(),
                    _D_dev.begin(), saxpy<double>(-c));

  _X_dev = _A_dev;
  double alpha = 1.0;
  double beta = 0.0;
  for (int i = 2; i < q; i++) {

    c = c * (q - i + 1) / (i * (2 * q - i + 1));

    /* X = A * X */
    checkCublasStatus(cublasLtMatmul(
        _lthandle, _operation, &alpha, thrust::raw_pointer_cast(_A_dev.data()),
        _layout, thrust::raw_pointer_cast(_X_dev.data()), _layout, &beta,
        thrust::raw_pointer_cast(_X_dev.data()), _layout,
        thrust::raw_pointer_cast(_X_dev.data()), _layout, &_heuristics.algo,
        thrust::raw_pointer_cast(_workspace), _workspace_size, 0));

    /* N += c * X */
    thrust::transform(_N_dev.begin(), _N_dev.end(), _X_dev.begin(),
                      _N_dev.begin(), saxpy<double>(c));

    sign *= -1.0;

    /* D += sign * c * X */
    thrust::transform(_D_dev.begin(), _D_dev.end(), _X_dev.begin(),
                      _D_dev.begin(), saxpy<double>(sign * c));
  }

  set_identity(_eA_dev);

  /* factorize D */
  checkCuSolverStatus(
      cusolverDnDgetrf(_solve_handle, _row_size, _row_size,
                       thrust::raw_pointer_cast(_D_dev.data()), _row_size,
                       (double *)thrust::raw_pointer_cast(_workspace),
                       thrust::raw_pointer_cast(_P_dev.data()),
                       thrust::raw_pointer_cast(_info)));

  if (*_info != 0) {
    throw std::runtime_error{"LU factorization was unsuccsessful"};
  }

  /*Solve D * A = N */
  checkCuSolverStatus(
      cusolverDnDgetrs(_solve_handle, CUBLAS_OP_N, _row_size, _row_size,
                       thrust::raw_pointer_cast(_D_dev.data()), _row_size,
                       thrust::raw_pointer_cast(_P_dev.data()),
                       thrust::raw_pointer_cast(_N_dev.data()), _row_size,
                       thrust::raw_pointer_cast(_info)));

  if (*_info != 0) {
    throw std::runtime_error{"LU factorization was unsuccsessful"};
  }

  for (int i = 0; i < scale; i++) {
    /* N *= N */
    checkCublasStatus(cublasLtMatmul(
        _lthandle, _operation, &alpha, thrust::raw_pointer_cast(_N_dev.data()),
        _layout, thrust::raw_pointer_cast(_N_dev.data()), _layout, &beta,
        thrust::raw_pointer_cast(_N_dev.data()), _layout,
        thrust::raw_pointer_cast(_N_dev.data()), _layout, &_heuristics.algo,
        thrust::raw_pointer_cast(_workspace), _workspace_size, 0));
  }
  _eA_host = _N_dev;

  std::cout << "eA dev: " << std::endl;
  print_matrix(_eA_host, _row_size);
}
