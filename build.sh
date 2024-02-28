
rm -rf build

mkdir -p /spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/
mkdir -p /spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/
mkdir -p build/lib.linux-x86_64-cpython-311/

GCC_VERSION=9.2.0


/usr/local/GCC/${GCC_VERSION}/bin/g++ -MMD -MF /spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/flash_api.o.d \
-DNDEBUG -fwrapv -O2 -Wall \
-fPIC -O2 -isystem /data/zhongz2/anaconda3/envs/th21_ds/include \
-fPIC -O2 -isystem /data/zhongz2/anaconda3/envs/th21_ds/include \
-fPIC \
-v \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/cutlass/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/TH \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/THC \
-I/usr/local/CUDA/12.1.0/include \
-I/usr/local/cuDNN/8.9.2/CUDA-12/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/include/python3.11 \
-c /spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/flash_api.cpp \
-o /spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/flash_api.o \
-O3 -std=c++17 -fPIC -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' \
'-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=flash_attn_2_cuda -D_GLIBCXX_USE_CXX11_ABI=0 \
&

/usr/local/CUDA/12.1.0/bin/nvcc  \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/cutlass/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/TH \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/THC \
-I/usr/local/CUDA/12.1.0/include \
-I/usr/local/cuDNN/8.9.2/CUDA-12/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/include/python3.11 \
-c /spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu \
-o /spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.o \
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
--expt-relaxed-constexpr --compiler-options -fPIC -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ \
-U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
--expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_80,code=sm_80 --threads 4 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' \
'-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=flash_attn_2_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin /usr/local/GCC/${GCC_VERSION}/bin/gcc \
&


/usr/local/CUDA/12.1.0/bin/nvcc  \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/cutlass/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/TH \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/THC \
-I/usr/local/CUDA/12.1.0/include \
-I/usr/local/cuDNN/8.9.2/CUDA-12/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/include/python3.11 \
-c /spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.cu \
-o /spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.o \
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
--expt-relaxed-constexpr --compiler-options -fPIC -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda \
--use_fast_math -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 --threads 4 \
-DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' \
'-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=flash_attn_2_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin /usr/local/GCC/${GCC_VERSION}/bin/gcc \
&

# sm70 fwd hdim=128
/usr/local/CUDA/12.1.0/bin/nvcc  \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/cutlass/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/TH \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/THC \
-I/usr/local/CUDA/12.1.0/include \
-I/usr/local/cuDNN/8.9.2/CUDA-12/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/include/python3.11 \
-c /spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm70.cu \
-o /spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm70.o \
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
--expt-relaxed-constexpr --compiler-options -fPIC -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda \
--use_fast_math -gencode arch=compute_70,code=sm_70 --threads 4 \
-DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' \
'-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=flash_attn_2_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin /usr/local/GCC/${GCC_VERSION}/bin/gcc



# sm70 fwd hdim=64
/usr/local/CUDA/12.1.0/bin/nvcc  \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/cutlass/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/TH \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/THC \
-I/usr/local/CUDA/12.1.0/include \
-I/usr/local/cuDNN/8.9.2/CUDA-12/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/include/python3.11 \
-c /spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm70.cu \
-o /spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm70.o \
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
--expt-relaxed-constexpr --compiler-options -fPIC -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda \
--use_fast_math -gencode arch=compute_70,code=sm_70 --threads 4 \
-DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' \
'-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=flash_attn_2_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin /usr/local/GCC/${GCC_VERSION}/bin/gcc


# debug
/usr/local/CUDA/12.1.0/bin/nvcc  \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/cutlass/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/TH \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/THC \
-I/usr/local/CUDA/12.1.0/include \
-I/usr/local/cuDNN/8.9.2/CUDA-12/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/include/python3.11 \
-o test.exe \
test1.cu

# sm80 fwd hdim=64
/usr/local/CUDA/12.1.0/bin/nvcc  \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/cutlass/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/TH \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/THC \
-I/usr/local/CUDA/12.1.0/include \
-I/usr/local/cuDNN/8.9.2/CUDA-12/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/include/python3.11 \
-c /spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.cu \
-o /spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_fwd_hdim64_fp16_sm80.o \
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
--expt-relaxed-constexpr --compiler-options -fPIC -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda \
--use_fast_math -gencode arch=compute_80,code=sm_80 --threads 4 \
-DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' \
'-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=flash_attn_2_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin /usr/local/GCC/${GCC_VERSION}/bin/gcc


/usr/local/CUDA/12.1.0/bin/nvcc  \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/cutlass/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/TH \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/THC \
-I/usr/local/CUDA/12.1.0/include \
-I/usr/local/cuDNN/8.9.2/CUDA-12/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/include/python3.11 \
-c /spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.cu \
-o /spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.o \
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
--expt-relaxed-constexpr --compiler-options -fPIC -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda \
--use_fast_math -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 --threads 4 -DTORCH_API_INCLUDE_EXTENSION_H \
'-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' \
-DTORCH_EXTENSION_NAME=flash_attn_2_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin /usr/local/GCC/${GCC_VERSION}/bin/gcc \
&

/usr/local/CUDA/12.1.0/bin/nvcc  \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/cutlass/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/TH \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/THC \
-I/usr/local/CUDA/12.1.0/include \
-I/usr/local/cuDNN/8.9.2/CUDA-12/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/include/python3.11 \
-c /spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.cu \
-o /spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.o \
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
--expt-relaxed-constexpr --compiler-options -fPIC -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda \
--use_fast_math -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 --threads 4 -DTORCH_API_INCLUDE_EXTENSION_H \
'-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' \
-DTORCH_EXTENSION_NAME=flash_attn_2_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin /usr/local/GCC/${GCC_VERSION}/bin/gcc \
&


/usr/local/CUDA/12.1.0/bin/nvcc  \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/cutlass/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/TH \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/THC \
-I/usr/local/CUDA/12.1.0/include \
-I/usr/local/cuDNN/8.9.2/CUDA-12/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/include/python3.11 \
-c /spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.cu \
-o /spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.o \
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
--expt-relaxed-constexpr --compiler-options -fPIC -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ \
-U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr \
--expt-extended-lambda --use_fast_math -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 --threads 4 \
-DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' \
'-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=flash_attn_2_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin /usr/local/GCC/${GCC_VERSION}/bin/gcc \
&

/usr/local/CUDA/12.1.0/bin/nvcc  \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src \
-I/spin1/home/linux/zhongz2/flash-attention/csrc/cutlass/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/TH \
-I/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/include/THC \
-I/usr/local/CUDA/12.1.0/include \
-I/usr/local/cuDNN/8.9.2/CUDA-12/include \
-I/data/zhongz2/anaconda3/envs/th21_ds/include/python3.11 \
-c /spin1/home/linux/zhongz2/flash-attention/csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.cu \
-o /spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.o \
-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
--expt-relaxed-constexpr --compiler-options -fPIC -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda \
--use_fast_math -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 --threads 4 -DTORCH_API_INCLUDE_EXTENSION_H \
'-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' \
-DTORCH_EXTENSION_NAME=flash_attn_2_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -ccbin /usr/local/GCC/${GCC_VERSION}/bin/gcc \
&

exit;

/usr/local/GCC/${GCC_VERSION}/bin/g++ \
-v \
-shared -Wl,-rpath,/data/zhongz2/anaconda3/envs/th21_ds/lib \
-Wl,-rpath-link,/data/zhongz2/anaconda3/envs/th21_ds/lib \
-L/data/zhongz2/anaconda3/envs/th21_ds/lib \
-Wl,-rpath,/data/zhongz2/anaconda3/envs/th21_ds/lib \
-Wl,-rpath-link,/data/zhongz2/anaconda3/envs/th21_ds/lib \
-L/data/zhongz2/anaconda3/envs/th21_ds/lib \
/spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/flash_api.o \
/spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_bwd_hdim128_bf16_sm80.o \
/spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_bwd_hdim128_fp16_sm80.o \
/spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.o \
/spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_fwd_hdim128_fp16_sm80.o \
/spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_fwd_split_hdim128_bf16_sm80.o \
/spin1/home/linux/zhongz2/flash-attention/build/temp.linux-x86_64-cpython-311/csrc/flash_attn/src/flash_fwd_split_hdim128_fp16_sm80.o \
-L/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/lib \
-L/usr/local/CUDA/12.1.0/lib64 \
-L/usr/local/cuDNN/8.9.2/CUDA-12/lib64 \
-lc10 -ltorch -ltorch_cpu -ltorch_python -lcudart -lc10_cuda -ltorch_cuda \
-o build/lib.linux-x86_64-cpython-311/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so 

