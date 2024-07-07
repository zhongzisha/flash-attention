
git clone https://github.com/NVIDIA/cutlass.git csrc/cutlass


source /data/zhongz2/anaconda3/bin/activate debug_21
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12
module load gcc/11.3.0

TORCH_CUDA_ARCH_LIST="7.0 8.0" MAX_JOBS=8 pip install -e .
export MAX_JOBS=8
TORCH_CUDA_ARCH_LIST="8.0" FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE FLASH_ATTENTION_FORCE_BUILD=TRUE pip install -e .
export MAX_JOBS=8
TORCH_CUDA_ARCH_LIST="7.0" FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE FLASH_ATTENTION_FORCE_BUILD=TRUE pip install -e .

ln -sf build/lib.linux-x86_64-cpython-311/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so .

export LD_LIBRARY_PATH=/data/zhongz2/anaconda3/envs/th21_ds/lib/python3.11/site-packages/torch/lib:/data/zhongz2/anaconda3/envs/th21_ds/lib:$LD_LIBRARY_PATH



nm -D build/lib.linux-x86_64-cpython-311/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so | c++filt | grep "Flash_fwd_params"

ldd -r build/lib.linux-x86_64-cpython-311/flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so

# 20240226, finally got the dynamic link library compiled
# the main problem is in the C++ template, it will unroll the template
# if we don't have the implementation of the functions, we need to remove the definition






