

#include "cute/algorithm/copy.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// #include <torch/torch.h>
// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/detail/CUDAHooks.h>
// #include <torch/nn/functional.h>
// #include <c10/cuda/CUDAGuard.h>


#include <cute/tensor.hpp> 

using namespace cute;


constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k;

    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window size
    int window_size_left, window_size_right;

    // // Random state.
    // at::PhiloxCudaState philox_args;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;
    bool is_causal;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    bool is_rotary_interleaved;

    int num_splits;  // For split-KV version

    void * __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;
};




template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::half_t>
struct Flash_kernel_traits {

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
    using Element = elem_type;
    static constexpr bool Has_cp_async = true;
#else
    using Element = cutlass::half_t;
    static constexpr bool Has_cp_async = false;
#endif

    using ElementAccum = float;
    using index_t = int64_t;

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
#pragma message("cuda arch >= 800")
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::half_t>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
    >;
// #else 
//     using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#elif defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ == 750
#pragma message("cuda arch == 750")
    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
#elif defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ == 700
#pragma message("cuda arch == 700")
    using MMA_Atom_Arch = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;
#else
#pragma message("cuda arch < 700")
    using MMA_Atom_Arch = MMA_Atom<SM61_DP4A>;
#endif

#if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 750
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;
#else
    using SmemCopyAtom = Copy_Atom<DefaultCopy, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<DefaultCopy, elem_type>;
#endif
};

// If Share_Q_K_smem is true, that forces Is_Q_in_regs to be true
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false, typename elem_type=cutlass::half_t,
         typename Base=Flash_kernel_traits<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;   // 4
    static constexpr int kNThreads = kNWarps * 32;  // 4*32 = 128

    static constexpr int kBlockM = kBlockM_;  // 128
    static constexpr int kBlockN = kBlockN_;  // 64
    static constexpr int kHeadDim = kHeadDim_;  // 64
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;  // 64
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32); // 64
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;  // 3

    // 16x8x16, 4x1x1, 64x16x16
    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _16>>;

    // <3x3x3> <8,64> <64,1>
    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    // https://github.com/ColfaxResearch/cutlass-kernels/blob/a222587e6d59b93ba704853d3946fb686d8b8892/src/fmha/fmha_forward.cu#L434
    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 3, 3>{},
                    Layout<Shape<Int<8>, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemSize = Share_Q_K_smem ? std::max(kSmemQSize, kSmemKVSize) : kSmemQSize + kSmemKVSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per read
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));  // Val layout, 8 vals per store

    using GmemLayoutAtomOaccum = std::conditional_t<
        kBlockKSmem == 32,
        Layout<Shape <_16, _8>,  // Thread layout, 8 threads per row
               Stride< _8, _1>>,
        Layout<Shape <_8, _16>,  // Thread layout, 16 threads per row
               Stride< _16, _1>>
    >;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    using GmemLayoutAtomRotcossin = GmemLayoutAtom;
    using GmemTiledCopyRotcossin = decltype(
        make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per load
    using GmemTiledCopyRotcossinCont = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtomRotcossin{},
                        Layout<Shape < _1, _8>>{}));  // Val layout, 8 vals per load
};



/// Simple copy kernel.
//
// Uses local_partition() to partition a tile among threads arranged as (THR_M, THR_N).
template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout) 
{
  using namespace cute;

  // Slice the tiled tensors
  Tensor tile_S = S(make_coord(_,_), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_,_), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)

  // Construct a partitioning of the tile among threads with the given thread arrangement.

  // Concept:                       Tensor    Layout          Index
  Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{}, threadIdx.x);
  Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{}, threadIdx.x);

  // Construct a register-backed Tensor with the same shape as each thread's partition
  auto fragment = make_fragment_like(thr_tile_S);

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(thr_tile_S, fragment);
  copy(fragment, thr_tile_D);
}

/// Vectorized copy kernel.
///
/// Uses `make_tiled_copy()` to perform a copy using vector instructions. This operation
/// has the precondition that pointers are aligned to the vector size.
///
template <class TensorS, class TensorD, class ThreadLayout, class VecLayout>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, ThreadLayout, VecLayout) 
{
  using namespace cute;
  using Element = typename TensorS::value_type;

  // Slice the tensors to obtain a view into each tile.
  Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)

  // Define `AccessType` which controls the size of the actual memory access.
  using AccessType = cutlass::AlignedArray<Element, size(shape(VecLayout{}))>;

  // A copy atom corresponds to one hardware memory access.
  using Atom = Copy_Atom<UniversalCopy<AccessType>, Element>;

  // Construct tiled copy, a tiling of copy atoms.
  //
  // Note, this assumes the vector and thread layouts are aligned with contigous data
  // in GMEM. Alternative thread layouts are possible but may result in uncoalesced
  // reads. Alternative vector layouts are also possible, though incompatible layouts
  // will result in compile time errors.
  auto tiled_copy =
    make_tiled_copy(
      Atom{},                       // access size
      ThreadLayout{},               // thread layout
      VecLayout{});                 // vector layout (e.g. 4x1)

  // Construct a Tensor corresponding to each thread's slice.
  auto thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  Tensor thr_tile_S = thr_copy.partition_S(tile_S);
  Tensor thr_tile_D = thr_copy.partition_D(tile_D);

  // Construct a register-backed Tensor with the same shape as each thread's partition
  auto fragment = make_fragment_like(thr_tile_D);

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(tiled_copy, thr_tile_S, fragment);
  copy(tiled_copy, fragment, thr_tile_D);
}

/// Helper to convert a shape to a dim3
template <class Shape>
dim3 shape_to_dim3(Shape shape)
{
  using namespace cute;

  CUTE_STATIC_ASSERT_V(rank(shape) <= Int<3>{});
  auto result = append<3>(product_each(shape), 1u);

  return dim3(get<0>(result), get<1>(result), get<2>(result));
}



template<bool Varlen=true, int seqlen_q=1024, int seqlen_k=1024>
struct BlockInfo {

    __device__ BlockInfo(const int bidb)
        : sum_s_q(-1)
        , sum_s_k(-1)
        , actual_seqlen_q(seqlen_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , seqlen_k_cache(seqlen_k)
        , actual_seqlen_k(seqlen_k_cache)
        {
        }

    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride : uint32_t(sum_s_k) * row_stride;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    const int seqlen_k_cache;
    const int actual_seqlen_k;
};



template<typename Kernel_traits>
inline __device__ void compute_attn_1rowblock(float* Q, float* K, float* V, float* O, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;//128
    constexpr int kBlockN = Kernel_traits::kBlockN;//64
    constexpr int kHeadDim = Kernel_traits::kHeadDim;//64
    constexpr int kNWarps = Kernel_traits::kNWarps;//4

    // params.q_ptr = q.data_ptr();
    // params.k_ptr = k.data_ptr();
    // params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    //2097152, 2048, 64, 1
    int seqlen_q = 1024;
    int q_row_stride = 1;
    int k_row_stride = 1;
    int v_row_stride = 1;
    int q_head_stride = 64;
    int k_head_stride = 64;
    int v_head_stride = 64;
    // params.o_ptr = out.data_ptr();
    int o_row_stride = 1;
    int o_head_stride = 64;

        int q_batch_stride = 2097152;
        int k_batch_stride = 2097152;
        int v_batch_stride = 2097152;
        int o_batch_stride = 2097152;
        

    const BlockInfo</*Varlen=*/false> binfo(bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

        const index_t row_offset_o = binfo.q_offset(o_batch_stride, o_row_stride, bidb)
            + m_block * kBlockM * o_row_stride + bidh * o_head_stride;
        const index_t row_offset_lse = (bidb * kHeadDim + bidh) * seqlen_q + m_block * kBlockM;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(O) + row_offset_o),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(o_row_stride, _1{}));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));

        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tidx);
        Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

}

template<typename Kernel_traits>
inline __device__ void compute_attn(float* Q, float* K, float* V, float* O) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    compute_attn_1rowblock<Kernel_traits>(Q, K, V, O, bidb, bidh, m_block);
}

template<typename Kernel_traits>
__global__ void flash_fwd_kernel(float* Q, float* K, float* V, float* O) {
    compute_attn<Kernel_traits>(Q, K, V, O);
}



int main() {


  using namespace cute;
  using Element = float;

    const int batch_size = 2;
    const int seq_len = 1024;
    const int num_heads = 32;
    const int head_dim = 64;
    const int seqlen_q = seq_len;
    const int seqlen_k = seq_len;
    const int seqlen_q_rounded = seqlen_q;
    const int seqlen_k_rounded = seqlen_k;
    const int num_heads_k = head_dim;
    const int head_size = head_dim;
    const int head_size_rounded = head_size;

  auto tensor_shape1 = make_shape(batch_size, seqlen_q, num_heads, head_dim);
  thrust::host_vector<Element> h_Q(size(tensor_shape1));
  thrust::host_vector<Element> h_K(size(tensor_shape1));
  thrust::host_vector<Element> h_V(size(tensor_shape1));
  thrust::host_vector<Element> h_O(size(tensor_shape1));

  for (size_t i = 0; i < h_Q.size(); ++i) {
    h_Q[i] = static_cast<Element>(i);
    h_K[i] = static_cast<Element>(i);
    h_V[i] = static_cast<Element>(i);
    h_O[i] = static_cast<Element>(i);
  }

  thrust::device_vector<Element> d_Q = h_Q;
  thrust::device_vector<Element> d_K = h_K;
  thrust::device_vector<Element> d_V = h_V;
  thrust::device_vector<Element> d_O = h_O;

    using Kernel_traits = Flash_fwd_kernel_traits<32, 128, 64, 4>;
    // typename Kernel_traits::TiledMma tiled_mma;
    // int tidx = 10;
    // auto thr_mma = tiled_mma.get_thread_slice(tidx);

    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    printf("smem_size = %d\n", smem_size);

    const int num_m_block = (seq_len + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, batch_size, num_heads);
    printf("%d, %d, %d\n", num_m_block, batch_size, num_heads);

    auto kernel = &flash_fwd_kernel<Kernel_traits>;
    kernel<<<grid, Kernel_traits::kNThreads, smem_size>>>(d_Q.data().get(), d_K.data().get(), d_V.data().get(), d_O.data().get());



    return 0;
}























