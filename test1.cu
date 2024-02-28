
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

int main() {
//   Layout<Shape <_4, _2>,
//          Stride<_1,_16>> ThrID;
//   Layout<Shape <_8,_4>,
//                          Stride<_1,_8>> ALayout;
//   Layout<Shape <_8,_4>,
//                          Stride<_1,_8>> BLayout;
//   Layout<Shape <Shape <_2, _2,_2>,Shape <_2,_2, _2>>,
//                          Stride<Stride<_1,_16,_4>,Stride<_8,_2,_32>>> CLayout;

//     print(ThrID.layout());printf("\n");
//     print(ALayout.layout());printf("\n");
//     print(BLayout.layout());printf("\n");
//     print(CLayout.layout());printf("\n");
    
//     print_layout(ThrID);printf("\n");
//     print_layout(ALayout);printf("\n");
//     print_layout(BLayout);printf("\n");
//     print_layout(CLayout);printf("\n");

// #if defined(__CUDA_ARCH__) &&  __CUDA_ARCH__ >= 800
// #pragma message("cudaarch==800")
// #pragma message "The value of ABC: " XSTR(__CUDA_ARCH__)
//        printf("using 800");
//        using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
// #else 
// #pragma message("cudaarch==700")
// #pragma message "The value of ABC: " XSTR(__CUDA_ARCH__)
//        printf("using 700");
//        using MMA_Atom_Arch = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;
// #endif
// using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

       /*
              struct MMA_Traits<SM70_8x8x4_F32F16F16F32_TN>
              {
              using ValTypeD = float;
              using ValTypeA = half_t;
              using ValTypeB = half_t;
              using ValTypeC = float;

              using Shape_MNK = Shape<_8,_8,_4>;
              using ThrID   = Layout<Shape <_4, _2>,
                                   Stride<_1,_16>>;
              using ALayout = Layout<Shape <_8,_4>,
                                   Stride<_1,_8>>;
              using BLayout = Layout<Shape <_8,_4>,
                                   Stride<_1,_8>>;
              using CLayout = Layout<Shape <Shape <_2, _2,_2>,Shape <_2,_2, _2>>,
                                   Stride<Stride<_1,_16,_4>,Stride<_8,_2,_32>>>;
              };

              struct MMA_Traits<SM80_16x8x16_F32F16F16F32_TN>
              {
              using ValTypeD = float;
              using ValTypeA = half_t;
              using ValTypeB = half_t;
              using ValTypeC = float;

              using Shape_MNK = Shape<_16,_8,_16>;
              using ThrID   = Layout<_32>;
              using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2,  _2>>,
                                   Stride<Stride<_32,_1>,Stride<_16,_8,_128>>>;
              using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                                   Stride<Stride<_16,_1>,Stride<_8,_64>>>;
              using CLayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                                   Stride<Stride<_32,_1>,Stride<_16,_8>>>;
              };

using 700
((_2,_2,_2),_4,_8):((_1,_2,_4),_8,_32)
((_2,_2,_2),_4,_16):((_1,_2,_4),_8,_32)
using 800
((_2,_2),_2,_8):((_1,_2),_4,_8)
((_2,_2),_2,_16):((_1,_2),_4,_8)
       */
       const int kNWarps = 4;
#ifdef USE_SM800
#pragma message("cudaarch==800")      
       const int kBlockM = 128;
       const int kBlockN = 64; 
       const int kHeadDim = 128;
       printf("using 800\n");
       using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
       using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _16>>;
#else
#pragma message("cudaarch==700")
       const int kBlockM = 128;
       const int kBlockN = 64;
       const int kHeadDim = 128;
       printf("using 700\n");
       using MMA_Atom_Arch = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;
       using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<8 * kNWarps>, _8, _8>>;
#endif

       TiledMma tiled_mma;
       Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}); 
       Layout acc_s_layout = acc_s.layout();
       print(acc_s_layout); printf("\n");  //((_2,_2,_2),_2,_4):((_1,_2,_4),_8,_16)
       // print(acc_s); printf("\n");
#ifdef USE_SM800
       auto l = logical_divide(acc_s_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
       Layout acc_s_layout_new = make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
       print(acc_s_layout_new); printf("\n"); //((_4,_2),(_2,_4)):((_2,_8),(_1,_16))
#else
       auto l = logical_divide(acc_s_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
       Layout acc_s_layout_new = make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
       print(acc_s_layout_new); printf("\n"); //((_4,_2),(_2,_4)):((_2,_8),(_1,_16))
#endif
       Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
       Layout acc_o_layout = acc_o.layout();
       print(acc_o_layout); printf("\n");
       // print(acc_o); printf("\n");

       printf("%d, %d\n", size<0>(acc_o), size<1>(acc_o));
}