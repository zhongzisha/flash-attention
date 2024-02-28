
import torch

from flash_attn import flash_attn_func







if True:
    dtype = torch.bfloat16
    seqlen_q = seqlen_k = 1024
    d = 128
    device = "cuda"
    causal = False
    # set seed
    torch.random.manual_seed(0)
    batch_size = 8
    nheads = 32
    sm_scale = 0.5
    window_size = (-1, -1)
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    o = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    qq = q.clone().detach().transpose(1, 2)
    qq.requires_grad_(True)
    kk = k.clone().detach().transpose(1, 2)
    kk.requires_grad_(True)
    vv = v.clone().detach().transpose(1, 2)
    vv.requires_grad_(True)
    oo = o.clone().detach().transpose(1, 2)
    oo.requires_grad_(True)
    tri_out = flash_attn_func(q, k, v, 0.0, softmax_scale=sm_scale, causal=causal, window_size=window_size)
    tri_out.backward(o)
    tri_dv = v.grad.clone()
    tri_dk = k.grad.clone()
    tri_dq = q.grad.clone()

    # reference implementation
    M = torch.tril(torch.ones((seqlen_q, seqlen_q), device="cuda"))
    p = torch.matmul(qq, kk.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, vv)
    ref_out.backward(oo)
    ref_dv = vv.grad.clone().transpose(1, 2)
    ref_dk = kk.grad.clone().transpose(1, 2)
    ref_dq = qq.grad.clone().transpose(1, 2)
    ref_out = ref_out.transpose(1, 2)

    print(tri_out[:3, :3, 5, 5])
    print(ref_out[:3, :3, 5, 5])
    print(tri_dq[:3, :3, 5, 5])
    print(ref_dq[:3, :3, 5, 5])
    # assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    # assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=0)
    # assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0)
    # assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0)

