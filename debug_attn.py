import torch
import torch.nn.functional as F
import primus_turbo.pytorch as pt


def pytorch_attention(
    query,
    key,
    value,
    softmax_scale=None,
    causal=False,
    attn_mask=None,
    dropout_p=0.0,
):
    """
    使用PyTorch实现的标准attention算子
    
    Args:
        query: (batch, seqlen_q, num_heads, head_dim) 或 (batch, num_heads, seqlen_q, head_dim)
        key: (batch, seqlen_k, num_heads, head_dim) 或 (batch, num_heads, seqlen_k, head_dim)
        value: (batch, seqlen_k, num_heads, head_dim) 或 (batch, num_heads, seqlen_k, head_dim)
        softmax_scale: 缩放因子，默认为 1/sqrt(head_dim)
        causal: 是否使用causal mask (用于decoder)
        attn_mask: 自定义attention mask
        dropout_p: dropout概率
    
    Returns:
        output: 与query相同shape的输出
    """
    # 判断输入格式并转换为 (batch, num_heads, seqlen, head_dim)
    if query.dim() == 4:
        if query.shape[1] != key.shape[1]:  # 如果第二维不同，假设是 (batch, seqlen, num_heads, head_dim)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            transpose_back = True
        else:
            transpose_back = False
    else:
        raise ValueError(f"期望输入为4维张量，得到 {query.dim()} 维")
    
    batch_size, num_heads, seqlen_q, head_dim = query.shape
    seqlen_k = key.shape[2]
    
    # 设置缩放因子
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim ** 0.5)
    
    # 计算attention scores: Q @ K^T
    # (batch, num_heads, seqlen_q, head_dim) @ (batch, num_heads, head_dim, seqlen_k)
    # -> (batch, num_heads, seqlen_q, seqlen_k)
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale
    
    # 应用attention mask
    if causal:
        # 创建causal mask (下三角矩阵)
        causal_mask = torch.triu(
            torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=query.device),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
    
    if attn_mask is not None:
        attn_scores = attn_scores + attn_mask
    
    # Softmax
    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
    
    # Dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    # 转回原始dtype
    attn_weights = attn_weights.to(value.dtype)
    
    # 计算输出: attn_weights @ V
    # (batch, num_heads, seqlen_q, seqlen_k) @ (batch, num_heads, seqlen_k, head_dim)
    # -> (batch, num_heads, seqlen_q, head_dim)
    output = torch.matmul(attn_weights, value)
    
    # 如果需要转回原始格式
    if transpose_back:
        output = output.transpose(1, 2)
    
    return output


def pytorch_flash_attention(
    query,
    key, 
    value,
    softmax_scale=None,
    causal=False,
):
    """
    使用PyTorch 2.0+ 的scaled_dot_product_attention (类似Flash Attention)
    
    Args:
        query, key, value: (batch, seqlen, num_heads, head_dim)
        softmax_scale: 缩放因子
        causal: 是否使用causal mask
    
    Returns:
        output: (batch, seqlen, num_heads, head_dim)
    """
    # 转换为 (batch, num_heads, seqlen, head_dim) 格式
    q = query.transpose(1, 2).contiguous()
    k = key.transpose(1, 2).contiguous()
    v = value.transpose(1, 2).contiguous()
    
    # 设置缩放因子
    if softmax_scale is None:
        softmax_scale = 1.0 / (query.shape[-1] ** 0.5)
    
    # 使用PyTorch内置的scaled_dot_product_attention
    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
        scale=softmax_scale
    )
    
    # 转回 (batch, seqlen, num_heads, head_dim)
    output = output.transpose(1, 2)
    
    return output


def test_attn():
    query = torch.load("data-debug/query_nan_rank_0.pt")
    key = torch.load("data-debug/key_nan_rank_0.pt")
    value = torch.load("data-debug/value_nan_rank_0.pt")
    softmax_scale = query.shape[-1] ** (-0.5)
    
    print("=" * 50)
    print("输入检查:")
    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Query has NaN: {query.isnan().any()}")
    print(f"Key has NaN: {key.isnan().any()}")
    print(f"Value has NaN: {value.isnan().any()}")
    
    # 测试primus_turbo的flash attention
    print("\n" + "=" * 50)
    print("测试 Primus Turbo Flash Attention:")
    o_flash = pt.ops.flash_attn_func(query, key, value, softmax_scale=softmax_scale, causal=True)
    print(f"输出 shape: {o_flash.shape}")
    print(f"输出 has NaN: {o_flash.isnan().any()}")
    
    # 测试PyTorch标准实现
    print("\n" + "=" * 50)
    print("测试 PyTorch 标准 Attention:")
    o_pytorch = pytorch_attention(query, key, value, softmax_scale=softmax_scale, causal=True)
    print(f"输出 shape: {o_pytorch.shape}")
    print(f"输出 has NaN: {o_pytorch.isnan().any()}")
    
    # 测试PyTorch 2.0+ scaled_dot_product_attention
    try:
        print("\n" + "=" * 50)
        print("测试 PyTorch F.scaled_dot_product_attention:")
        o_pytorch_sdpa = pytorch_flash_attention(query, key, value, softmax_scale=softmax_scale, causal=True)
        print(f"输出 shape: {o_pytorch_sdpa.shape}")
        print(f"输出 has NaN: {o_pytorch_sdpa.isnan().any()}")
    except Exception as e:
        print(f"\nPyTorch SDPA 测试失败: {e}")
        print("(可能需要PyTorch 2.0+)")



if __name__ == "__main__":
    # 运行主测试
    test_attn()
    