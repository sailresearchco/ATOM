# PD index staging 重打包路径优化计划

**状态：待启动。前置条件 = 当前 DCP+MTP / PD 精度测试确认无问题。**

优化对象是 PD 分离下 producer 侧的 DSA index page 重打包路径，核心函数
`gather_dcp_preshuffled_index_pages`（`atom/model_ops/attentions/aiter_mla.py:156`）
及其调用链。这份文档记录动手前已经想清楚的部分，避免回来时重新推导。

相关背景 review：`cmq_docs/review_dcp_mtp_sparse_indexer.md` 第九节（PD 分离场景）。

---

## 0. 启动前必须先满足的条件

| # | 条件 | 说明 |
|---|---|---|
| 0.1 | 精度测试通过 | 当前正在跑。**没通过之前不要动这段代码** —— 否则精度问题和优化改动会纠缠在一起，无法二分 |
| 0.2 | 清掉 `# region agent log` 调试块 | `mooncake_connector.py:1819-1858` 还有一段硬编码绝对路径 `/it-share/.../.cursor/debug-187c06.log` 的调试写文件代码（review P3）。它在热路径上、多 rank 并发写同一文件，必须先删，否则性能数字全不可信 |
| 0.3 | 建立字节级验证门 | 见第 6 节。这是所有优化的唯一验收标准 |

---

## 1. 结论先行：这段代码是 launch-bound，不是 bandwidth-bound

单层单 chunk 的实际工作量：

```
256 页 × 2304 B = 576 KiB 读 + 576 KiB 写 ≈ 1.1 MiB
```

即使算上 16 字节粒度 gather 带来的约 25% 访存效率（MI300 cacheline 128 B），
HBM 时间也就 ~1 µs 量级。

而单次调用打出的 kernel 数：

```
zero_ / arange / mul / add / div / remainder / lt / clamp_max
/ as_tensor(H2D) / index(scheduler_blocks) / div / remainder
/ gather(keys) / logical_not / masked_fill_ / copy_(strided)
/ gather(scales) / logical_not / masked_fill_ / copy_
≈ 20 次 launch + 1 次 pageable H2D
```

按 ~5 µs/launch 估算是 ~100 µs，**比真实数据搬运重两个数量级**。

而 `_execute_staged_index_transfer` 对 **21 个 index 层各跑一遍**
（GLM-5.2 full-index layer 数），每 chunk 约 400 次 launch。

> 注意：上面 20 launch × 5 µs 是经验估算，不是实测。动手前先抓一段发送窗口的
> trace 确认（第 9 节推进顺序里的第 2 步），有可能瓶颈实际在
> `torch.as_tensor(list, device='cuda')` 的 pageable H2D 隐式同步上 ——
> 那样优先级 1 的收益比预期更大，优先级 3 可以直接跳过。

**所有优化方向都应该冲着减少 launch 和 Python 开销，而不是"把 gather 写快"。**

---

## 2. 必须记住的布局事实（推导优化的前提）

`source = runner.index_cache[layer]` 的分配形状（`aiter_mla.py:1284-1291`）是
`[num_physical_blocks, physical_block_size, aligned_index_dim]`，
但**所有真实读写方都不按这个形状看它**。`deepseek_v2.py:1463-1466` 每次读写前
重新 view 到 `[-1, kv_cache_block_size, aligned_index_dim]`，且所有写入
（`indexer_k_quant_and_cache` / `indexer_qk_rope_quant_and_cache`）和读取
（`cp_gather_indexer_k_quant_cache`）都硬编码 `preshuffle=True`。

真正的不变量：**一个 scheduler block（`kv_cache_block_size` 个 token）
= 一个 preshuffle 单元**，页内 plane 分离。以 `--block-size 16`、
`index_head_dim=128`、`aligned_index_dim=144` 为例，一页 2304 字节：

| 字节区间 | 内容 |
|---|---|
| `[0, 2048)` | key plane，16 token × 128 B fp8 key，按 MFMA 16×16 tile 重排 |
| `[2048, 2112)` | scale plane，16 个 fp32 per-token 量化 scale |
| `[2112, 2304)` | padding（`aligned_index_dim` 132 → 144 产生） |

key plane 内 token `t`、维度 `c` 的字节位于 `(c//16)*256 + t*16 + (c%16)`
—— column-tile 大步、token 中步、tile 内列小步。**最小连续单元是 16 字节**，
这是第 4 节 `index_select` 方案的基础。

`physical_block_size`（`ATOM_MLA_PAGE_SIZE`，当前配置 = 1）与 preshuffle 粒度
**无关**。上一版代码用 `source.shape[1] == 1` 判断源布局，正是踩了这个坑
（review PD-P0，已修）。

---

## 3. 优先级 1：把 per-chunk 索引计算从 per-layer 循环里提出来

**投入产出比最高的一条。**

函数第 215-230 行（`token_count` → `src_token_in_tile`）全部只依赖
`(src_block_ids, dcp_size, dcp_rank, scheduler_block_size)`，**不依赖 `source`**。
21 个层算出来的索引张量完全一样，现在被重复算了 21 遍，包括那次
Python list → CPU tensor → H2D。

做法两步：

1. 抽出 `build_dcp_index_gather_plan(src_block_ids, dcp_size, dcp_rank,
   scheduler_block_size, ...)`，返回小 dataclass（源块 id / token tile /
   tile 内 token 等索引张量 + `dst_pages`）。
   `gather_dcp_preshuffled_index_pages` 改成吃这个 plan。

2. **反转 `mooncake_connector.py:1765` 的循环嵌套。** 现在是 region 外层、
   chunk 内层（chunk 循环在 `_execute_staged_index_transfer:1798` 内部），
   所以同一个 chunk 的 plan 被 21 个 region 各建一次。改成 chunk 外层、
   region 内层，plan 自然只建一次。`staged_regions` 已经是一个 list，
   只需把 chunk 循环上提，改动局部。

   *退路*：不动连接器结构，在闭包里按 chunk 记忆化。但当前嵌套顺序下单条缓存
   必然全 miss（region 0 的 chunk 0..N，然后 region 1 的 chunk 0..N），
   得缓存 `num_chunks` 条，反而更绕。**建议直接反转循环。**

预期效果：索引侧 launch 与 H2D 从 21× 降到 1×。

---

## 4. 优先级 2：用闭式解消掉 `valid` mask 和 `zero_()`

两个都是"GPU 在算一件 host 上一行就能算出来的事"。

### 4.1 `valid` 其实是个前缀

`src_ordinal[i] = (i·W + r) // B` 对 `i` 单调不减，一旦为假就永远为假。所以

```python
valid_tokens = cdiv(src_count * B - r, W)   # clamp 到 [0, dst_pages * B]
```

核验：`src_count=5, W=4, r=1, B=16` → `cdiv(79, 4) = 20`；
而 `i=19 → g=77 → ordinal 4 < 5` 有效，`i=20 → g=81 → ordinal 5` 越界。对上了。

收益：干掉 `lt`、`clamp_max`、两次 `logical_not`、两次 `masked_fill_`，
而且只需 gather 前 `valid_tokens` 个 token。

### 4.2 `output.zero_()` 是纯浪费

key plane 和 scale plane 随后被 100% 覆盖写，唯一真正需要零的是 padding 区
（2304 中的 192 字节），而 padding **没有任何人会写**，一次性零就够。

- 前提：`aiter_mla.py:1475` 的 `torch.empty` → `torch.zeros`
  （顺便不再往线上发未初始化字节，对 bit-compare 调试友好）。
- 改用前缀后，残留脏数据只可能出现在**最后一页**的尾部：
  `dst_pages = ceil(src_count/W)` 保证 `(dst_pages-1)·W < src_count`，
  所以前面的页一定是满的。每次只需清最后一页那 2304 字节。
- **实现时把这个不等式写成断言，不要只写注释。**

注意：plane 分离布局下"某个 token 的字节"不是连续尾部，所以不能只清一小段；
清整个最后一页（2304 B）是简单且正确的做法。

预期效果：从 ~20 launch 降到 ~6，另省每层每 chunk 576 KiB 的写。

---

## 5. 优先级 3：融合成一次 gather

现在 key plane 走「gather → masked_fill → permute 后 strided copy」三趟，
中间物化 512 KiB 的 `selected_keys`。两条路：

### (a) 纯 torch，一次 `index_select`（建议先做这个）

key plane 最小连续单元是 16 字节，源和目标都可以 view 成 `[N, 16]` 的行。
预先算好"目标行序 → 源行号"的 int32 索引，一次 `index_select` 写出整个
key plane，scale 再一次。共 2 个 kernel，无中间张量，无 permute。

索引张量大小 `dst_pages · B · column_tiles = 256 · 16 · 8 = 32768` 项 int32
= 128 KiB，属于优先级 1 的 plan，跨层复用。

### (b) 一个 Triton kernel

每个 program 处理一个 `(page, token)`，源偏移纯算术推出，key 和 scale 一起搬，
不物化任何索引。1 个 kernel。和 `atom/utils/block_convert.py` 的
`kv_indices_generate_triton` 同一套路数，符合代码库习惯。

**建议顺序**：先 (a)，改动小、纯 torch、CPU 可测，先把正确性用测试钉死。
到 (a) 之后单次调用只剩 3-4 个 launch，(b) 的增量收益有限，作为可选项。

---

## 6. 优先级 4：签名和校验瘦身

- **删掉 `block_ratio` 参数。** 它和 `source.shape[1]` 冗余
  （`block_ratio = scheduler_block_size // source.shape[1]`）。调用点直接传
  `source.view(-1, B, aligned_dim)`，函数内不再关心物理页粒度。
  9 参数降到 7，同时消掉"传进来的 source 到底是哪个粒度"这类误用
  —— 这正是 PD-P0 的土壤。
- **5 个 `raise ValueError` 全是配置不变量**（`B % 16`、`head_dim % 16`、
  页大小匹配、quant 整除、`shape[0] % ratio`），运行期不变。挪到 pool 分配处
  校验一次，热路径别每层每 chunk 跑一遍 Python 比较和 f-string 准备。
- **`source.view(torch.uint8).reshape(...)` 预先算好。** init 时把每层的
  `[num_scheduler_blocks, page_bytes]` uint8 视图存成 list，热路径直接下标。
- **命名与 docstring：**
  - `scheduler_blocks` 装的是"每个目标 token 对应的源块 id"，
    改成 `src_block_id_per_token` 之类。
  - docstring 里 "Prefill commonly allocates one-token physical pages"
    会让人误以为源布局取决于物理页大小 —— 这正是把上一版引到 PD-P0 的错觉。
    删掉 `block_ratio` 后一起重写。

---

## 7. 优先级 5：顺手能修的正确性/健壮性问题

**这几条和性能优化互不依赖，应该单独一个 commit** —— 混在一起会让第 8 节的
字节比对失去意义。

| # | 问题 | 位置 |
|---|---|---|
| 7.1 | `token_tiles > 1`（`--block-size 32/64`）时 key plane 外层 tile 顺序是**未验证的假设**。`B=16` 下 `token_tiles == 1`，这个维度退化掉了，现跑的配置覆盖不到。要么加显式 raise，要么补测试 | `aiter_mla.py:227-244` |
| 7.2 | scale plane 字节数两套算法：`KVTransferRegion` 用 `tokens_per_page * 4`，函数用 `scales_per_token * 4`，仅在 `index_head_dim == 128` 时相等（review PD-P5）。抽共用 helper | `aiter_mla.py:1404` vs `:246-248` |
| 7.3 | `torch.cuda.current_stream().synchronize()` 在 worker 线程上没设当前设备，`local_rank != 0` 时 fence 错设备，NIC 可能读到写一半的 staging（review PD-P2）。改显式设备 + `torch.cuda.Event` | `mooncake_connector.py:1866`（另有 `:2006`） |
| 7.4 | staging pool 分配条件与 PD 角色无关（`hasattr(runner,"index_cache") and dcp_world_size == 1`），任何 dcp=1 的 DSA 部署都白占 9.4 MiB 并注册 RDMA（review PD-P3）。用现成的 `_replicated_index_cache_transfer_supported(config)`（`aiter_mla.py:263`）门控 | `aiter_mla.py:1462-1490` |

7.3 换成 event 之后顺便打开了流水的门：pool 有 16 个 slot（`aiter_mla.py:1467`），
现在完全没用上，可以做到"gather 第 L+1 层 / RDMA 第 L 层"重叠。这算优先级 6，
在前面几条落地并测稳之后再看。

---

## 8. 验证方案（动手第一步）

`cmq_scripts/verify_index_staging_layout.py` 已经是一个纯 CPU 的字节级参考实现
对照，用 `ast.unparse` 单独 exec 这个函数，**不需要 GPU 和 aiter**。

计划：

1. 提成 `tests/test_dcp_index_staging.py`，让 CI 能跑
   （对照 review P4：现有新测试 GPU-gated，CI 永远 skip）。
2. 参数化覆盖 `(W, r, B, HD, src_count)` 组合，**必须包含不满页的情形**
   （`src_count % W != 0`）。
3. 之后每一步优化都用同一条标准卡：**改动前后 staging 的字节逐位相同**。

### 必须澄清的前提：端到端分数证明不了任何事

按 review 第 9.2 节，这条 PD 路径目前**没有任何有效的端到端验证**。
GSM8K 0.95 是因为 ctx=948 < topk 预算 2048，top-k 把整个上下文全选进来，
**index cache 内容压根不参与决策**。所以"优化前后端到端分数不变"是空的，
字节比对是唯一可靠的门。

如果要真正做端到端验证，得先构造 **ctx > 2048** 的负载。
这一条对当前正在跑的精度测试同样适用 —— 值得先确认那批测试的 ctx 分布。

---

## 9. 推进顺序建议

```
0. 精度测试通过 + 清掉 agent log 调试块
1. 第 8 节的验证基建（pytest 化）           ← 先做这个
2. 抓 trace 确认瓶颈确实在 launch 上
3. 优先级 1 + 2（最局部、收益最确定、不碰布局假设）
4. 优先级 5（独立 commit）
5. 优先级 3(a) + 优先级 4
6. 视情况：优先级 3(b) Triton、event 流水
```

每步之间跑一遍字节比对测试。
