# DCP + MTP sparse indexer 叠加支持：未提交改动 review

Review 对象：working tree 中的未提交修改（分支 `Jasen/dcp-kv-transfer-pr2090`），
目标是让 DeepSeek-V3.2/GLM-5.2 的 sparse indexer（DSA）在 **DCP 分片 index cache**
下与 **MTP verify**、以及 **PD 分离**同时工作。涉及四个文件：

| 文件 | 增删 | 场景 |
|---|---|---|
| `atom/model_ops/dcp_ops.py` | +174/-24 | DCP + MTP |
| `atom/model_ops/attentions/aiter_mla.py` | +379/-0 | DCP + MTP、PD |
| `atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py` | +232/-20 | PD |
| `atom/kv_transfer/disaggregation/types.py` | +9/-1 | PD |

另有两个新增测试文件：`tests/test_dcp_mtp_index_lens.py`（MTP）、
`cmq_scripts/verify_index_staging_layout.py`（PD，本次 review 为定位 PD-P0 而写）。

## 结论

**DCP + MTP 这一半的核心数学是对的**（第一～二节），端到端也已验证。
**PD 那一半有一个确定的字节级布局错误，当前配置下 100% 命中**（第九节）。
整份 working tree 还不是可提交状态。

| # | 级别 | 场景 | 状态 | 问题 | 影响 |
|---|---|---|---|---|---|
| P0 | 阻塞 | MTP | 未修 | 依赖一份**未提交**的 aiter 本地改动（`PerQueryContext` / `[B,N]` context_lens） | 换未打补丁的 aiter → **静默算错，不报错** |
| P1 | 正确性 | MTP | 未修，**且已恶化** | draft step 的 `dcp_local_context_lens` 陈旧读；修 同源问题 #2 时拆掉了它唯一的意外护栏 | V3.2 类配置下静默丢最新 owned token，掉接受率 |
| P2 | 清理 | MTP | **已修** | `by_pos` 死代码链 → 两个函数已合并为按 `next_n` 分派的一个 | — |
| P3 | 清理 | 两者 | 7 段删到 1 段 | 剩 `dcp_ops.py` 一段（20-shot 实验的在飞仪器） | 只读文件系统上直接 `OSError`；热路径 `getattr` |
| P4 | 测试 | MTP | 未修 | 测试 GPU-gated，CI 跑不到 | P2 这次重构因此**零自动化覆盖** |
| **PD-P0** | **阻塞** | **PD** | **`gather_dcp_preshuffled_index_pages` 按错误布局读源 index cache** | **搬过去的 index page 全是错字节；已实测 32/32 token 错** |
| PD-P1 | 回归 | PD | 新增的 `consumer_block_bpb` 校验会打死 replicated index cache 路径 | 原本可用的逃生通道（`ATOM_DCP_REPLICATE_INDEX_CACHE=1`）首传即 raise |
| PD-P2 | 正确性 | PD | worker 线程上 `torch.cuda.current_stream()` fence 错设备 | local_rank≠0 时 fence 无效，NIC 可能读到写一半的 staging |
| PD-P3 | 资源 | PD | staging pool 的分配条件与 PD 角色无关 | 所有 dcp=1 的 DSA 部署都白占 9 MiB 并注册 RDMA |
| PD-P4 | 健壮性 | PD | 页数不匹配时 raise 落进宽 `except` | 退化成请求挂死等超时，而不是干净报错 |
| PD-P5 | 一致性 | PD | scale plane 字节数有两套算法，仅在 `index_head_dim==128` 时相等 | 换 head_dim 即静默错位 |

---

## 一、改动做了什么

`dcp_decode_candidate_exchange` 原来有一句硬 assert：

```python
assert attn_metadata.max_seqlen_q == 1, (
    "DCP + DeepSeek-V3.2 sparse indexer (DSA) currently supports "
    "qlen=1 decode only (MTP verify not yet supported)."
)
```

改动把它替换成 `next_n == 1` / `else` 双分支，并新增：

| 位置 | 内容 |
|---|---|
| `dcp_ops.py` | `dcp_local_context_lens(..., next_n=1)` 统一普通 decode 与 MTP，返回 request-major flat tensor |
| `dcp_ops.py:1131-1188` | MTP 分支：`[B,N]` context_lens 单次 kernel launch |
| `dcp_ops.py:1231-1244` | `top_k_per_row_decode` 的 `next_n` 参数改传常量 `1` |
| `aiter_mla.py:330-332` | `self.max_spec_q = drafter.mtp_k + 1` |
| `aiter_mla.py` | 一个泛化的 `[B*N]` `CpuGpuBuffer`（N=1 与 MTP 共用） |
| `aiter_mla.py:2093-2164` | `prepare_decode` 里 host 侧计算并发布 |
| `aiter_mla.py:2364-2370` | `build()` 挂到 attn_metadata |
| `aiter_mla.py:2686-2695` | cudagraph capture 路径挂到 attn_metadata |

配套新增 `tests/test_dcp_mtp_index_lens.py`（brute-force 参考实现对照）。

---

## 二、正确的部分（可以放心的地方）

### 2.1 per-query 本地长度的推导成立

改动的核心断言：MTP verify 下"本地 KV 长度"从 per-request 变成
per-query-token，而且**不是每个 draft position 加 1，是每 W 个位置才加 S**。

证明链：

1. 非 DCP 时 row `(b,j)` 的有效集合是全局位置 `[0, ctx-N+j]`
   —— kernel 用 `context_end = context_length - next_n + pid_next_n`。
2. DCP 时 index cache 和 main KV 共用 `_dcp_round_robin_slot`
   （`aiter_mla.py:1975-1991`），**非 owner 返回 `-1`**，所以本地 KV 就是
   "按全局位置升序紧密排列的、本 rank 拥有的 token"。
3. 因此"前 P 个全局位置里本 rank 拥有几个" = `get_dcp_local_seq_lens(P)`，
   正好是本地下标上界。
4. 于是 `visible = ctx - next_n + 1 + j` → `get_dcp_local_seq_lens(visible)`，
   配合 aiter 的 `PerQueryContext`（`context_end = context_length - 1`），
   得到的有效集合 = **非 DCP 结果在本 rank 分片上的限制**。

第 4 步的等式就是这条路该满足的不变量，改动满足它。

### 2.2 行布局一致

| 对象 | 行索引 |
|---|---|
| kernel `out_logits` | `pid_batch * next_n + pid_next_n` |
| kernel `PerQueryContext` 读 context_lens | `pid_batch * next_n + pid_next_n` |
| `local_ctx.view(batch_size, next_n)` | row-major（request-major） |
| `padded_q_fp8_decode_tokens` | `(bs, next_n, ...)` |

四者完全对齐。

### 2.3 `top_k_per_row_decode(..., 1, ...)` 是必须的

`local_ctx` 已经是 per-row 的了；如果继续传 `next_n`，kernel 会按 per-request
长度**再推一次**每行边界，正好是这条 MTP 路径要避开的 shard-unaware 算术。
对 qlen=1 分支行为不变（原来 `next_n` 恒为 1）。

### 2.4 未初始化列读不到

`PerQueryContext` 下 kernel 只写 `[0, ceil(L/ChunkK)*ChunkK)`，而
`local_logits` 是 `torch.empty`。但下游 `top_k_per_row_decode` 和
`dcp_pack_topk_candidates` 都按 `local_ctx[r]` 截断，读不到未初始化列。

### 2.5 buffer 只是快路径，取错不掉正确性

两个新 buffer 只是"host 预算好的加速路径"，不发布时 device fallback 语义相同。
所以 `max_spec_q` 取错（例如 PP 非 last rank 没有 `drafter`）**只掉性能不掉
正确性**。这个设计属性是好的，值得在 PR 描述里明确写出来。

### 2.6 端到端已验证

| 实验 | 配置 | GSM8K (200, 5-shot) |
|---|---|---|
| `debug187c06_single_launch_real_accept_20260901_021340` | DCP4, mtp_k=3, replicate=0, **spec_accept_rate=off** | **0.98** |
| `dcp4_mtp_indexer_cut_acc_20260831_120659` | 同上，但未关 synthetic acceptance | 0.33 |
| `dcp4_mtp_replicate_ctl_20260831_124421` | replicate=1（对照），未关 synthetic acceptance | 0.26 |

**注意**：Aug-31 的两次低分是 synthetic acceptance 的伪影（replicate 对照组同样低，
说明与 indexer 分片无关），不是这条路的 bug。唯一有效的读数是 0.98。

`.cursor/debug-187c06.log` 也确认新路径真的执行了：

```json
{"hypothesisId": "H2", "data": {"dcpRank": 0, "batchSize": 256, "nextN": 4,
 "contextShape": [256, 4], "kernelLaunchesPerIndexLayer": 1}}
```

4 个 rank 全部命中，`nextN=4`、`kvBlockSize=16`、`dcpInterleave=1`。

---

## 三、P0：依赖未提交的 aiter 本地改动（阻塞）

### 证据

```console
$ docker exec atom_cmq_43_pd_dev_0830 bash -lc 'cd /app/aiter-test && git status --short'
 M ops/triton/_triton_kernels/attention/pa_mqa_logits.py
 M ops/triton/attention/pa_mqa_logits.py
 M ops/triton/gluon/pa_mqa_logits.py
```

`[B, N]` context_lens / `PerQueryContext` **不在 aiter 上游**（HEAD 是
`f4e7c7509`）。`.runtime/aiter-debug/` 就是这三个文件的副本。ATOM 侧没有任何
版本门禁、能力探测或 docs 说明。

### 失败模式：静默算错

上游 `deepgemm_fp8_paged_mqa_logits` **完全没有 context_lens 形状校验**
（ndim 检查是这次一起加的）。传 `[B,N]` 进去时：

1. kernel 按 `context_len_ptr + pid_batch` 取扁平下标，request `b` 拿到的是
   `[B,N]` 展平后第 `b` 个元素 —— 即 request `b//N` 的 draft position `b%N`，
   **跨请求乱配**。
2. 取到的值可能比真实本地长度**更长**，那么 kernel 少写的列会被
   `top_k_per_row_decode`（按正确的 `local_ctx` 读）当成有效 score 读进去
   → **不确定的错误输出，无任何报错**。

这是最坏的失败模式，必须显式挡住。

### 建议

1. **aiter 侧先提 PR 合上游。** 提之前把 patch 里夹带的 isort 顺序调整
   （`from packaging.version import Version` 等几处 import 重排）剥掉 ——
   那会和 aiter 自己的 CI 打架。
2. **补齐 patch 内部一致性。** `_deepgemm_fp8_paged_mqa_logits_stage1`
   （`_triton_kernels/attention/pa_mqa_logits.py:300`）**没有**加
   `PerQueryContext`，还在用 `context_length - next_n + pid_next_n`。
   ATOM 当前不走 stage1，但谁将来切到两段式就静默错。
   varctx 那条已经用 `ValueError` 挡住了，这处理是对的，stage1 应同样处理。
3. **ATOM 侧加一次性能力探测。** 例如在 builder `__init__` 里检查
   `_compile_deepgemm_fp8_paged_mqa_logits` 的签名有没有 `PerQueryContext`，
   探测不到就 raise，把原来那句 "MTP verify not yet supported" 的语义保留成
   显式报错。**绝对不要静默退化** —— 这个 feature 的失败模式恰好是无声的。

---

## 四、P1：draft step 的 `dcp_local_context_lens` 陈旧读

### 机制

`var["dcp_local_context_lens"]` **只有 host `prepare_decode` 会刷新**。
但 MTP draft loop 里 `context_lens` 是在 **device 上** `+1` 的：

- `prepare_mtp_decode(update_context_lens=True)`（fused kernel），或
- fallback 的 `attn_metadata.context_lens[:running_bs] += 1`
  （`eagle_proposer.py:666`）

`dcp_local_context_lens` 不会跟着走。draft step 的 indexer 走 `next_n == 1`
分支，而"是否用已发布 buffer"的判据是**形状巧合**（`dcp_ops.py:960-961`）：

```python
local_ctx = getattr(attn_metadata, "dcp_local_context_lens", None)
if local_ctx is not None and local_ctx.shape[0] == num_rows:
    return local_ctx
```

一旦 `running_bs == scheduled_bs` 就命中陈旧值，本地长度比真实值短最多
`mtp_k` 个 token → 本地打分丢掉最新的 owned token。方向是"变短"，
所以**不会 fault，只掉接受率/精度**，属于最难发现的那类。

### GLM-5.2 为什么没踩到

两层原因，**都不是这份改动提供的保障**：

1. `config.json` 里 `index_share_for_mtp_iteration: true` → draft step 0 用的是
   target 的 metadata（fresh，走 MTP 分支），steps 1+ `skip_topk=True`
   **根本不跑 indexer**。
2. 即使跑，captured 的 draft mid-step 是用 `build_for_cudagraph_capture` 建的
   metadata，而那条路径**没有**设 `dcp_local_context_lens` → `None` →
   graph 里烘进去的是 device fallback，读被 device 刷新过的 `context_lens`，
   反而是对的。

> **⚠️ 第 2 条已经不成立了。** 后续修 同源问题 #2 时给 capture 路径补上了
> `attn_matadata.dcp_local_context_lens = var[...].gpu[:bs]`
> （`aiter_mla.py:2771-2773`），于是 captured draft mid-step 现在会命中
> **已发布但陈旧**的值，而不再退回 device fallback。
> 换句话说：**修 #2 顺手拆掉了 P1 唯一的意外护栏。** 现在只剩
> `index_share_for_mtp_iteration` 这一层在挡，V3.2 没有它。
> 这让下面那个 `_enter_decode_metadata` 的修法从"锦上添花"变成**必须做**。

### 谁会踩到

DeepSeek-V3.2 没有 `index_share_for_mtp_iteration`
（`_share_mtp_indices` 要求 `getattr(draft_hf, "index_share_for_mtp_iteration", False)`），
draft steps 1+ 会真的跑 qlen=1 indexer；走 eager / 没录到 recording 的 mid-step
时命中陈旧值。

HEAD 里这条路**不可达**（verify step 的 `max_seqlen_q == 1` assert 先炸），
所以是这次改动新暴露的。

### 建议

把不变量写进代码，而不是靠形状巧合：

> 在 `_enter_decode_metadata`（`eagle_proposer.py:357-435`）里显式把
> `attn_metadata.dcp_local_context_lens` 置 `None`，让 draft step 一律走
> device fallback。

fallback 本来就正确，只多 8 个 elementwise kernel，而 draft 只有 `mtp_k` 步，
代价可忽略。

另一种做法是在 `prepare_mtp_decode` 里跟着 device 重算，但那要把现在
`dcp_local_rebuild` 分支里那个 S=1 的公式推广到任意 S，多一份需要同步维护的算式。

### 同源问题（fix-then-sweep 一起扫）—— 两条都已修

1. ~~`dcp_local_context_lens` 的 fallback 返回 `attn_metadata.context_lens` 的
   **完整长度**，没有 `[:num_rows]`~~ → **已修**。签名从 `num_rows` 换成
   `(batch_size, next_n=1)`，fallback 现在取 `context_lens[:batch_size]`，
   返回长度恒为 `batch_size * next_n`。已在 CPU 上验证
   （`running_bs=6 / scheduled_bs=3, N=4` → 返回 12 行）。
2. ~~`build_for_cudagraph_capture` 从来没设过 `dcp_local_context_lens`~~
   → **已修**（`aiter_mla.py:2771-2773`），两个 buffer 在 capture 路径上行为
   一致了。**但这个修复的副作用见上面那个警告框** —— 它拆掉了 P1 的护栏。

### 新增的残留：形状判据仍是"长度相等"

合并后的判据是 `local_ctx.shape[0] == rows`（`rows = batch_size * next_n`）。
两点值得注意：

- 旧的 MTP 版用的是 `>= rows` 然后切片 `[:rows]`；现在改成 `==` 且不切片，
  所以 `running_bs > batch_size` 的 padded graph replay 会**退回 device 计算**
  而不是复用已发布 buffer。只掉性能不掉正确性，但和 docstring 宣传的
  "prefer it to avoid repeating device arithmetic" 有出入。
- 判据只比长度，不比 `(batch_size, next_n)` 的拆分方式。真实路径上 `next_n`
  就是 `max_seqlen_q`，所以拆不错；但这仍是 P1 那种"靠形状巧合"的模式。
  如果要彻底，把 `next_n` 一起发布出来做断言。

---

## 五、P2：`by_pos` 死代码链（已解决）

### 原现状

`dcp_local_context_lens_mtp` 的第二个返回值在**唯一**调用点被丢掉：

```python
local_ctx, _local_ctx_by_pos = dcp_local_context_lens_mtp(...)
```

但为它付的代价是全额的：

- 一个 `CpuGpuBuffer(max_spec_q, max_bs)`（`aiter_mla.py:522-524`）
- 每步 host 上一次转置写入（`aiter_mla.py:2162-2164`）
- 每步一次全量 `copy_to_gpu()` H2D（`aiter_mla.py:2368-2370`）
- 两处 metadata 赋值（`build()` + capture）

### 更麻烦的是注释

三处注释在讲一个**已经不成立**的故事：

| 位置 | 内容 |
|---|---|
| `dcp_ops.py` docstring | "the paged-MQA-logits kernel looks its context length up by request (`context_len_ptr + pid_batch`) and so has to be driven one draft position at a time" |
| `aiter_mla.py:506-511` | "Both layouts are published because the indexer's two consumers index differently" |
| `tests/test_dcp_mtp_index_lens.py` | 把 by_pos 语义写成断言，含 `assert by_pos[0].is_contiguous()` |

实现恰恰用了 `[B,N]` **一次** launch。这些都是前一版实现（per-position 循环、
`next_n` 次 launch）的遗留，和 CLAUDE.md 的 **name-matches-function** 规则直接
冲突 —— 下一个读代码的人会照着注释去找那个不存在的第二个消费者。

### 建议

`dcp_local_context_lens_mtp` 只返回 `per_token`，删掉 by_pos 的 buffer、
host 填充、两处 metadata 赋值和测试里对应断言，docstring 改成
"`[B,N]` PerQueryContext 一次 launch" 的真实理由。

顺手消掉一个坑：`by_pos_np[:max_seqlen_q, scheduled_bs:running_bs] = 0`
只清到 `running_bs`，但 capture 路径暴露的是**全宽** `.gpu`，将来谁真去用就会读到
`running_bs` 之后的脏数据。

### 处理结果（已完成）

做得比"删掉 by_pos"更彻底：`dcp_local_context_lens_mtp` 整个函数被**合并回**
`dcp_local_context_lens(..., batch_size, next_n=1)`，普通 decode 与 MTP 现在也
共用同一个 `dcp_local_context_lens` buffer：容量是 `max_bs * max_spec_q`，
当前有效形状是 request-major `[B*N]`。`by_pos` buffer、per-token 专用 buffer、
host 转置、重复 H2D、两处 metadata 分支、测试里的相关断言，以及讲错故事的注释
全部消失。
`dcp_decode_candidate_exchange` 里原来的 `if next_n == 1 / else` 双分支也塌成了
一次调用加一个 `local_ctx if next_n == 1 else local_ctx.view(batch_size, next_n)`。

仓库自带的两个测试在无 GPU/triton 环境下 **skip**（P4 未修），所以这次合并没有
自动化覆盖。用 `cmq_scripts/verify_merged_ctx_lens.py` 在 CPU 上补验过：

```console
formula vs brute force: 28 groups, 0 failures
next_n=1 vs pre-refactor formula: 0 mismatches
next_n>1 uses per_token buffer (zero-copy): True
next_n>1 ignores the [B] buffer:            True
next_n=1 uses the [B] buffer (zero-copy):   True
fallback honours batch_size (rows=12, want 12):  True
```

覆盖了 `W∈{2,4,8}`、`S∈{1,16,64}`、`next_n∈{1,2,3,4}`、`ctx < next_n` 的
padded 尾部，以及发布名分派和形状守卫。**`next_n=1` 与重构前的 per-request
公式逐位相同**，所以普通 decode 路径没有行为变化。

---

## 六、P3：调试代码（7 段已删 6 段，剩 1 段）

`rg -c "# region agent log"` 当前只剩：

| 文件 | 段数 | 位置 |
|---|---|---|
| `dcp_ops.py` | 1 | `dcp_local_context_lens`（`runId="gsm8k-20shot"`, `hypothesisId="H20"`） |

原先的 7 段中，`aiter_mla.py` 的 3 段和 `mooncake_connector.py` 的 2 段已删除，
`dcp_ops.py` 的 2 段被这 1 段新的取代。剩下这段看起来是为 9.7 里那个
"让 ctx > 2048 才能真正验证 indexer"的 20-shot 实验现加的在飞仪器，不是残留 ——
但提 PR 前仍要删，而且它有两个具体危害：

**1. 它会让函数在只读文件系统上直接抛异常。** 实测：

```console
OSError: [Errno 30] Read-only file system:
  '/it-share/mengqingcao/code/ATOM/.cursor/debug-187c06.log'
```

硬编码绝对路径 + 无 try/except，容器里那个目录只要不可写（沙箱、只读挂载、
换机器、换用户）整条 indexer 路径就崩。这也是为什么上面的 CPU 验证脚本必须先把
这段剥掉才能跑。

**2. 它在热路径上做函数属性探测。** `getattr(dcp_local_context_lens,
"_agent_logged_widths", set())` 每次调用都执行一遍，且**每次未命中都新建一个
`set()`**（21 个 full-index layer × 每步）。

其余问题（对已删除的那 6 段同样适用，留档）：

- 硬编码绝对路径 `/it-share/mengqingcao/code/ATOM/.cursor/debug-187c06.log`
- 从多个 GPU worker 进程并发 append 同一个文件；PD 下 prefill 和 decode
  **两个 server** 也在写同一个路径
- `prepare_decode` 里为了打日志额外算了一个 `old_implied`
- `dcp_decode_candidate_exchange` 每次调用都要 `getattr` 探一个函数属性
  （21 个 full-index layer × 每步）
- `mooncake_connector.py` 里两段都在 RDMA 发送路径上做同步文件 IO，
  且 `_agent_logged_*` 这些标记属性是 `getattr` 动态挂的，不在 `__init__` 里

（历史记录，供理解 P2 的成因）已删的 `dcp_ops.py` 那两段本身就是没收尾的痕迹 ——
log 里能看到它们在同一次调用里相隔 8ms 连续输出两个矛盾的 `runId`：

```
1788231552675  runId="pre-fix"   kernelLaunchesPerIndexLayer=4
1788231552683  runId="post-fix"  kernelLaunchesPerIndexLayer=1
```

正是从 per-position 循环切到单次 launch 之后没清场，和 P2 的 by_pos 同源。
这批代码现在已经清理干净了。

---

## 七、P4：新测试在 CI 里跑不到

`tests/test_dcp_mtp_index_lens.py` 有：

```python
if not torch.cuda.is_available():
    pytest.skip("Triton kernels need a GPU", allow_module_level=True)
```

但 CLAUDE.md 说 `python -m pytest tests/` 是 no-GPU 的（mock AITER 和
`torch.cuda`）。而 `dcp_local_context_lens` 的 fallback 是**纯 torch，没有 triton
kernel** —— 把 `DEV` 改成 `"cpu"` 就能真跑起来。

这是这次改动里唯一能纯 CPU 覆盖的新逻辑，而且它的 brute-force 参考实现
（`_owned_before`）写得很好，值得让它进 CI 而不是永远 skip。需要确认
module-level 的 `from atom.model_ops.dcp_ops import ...` 在 CI 的 mock 下能过
（`test_dcp_topk.py` 有同样的结构，可以照抄它的处理）。

---

## 八、两个不算 bug 但值得记的量级

MTP 下 indexer 的通信和显存都是 qlen=1 的 `next_n` 倍：

| 项 | 公式 | bs=256, next_n=4, W=4 |
|---|---|---|
| `send` | `[2, bs*next_n, k_loc]` fp32 | 16 MiB / 层 |
| `recv`（all_gather） | ×W | 64 MiB / 层 |
| 全模型（21 个 full-index layer） | ×21 | ~1.3 GiB / step |
| `local_logits` | `[bs*next_n, l_max]` fp32 | 4× qlen=1 |

`k_loc` 固定 `topk_tokens = 2048`（CUDAGraph 要求静态尺寸，不能用实时本地长度）。

都不是错误，但 benchmark 时应该单独看一眼这块，别把 MTP 的收益全喂给 indexer
的通信。

---

## 九、PD 分离场景：producer 侧 index page 重打包

### 9.0 这部分改动想解决什么

PD 分离下 prefill 节点（producer）算完整份 KV + DSA index cache，RDMA 推给 decode
节点（consumer）。consumer 开 DCP 时 index cache 是**按 token round-robin 分片**的，
而 index page 是 **preshuffled**（MFMA 16×16 tile + plane 分离）—— 页内没有
"按 token 寻址"的字节区间，所以没法用带 stride 的 RDMA 描述符挑出"每 W 个 token
取一个"。改动前直接报错：

> A DCP interleave of `{interleave}` shards the DSA index cache below a page,
> which its preshuffled layout does not allow. Run the decode node with
> `ATOM_DCP_REPLICATE_INDEX_CACHE=1`, or with an interleave of `{block_size}`.

新方案是 **producer 侧 GPU 重打包**：发送前把属于 consumer rank `r` 的 token 从
producer 自己那份稠密 index cache 里 gather 到一块 staging buffer，按 consumer 的页
布局排好，然后整页 RDMA。线格式重新变成页对齐的。

| 位置 | 内容 |
|---|---|
| `types.py:168-173` | `KVTransferTensors` 新增 4 个字段（`index_staging_region` / `_pool_size` / `_chunk_pages` / `gather_sharded_index`） |
| `aiter_mla.py:156-283` | 新函数 `gather_dcp_preshuffled_index_pages`，重打包本体（纯 torch） |
| `aiter_mla.py:1519-1620` | 分配 staging pool、注册 region、定义 `gather_sharded_index` 闭包 |
| `mooncake_connector.py:1256-1273` | staging slot 的 acquire/release |
| `mooncake_connector.py:1654-1690` | 路由：index region 改走 staged 路径；新增 `consumer_block_bpb` 校验 |
| `mooncake_connector.py:1799-1900` | `_execute_staged_index_transfer`：分块 gather → fence → 整页 RDMA |

验证用的拓扑（`cmq_scripts/mesh/dcp/single_node/start_glm52_pd_cpp4_dcp4_indexer_cut.sh`）：
prefill PP4 TP1 在 GPU 0-3、decode TP4 DCP4 在 GPU 4-7，两侧 `--block-size 16`、
`ATOM_MLA_PAGE_SIZE=1`、`--method mtp --num-speculative-tokens 3`，
decode 侧 `ATOM_DCP_REPLICATE_INDEX_CACHE=0`。

### 9.1 对的部分

**分块切分的下标算术是对的。** consumer 的 dst page `p` 在 rank `r` 上装本地 token
`[p·B, (p+1)·B)`，映射回全局 token 就是 `[p·B·W + r, ((p+1)·B-1)·W + r]`，
恰好横跨 `W` 个 producer scheduler block，所以

```python
src_start = dst_start * dcp_size
src_chunk = src_block_ids[src_start : min(len(src_block_ids), (dst_start + len(dst_chunk)) * dcp_size)]
```

是正确的，而且和函数内部把 `local_token` 从 0 起算的 re-base 对得上。
`global_token = local_token * W + r` 也正好是 `S=1` 时的 `dcp_global_pos`，
函数对 `interleave != 1` 显式 raise，没有偷偷放过。

**pool slot 跨 chunk 复用是安全的。** `_rdma_write_with_retry` 走的是
`batch_transfer_sync_write` —— **同步**返回，所以下一个 chunk 覆写同一个 slot 时
NIC 已经读完了。这点容易看错，值得在 PR 描述里点明。

**region → index cache 行的映射在 PP 下是对的。** 闭包里
`index_region_idx = region_idx - num_layers` 用的是 **producer-local** 编号，
而 `block_region_consumer_indices` 才负责翻译到 consumer 的全局编号，两者没串。

**PD 没有给 decode 侧的统一 buffer 带来新的陈旧读。** decode 侧仍在
`prepare_decode` 的 DCP 分支发布当前 `[B*N]`；进入 MTP draft 的 qlen=1 metadata
时则显式置 `None`，让 indexer 从已在 device 上递增的 `context_lens` 推导 live
`[B]`，避免把 verify 的 request-major `[B*N]` 前缀误当成 per-request 数据。

### 9.2 PD-P0：源 index cache 的布局读错了（阻塞）

`gather_dcp_preshuffled_index_pages` 用 `source.shape[1]` 判断源布局：

```python
source_page_size = source.shape[1]
...
if source_page_size == 1:
    source_rows = source_bytes.reshape(source.shape[0], source_page_size, aligned_index_dim)
    selected_keys = source_rows[physical_blocks, physical_token, :index_head_dim]   # ← 按 token 交错读
```

`source = runner.index_cache[layer]`，形状是
`[num_physical_blocks, physical_block_size, aligned_index_dim]`，其中
`physical_block_size = ATOM_MLA_PAGE_SIZE`（脚本里是 **1**）。于是取 `== 1` 分支，
把第 `t` 行的字节 `[0:128]` 当成 "token t 的 key"、`[128:132]` 当成它的 scale ——
这是**按 token 交错**的假设。

**但 `shape[1]` 是 MLA latent 页大小，和 index cache 的 preshuffle 粒度无关。**
ATOM 里 index cache 的逻辑页恒等于 `kv_cache_block_size`，
`deepseek_v2.py:1463-1466` 每次读写前都重新 view 一遍：

```python
index_block_size = runner_block_size * (get_dcp_world_size() if replicated_index_cache else 1)
kv_cache = kv_cache.view(-1, index_block_size, kv_cache.shape[-1])
```

而所有写入（`indexer_k_quant_and_cache` / `indexer_qk_rope_quant_and_cache`）和
所有读取（`cp_gather_indexer_k_quant_cache`、`deepgemm_fp8_paged_mqa_logits`）
都是**硬编码** `preshuffle=True` / `Preshuffle=True`。所以一个 scheduler block
（16 token、2304 字节）的真实布局是：

```
[   0, 2048)  MFMA-16x16 tile 的 key plane（16 token 全部）
[2048, 2112)  16 个 fp32 scale
[2112, 2304)  padding
```

也就是 **plane 分离**，不是按 token 交错。函数的**写**端恰好是按 plane 分离写的
（`output[:, :B*hd]` 放 key、`output[:, B*hd : B*hd+B*4]` 放 scale），
只有 `shape[1]==1` 的**读**端用了另一套假设 —— 同一个函数里两个分支对布局的
理解不一致。

#### 实测确认

`cmq_scripts/verify_index_staging_layout.py`（纯 CPU，抽出该函数单独跑）：

```console
config: B=16 HD=128 ALIGNED=144 W=4 rank=1 src_blocks=8
expected dst_pages=2

current call (shape[1]==1):        pages=2 checked=32 wrong_key_tokens=32 wrong_scale_tokens=32
re-viewed at kv_cache_block_size:  pages=2 checked=32 wrong_key_tokens=0  wrong_scale_tokens=0
```

**当前调用方式下 32 个 token 的 key 和 scale 全错**；把源按 `kv_cache_block_size`
重新 view 后全对。

上面这个对照依赖我对 tile 内部顺序的假设（取自该函数自己的 `%16==0` 分支）。
还有一个**不依赖任何 tile 顺序**的证明：`==1` 分支把 token `t` 读成源块的字节
`[t·144, t·144+128)`，而 key plane 只到 2048、scale plane 只到 2112：

```
  token  0 -> source bytes [   0,  128) : inside key plane
  token 13 -> source bytes [1872, 2000) : inside key plane
  token 14 -> source bytes [2016, 2144) : straddles key/scale planes
  token 15 -> source bytes [2160, 2288) : pure padding (reads zeros)
```

每个 scheduler block 的 token 15 读到的**必然是纯 padding（全 0）**，token 14 必然
跨 plane 边界 —— 无论 aiter 的 tile 顺序如何。

`bytesPerPage: 2304` 这个数在 `.cursor/debug-187c06.log` 里也对得上，
说明确实按 16 token/页在搬，只是页内取错了字节。

#### 为什么 GSM8K 0.95 没抓到

`debug187c06_pd_staging_v2_20260901_1110/gsm8k_l200.log` 是 0.950 / 0.965，看着很好。
但同一批日志里：

```json
{"location": "aiter_mla.py:prepare_decode", "data": {"globalCtx": [948], ...}}
```

**上下文只有 948 token，而 top-k 预算是 2048**（`assert topk_tokens == 2048`）。
ctx < 2048 时 top-k 把整个上下文全选进来，选择结果与 logits 数值无关 ——
**indexer 根本不筛选，index cache 的内容压根不参与决策**。所以这个分数对
"index page 字节是否正确"完全不敏感。

> 这条 PD 路径目前**没有任何有效的端到端验证**。要验证必须让 ctx > 2048，
> 或者直接做字节比对（见 9.7）。

#### 建议

源在传进去之前先 view 到 scheduler block 粒度，和 `deepseek_v2.py:1466` 保持同一
个不变量：

```python
source = runner.index_cache[index_region_idx]
block_size = runner.config.kv_cache_block_size
pages = gather_dcp_preshuffled_index_pages(
    source.view(-1, block_size, source.shape[-1]),
    slot, src_block_ids, dcp_size, dcp_rank,
    index_head_dim, block_size,
)
```

这样 `source_page_size` 恒等于 `block_size`（连接器侧已经用
`self.block_size % 16` 挡住了非 16 倍数），于是：

- `block_ratio` 参数、`physical_blocks`/`physical_token` 的两级拆分、
  `source_page_size == 1` 整个分支**全部可以删掉**；
- 那句 docstring "Prefill commonly stores one token-contiguous physical page while
  decode stores 16-token MFMA tiles" 是**错的**（两侧同一份代码、同一个
  `--block-size`，都是 preshuffled 16），必须改掉，否则会把下一个人引到同一个坑里。

顺带消掉一处不对称：写端硬编码"一个 scheduler block = 一个 (key plane | scale plane)
单元"，读端却支持多物理页。改成 `block_ratio=1` 后两端自然一致。

### 9.3 PD-P1：新增的 bpb 校验打死了 replicated index cache 路径

新加的握手校验对**所有** region 生效，包括 index region，而且在 staged 分流**之前**：

```python
bpb = self._per_block_bytes_list[region_idx]
if consumer_bpb is not None and consumer_bpb[cmap[region_idx]] != bpb:
    raise RuntimeError(f"Region byte-size mismatch for req {req_id}: ...")
```

但 replicated index cache 的 consumer 页**按设计就是 producer 的 W 倍宽**。
分配处（`aiter_mla.py:1345-1350`）`physical_block_size * index_page_factor`，
`index_page_factor = dcp_world_size`；`plan_replicated_index` 的 docstring 和实现也
明说了：

```python
dst_page = dst_ids[dst_block[keep]] * src_page_bytes * dcp_size
```

代入脚本的数：producer index bpb = 2304，replicated consumer = 9216。
校验必然 raise。也就是说**这个校验会打死它本该保护的那条路径** ——
而那正是改动前错误信息推荐的逃生通道（`ATOM_DCP_REPLICATE_INDEX_CACHE=1`）。

建议：`replicates_index and role == INDEX_CACHE_ROLE` 时跳过该校验，或者改成
校验 `consumer_bpb == bpb * (dcp_size if replicated_index else 1)`。后者更好，
因为它把"W 倍宽"这个不变量也一起断言了。

### 9.4 PD-P2：fence 打在了错的设备上

> **已修复并经 GSM8K 20-shot 验证。** `mooncake-send-worker` 线程池现在用
> connector 初始化时的 CUDA device 调用 `torch.cuda.set_device`；修复后四个
> PP rank 的 worker device 分别为 0/1/2/3，PD 分离精度从 0.9227 恢复到
> 0.9644，与 PD mix 的 0.9712 在统计误差内。

```python
staging_base, staged_pages = self._gather_sharded_index(...)
# The callback enqueues GPU writes on this worker thread's
# current stream. Fence them before Mooncake lets the NIC read.
torch.cuda.current_stream().synchronize()
```

`_execute_block_transfer` 跑在 `ThreadPoolExecutor(max_workers=16,
thread_name_prefix="mooncake-send-worker")` 的 worker 线程上，而
`mooncake_connector.py` 里**没有任何 `torch.cuda.set_device`**。
CUDA 的 current device 是**每线程**状态，新线程默认是 device 0；
主线程那次 `torch.cuda.set_device(self.device)` 只作用于主线程。

prefill 侧 `HIP_VISIBLE_DEVICES=0,1,2,3` + PP4，每个 rank 进程能看到 4 个设备并
`set_device(local_rank)`。gather 的 kernel 会入队到**张量所在设备**在本线程的
current stream（device N 的 default stream），而无参数的
`torch.cuda.current_stream()` 返回的是**本线程 current device（=0）**的 stream。
于是 **local_rank 1/2/3 上这个 fence 完全无效**，NIC 可能读到还没写完的 staging
buffer → index page 部分是旧内容，表现为随负载抖动的、不可复现的精度掉点。

这个模式**不是这次新引入的**：`mooncake_connector.py:2017` 的 compressor-state
`_gather_slot` 早就这么写，它的注释还写着 "Without this, the RDMA can race the
still-in-flight gather kernel on TBO prefill (page fault under high concurrency)"
—— 说明这个 race 已经被撞到过一次。新路径只是把暴露面放大了（21 个 index layer ×
每个 chunk 各一次）。

建议一次修掉两处。`__init__` 跑在主线程上（第 659 行已经在用
`torch.cuda.current_device()`），把设备存下来，再让 worker 线程一开就绑上：

```python
# __init__，主线程
self._cuda_device = torch.cuda.current_device()
self._send_executor = ThreadPoolExecutor(
    max_workers=...,
    thread_name_prefix="mooncake-send-worker",
    initializer=torch.cuda.set_device,
    initargs=(self._cuda_device,),
)
```

这样两处 `torch.cuda.current_stream()` 都自动落到正确设备。不想动线程池的话，
最小改动是两处都写显式设备：
`torch.cuda.current_stream(self._cuda_device).synchronize()`。
（gather 本身入队的是**张量所在设备**在本线程的 current stream，设备是对的、
只是 default stream；错的只有那个无参数的 fence。）

**另外两个性能问题（不影响正确性）**：

- gather 入队在 **default stream** 上，会和 prefill 的前向串行；producer 还在跑
  其他请求的 prefill 时，这部分 GPU 工作直接插进关键路径。
- 每个 chunk 都做一次**全 host 阻塞**同步，`21 个 index region × ceil(dst_pages/256)`
  次，全程占着那个 pool slot。改成 per-slot 的 `cuda.Event` + 专用 stream 可以把
  fence 粒度收紧，也不再占用 default stream。

### 9.5 PD-P3：staging pool 的分配条件与 PD 角色无关

```python
if hasattr(runner, "index_cache") and self.dcp_world_size == 1:
```

`dcp_world_size == 1` 只是"我不是被分片的那一侧"的**代理条件**，不等于"我是
producer"。而 `get_kv_transfer_tensors` 是**每次 `allocate_kv_cache` 都无条件调用**
的（V4 backend 在 `deepseek_v4_attn.py:1820-1826` 明确注释了这点，并且它自己会查
`runner.config.kv_transfer_config`；这里没查）。后果：

1. **所有 dcp=1 的 DSA 部署**都会分配 `16 × 256 × 2304 B ≈ 9 MiB`
   并把它注册进 RDMA —— 包括完全没开 PD 的单机 server 和 offline inference。
   而且分配发生在 KV cache 显存 profile **之后**，紧配置下可能把启动推到 OOM。
2. producer 自己开 DCP > 1 时不分配 staging，于是撞上
   `"Sharded preshuffled DSA index transfer requires ... producer index staging"`。
   这个限制本身是合理的（producer 自己的 index cache 也被分片了，重打包得先跨 rank
   gather），但错误信息把原因归到"缺 staging"，会让人去找错方向。

建议：条件改成 `runner.config.kv_transfer_config` 存在且角色是 producer；
DCP producer 的组合单独给一句写明真实原因的错误。

### 9.6 PD-P4 / PD-P5：两个小一点的

**PD-P4：页数不匹配退化成挂死。**

```python
if staged_pages != len(dst_chunk):
    raise RuntimeError(f"Index staging produced {staged_pages} pages for {len(dst_chunk)} destinations")
```

`try/finally` 只负责还 slot，异常继续上抛，最终被 `_execute_transfer` 的宽
`except Exception` 捕获并打日志（`mooncake_connector.py:1537`）—— consumer 永远收不到
write-done，**只能等超时**。正常情况下这个断言不会触发（9.1 里那段切分算术保证了
`cdiv` 结果相等），但 `len(dst_block_ids)·W > len(src_block_ids)` 且超出 clip 范围时
会触发，比如开 prefix caching 后的部分命中传输（脚本里 `ENABLE_PREFIX_CACHING=0`
默认关着，所以现在扫不到）。建议要么返回 `False` 走正常失败路径，要么先证明它不可达
并把证明写进注释。

**PD-P5：scale plane 字节数有两套算法。**

| 位置 | 公式 | `index_head_dim=128` |
|---|---|---|
| `aiter_mla.py:1457-1465`（region 描述符） | `tokens_per_page * 4` | 64 |
| `aiter_mla.py:258-260`（staging） | `scheduler_block_size * (index_head_dim // 128) * 4` | 64 |

两者只在 `index_head_dim == 128` 时相等。同一个 plane 在两处用不同公式描述，
换 head_dim 就静默错位。另外 `quant_block_size=128` 是个默认参数、调用方从不传，
而模型侧是把自己的 `quant_block_size` 传给 kernel 的 —— 应该从
`runner.config.hf_config` 取同一个源，别在这里另立一个默认值。

### 9.7 这条路该怎么验

现有的 GSM8K 跑分对这条路是盲的（9.2）。修完 PD-P0 之后建议按这个顺序验：

1. **字节比对（最直接，不需要长上下文）**：同一个 prompt 分别跑
   ①PD + DCP4 分片，②非 PD 的 DCP4 单机。传输完成后在 decode 侧 dump 某个
   full-index layer 的若干 index page，两者应当**逐字节相等**。这直接判定重打包，
   不经过 top-k 的掩盖。
2. **长上下文精度**：让 ctx > 2048 使 top-k 真正开始筛选，再跑一次
   accuracy。可以用 `run-atom-workload` 的长上下文数据集，或把 GSM8K 换成
   带长 few-shot 前缀的变体。
3. **replicated 对照**：`ATOM_DCP_REPLICATE_INDEX_CACHE=1` 跑同一个 case
   —— 修完 PD-P1 才跑得起来，它是这条路唯一的独立参照。
4. **多 rank fence 验证**：PD-P2 修之前，在 local_rank≠0 上高并发跑，看是否出现
   随机掉点；修之后应当消失。

---

## 十、建议的动手顺序

**先 PD，因为那里有确定的错误：**

1. **PD-P0**：源改成按 `kv_cache_block_size` re-view，删掉 `block_ratio` /
   `source_page_size == 1` 整条分支，改掉那句错的 docstring。
2. **PD-P1**：bpb 校验对 replicated index region 放行（或改成带 W 倍数的断言）。
3. **PD-P2**：executor 加 `initializer=torch.cuda.set_device`，同时修掉
   `_gather_slot` 那处同源问题（fix-then-sweep）。
4. **PD-P3 / PD-P4 / PD-P5**：分配条件加角色判断、raise 改成返回 `False`、
   scale plane 公式统一到一处。
5. 按 9.7 验证，**至少要跑到第 1 项**（字节比对）—— 没有它就等于没验。

**然后是 MTP 那一半：**

6. ~~**P2**：删 by_pos 链~~ → **已完成**，而且做成了函数合并，比原建议更彻底。
   两个同源问题也一起修了。
7. **P1**（优先级已提高）：`_enter_decode_metadata` 里把
   `dcp_local_context_lens` 置 `None`。原来还有 capture 路径不发布这个 buffer
   在意外兜底，修 同源问题 #2 之后那层没了，现在只剩
   `index_share_for_mtp_iteration` 挡着，V3.2 没有它。
8. **P3**：删掉 `dcp_ops.py` 剩下那段调试代码（20-shot 实验跑完就删）。
   它会在只读文件系统上让 indexer 直接 `OSError`。
9. **P0**：加 aiter 能力探测 + 显式报错；aiter 侧起 upstream PR。
10. **P4**：测试改 CPU-runnable。这次 P2 的重构是零自动化覆盖做完的，
    把 `cmq_scripts/verify_merged_ctx_lens.py` 和
    `cmq_scripts/verify_index_staging_layout.py` 两个 CPU 对照收成正式单测 ——
    它们分别能挡住 next_n 分派回归和 PD-P0 这类布局回归。
11. 重跑 GSM8K 确认 0.98 没退，并补一次 V3.2（无 index sharing）的 DCP+MTP 验证
    —— 那才是 P1 真正覆盖的配置。

untracked 产物别混进 PR：`.runtime/`、`1`、`n_gpu.patch`、`unit-report.xml`、
`cmq_docs/`、`cmq_scripts/`。
