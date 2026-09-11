# Metric-list coverage experiment

Same corpus and extractor settings as Step 1 (window 12, exclusion 8, nearest-cue attribution). n = 357 PRs, 177 validated.

## Coverage by configuration

| configuration                      |   validated ≥1 dim % |   agent % |   human % |   mean #dims (validated) |   non-validated ≥1 dim % (noise) |   newly covered validated PRs |   validated PRs hit by new dims |   non-validated PRs hit by new dims |
|:-----------------------------------|---------------------:|----------:|----------:|-------------------------:|---------------------------------:|------------------------------:|--------------------------------:|------------------------------------:|
| baseline                           |                 43.5 |      39.8 |      53.1 |                      0.6 |                              5   |                           nan |                             nan |                                 nan |
| +D10 Storage / disk footprint      |                 43.5 |      39.8 |      53.1 |                      0.6 |                              5   |                             0 |                               0 |                                   0 |
| +D11 GPU / accelerator             |                 43.5 |      39.8 |      53.1 |                      0.6 |                              5   |                             0 |                               1 |                                   0 |
| +D12 Errors / timeouts under load  |                 43.5 |      39.8 |      53.1 |                      0.7 |                              5   |                             0 |                               3 |                                   1 |
| +D13 UI / rendering responsiveness |                 43.5 |      39.8 |      53.1 |                      0.6 |                              5   |                             0 |                               1 |                                   0 |
| +D14 Work volume processed         |                 44.1 |      40.6 |      53.1 |                      0.7 |                              5   |                             1 |                               4 |                                   0 |
| +D15 LLM tokens / context          |                 43.5 |      39.8 |      53.1 |                      0.6 |                              5   |                             0 |                               0 |                                   0 |
| +D16 Cache effectiveness           |                 43.5 |      39.8 |      53.1 |                      0.6 |                              5   |                             0 |                               0 |                                   0 |
| +size-tables (D6)                  |                 44.1 |      40.6 |      53.1 |                      0.7 |                              5.6 |                             1 |                                 |                                     |
| all candidates                     |                 44.6 |      41.4 |      53.1 |                      0.7 |                              5.6 |                             2 |                               8 |                                   1 |

*newly covered* = validated PRs with 0 dimensions under the baseline that gain ≥ 1 under the configuration; *noise* = share of PRs **without** RQ2 validation evidence in which the list fires.

## Dimension frequency under 'all candidates' (% of validated PRs)

|     | label                         |   agent % |   human % |   all % |   non-validated % (noise) |
|:----|:------------------------------|----------:|----------:|--------:|--------------------------:|
| D0  | Unspecified performance       |       2.3 |       6.1 |     3.4 |                       0   |
| D1  | Latency / execution time      |      31.2 |      42.9 |    34.5 |                       4.4 |
| D2  | Throughput                    |       3.1 |       2   |     2.8 |                       0.6 |
| D3  | Memory                        |       7.8 |      10.2 |     8.5 |                       0.6 |
| D4  | CPU / compute work            |       3.9 |       0   |     2.8 |                       0   |
| D5  | I/O and network               |       3.1 |       4.1 |     3.4 |                       0   |
| D6  | Artifact size                 |       3.1 |       4.1 |     3.4 |                       0.6 |
| D7  | Build and CI time             |       2.3 |       2   |     2.3 |                       0   |
| D8  | Energy and cost               |       1.6 |       0   |     1.1 |                       0.6 |
| D9  | Scalability / concurrency     |       2.3 |       2   |     2.3 |                       0   |
| D11 | GPU / accelerator             |       0.8 |       0   |     0.6 |                       0   |
| D12 | Errors / timeouts under load  |       2.3 |       0   |     1.7 |                       0.6 |
| D13 | UI / rendering responsiveness |       0.8 |       0   |     0.6 |                       0   |
| D14 | Work volume processed         |       3.1 |       0   |     2.3 |                       0   |

## Audit snippets for new dimensions / enrichments

### +D11 GPU / accelerator — D11 (2 PR-source hits; 2 in validated PRs)
- [AI Ag|val|description] `cuda graph` ⟷ `0.6b` — ase. ## testing before ``` vllm serve qwen/qwen3-0.6b ... capturing cuda graph shapes: 100%|███████| 67/67 [00:34<00:00, 1.92it/s] info 0 (https://github.com/vllm-project/vllm/pull/21146)
- [AI Ag|val|issue_comments] `cuda graph` ⟷ `0.6b` — scription) for `vllm serve qwen/qwen3-0.6b` ``` # main capturing cuda graph shapes: 100%|██████████████| 67/67 [00:10<00:00, 6.23it/s] (https://github.com/vllm-project/vllm/pull/21146)

### +D12 Errors / timeouts under load — D12 (4 PR-source hits; 3 in validated PRs)
- [AI Ag|val|code_diff] `timeout` ⟷ `from 90s to 60` — but proceeding with build...'); - } - startbuild(); -}); + // set timeout for migrations (reduced from 90s to 60s) + const migrationtim (https://github.com/different-ai/zero-finance/pull/216)
- [AI Ag|val|code_diff] `success rate` ⟷ `-8%` — │ +│ selective buy (10-14 score): 78% success rate ⚠️ │ +│ ├─ 9 events recommended (https://github.com/langchain-ai/langchain/pull/31987)
- [AI Ag|val|code_diff] `timeouts` ⟷ `from 30s to 15` — pooling**: increased max idle connections from 5 to 30 +- **optimized timeouts**: reduced connect timeout from 30s to 15s +- **adaptive time (https://github.com/gmathi/NovelLibrary/pull/246)
- [AI Ag|NOV|commit_messages] `timeout` ⟷ `from 1500ms to 1000` — change default hotstuff-min-timeout for collection nodes from 1500ms to 1000ms this change reduce (https://github.com/onflow/flow-go/pull/7598)

### +D13 UI / rendering responsiveness — D13 (1 PR-source hits; 1 in validated PRs)
- [AI Ag|val|code_diff] `smooth` ⟷ `60fps` — er) +- memory usage: optimized with better caching +- list scrolling: smooth 60fps performance + +## 🧪 testing + +### performance testing +- (https://github.com/gmathi/NovelLibrary/pull/246)

### +D14 Work volume processed — D14 (5 PR-source hits; 5 in validated PRs)
- [AI Ag|val|description] `search space` ⟷ `reduced from 8 to 2 vrps (75%` — nce asn filtering is highly selective: - **test case**: with 8 vrps, search space reduced from 8 to 2 vrps (75% reduction) for asn 13335 - (https://github.com/tomhrr/cosh/pull/181)
- [AI Ag|val|commit_messages] `operations` ⟷ `30% - improved` — all identified issues performance improvements: - reduced dom query operations by 20-30% - improved responsiveness during drag/drop operat (https://github.com/ateliee/jquery.schedule/pull/58)
- [AI Ag|val|code_diff] `operations` ⟷ `30% reduction` — zation alone is expected to provide: +- 20-30% reduction in dom query operations +- improved responsiveness during drag/drop operations +- b (https://github.com/ateliee/jquery.schedule/pull/58)
- [AI Ag|val|commit_messages] `operations` ⟷ `by ~50%` — 5 identified inefficiencies - reduce database round trips for search operations by ~50% (https://github.com/sourcebot-dev/sourcebot/pull/357)
- [AI Ag|val|commit_messages] `operations` ⟷ `0.1% improvement` — 7s (0.1% improvement) - focus on reducing expensive view layer update operations during import (https://github.com/saturday06/VRM-Addon-for-Blender/pull/964)

### +size-tables (D6) — D3 (21 PR-source hits; 20 in validated PRs)
- [AI Ag|val|description] `allocations` ⟷ `148.381mb` — before: ``` julia> @time pkg.instantiate() 0.390297s (1.95 m allocations: 148.381mb, 16.29% gc time, 31.03% compilation time: 68% o (https://github.com/JuliaLang/Pkg.jl/pull/4304)
- [AI Ag|val|code_diff] `memory` ⟷ `5mb` — "django.core.cache.backends.locmem.locmemcache", + # limit memory usage (in bytes, 5mb) + "options": { + (https://github.com/OWASP-BLT/BLT/pull/4290)
- [AI Ag|val|code_diff] `memory` ⟷ `11.7 kb` — golden test checklist (53/284) | index | name | status | duration | memory | |------:|------|:-----:|---------:|-------:| | 1 | 100-door (https://github.com/mochilang/mochi/pull/13059)
- [AI Ag|val|code_diff] `memory` ⟷ `11.7 kb` — golden test checklist (53/284) | index | name | status | duration | memory | |------:|------|:-----:|---------:|-------:| | 1 | 100-door (https://github.com/mochilang/mochi/pull/13057)
- [AI Ag|val|code_diff] `memory` ⟷ `11.7 kb` — golden test checklist (105/284) | index | name | status | duration | memory | |------:|------|:-----:|---------:|-------:| | 1 | 100-door (https://github.com/mochilang/mochi/pull/13066)
- [AI Ag|NOV|code_diff] `内存` ⟷ `提升** 消息处理性能提升 50%` — 24 小时 + + +### 7.3 监控指标 + +关键监控指标 +- 活跃设备数量 +- 消息处理延迟 +- 线程池使用率 +- 内存使用率 +- 网络连接状态 + +## 8. 预期收益 + +通过以上优化措施 预期可以达到 + +- **性能提升** 消息处理性能 (https://github.com/lunasaw/gb28181-proxy/pull/38)
- [AI Ag|val|commit_messages] `memory` ⟷ `6gb` — pilation - skip migrations for preview deployments - increase node.js memory allocation to 6gb - add optimized build commands and scripts - (https://github.com/different-ai/zero-finance/pull/216)
- [AI Ag|val|code_diff] `memory` ⟷ `4gb` — log('starting vercel-optimized build process...'); -console.log('node memory limit set to 4gb'); +console.log('starting optimized vercel bui (https://github.com/different-ai/zero-finance/pull/216)
- [AI Ag|val|description] `memory` ⟷ `8kb` — treaming approach with a fixed 8kb buffer. this significantly reduces memory consumption and improves performance when hashing large files, (https://github.com/vercel/turborepo/pull/10623)
- [AI Ag|val|commit_messages] `allocation` ⟷ `8kb` — _to_end with streaming file hash implementation: - remove vec::new() allocation for each file - use fixed 8kb buffer instead of loading ent (https://github.com/vercel/turborepo/pull/10623)
- [AI Ag|val|code_diff] `memory` ⟷ `8kb` — / stream the file content in chunks to avoid loading entire file into memory +let mut buffer = [0u8; 8192]; // 8kb buffer - optimal for most (https://github.com/vercel/turborepo/pull/10623)
- [AI Ag|val|code_diff] `内存` ⟷ `1mb` — = 1024 * 1024 # 大对象阈值 1mb + # 启用tracemalloc以获得更详细的内存信息 if not tracemalloc.is_tracing(): tracemallo (https://github.com/jxxghp/MoviePilot/pull/4579)
- [AI Ag|val|description] `allocation` ⟷ `50% allocation overhead reduction` — | 35-45% latency reduction | no change | | **memory pool** | 40-50% allocation overhead reduction | 45-55% allocation overhead reduction | (https://github.com/nnstreamer/nntrainer/pull/3312)
- [AI Ag|val|issue_comments] `heap` ⟷ `177,245,026b` — y allocation overhead. ##### 2-1 before ```bash ==873070== total heap usage: 137,271 allocs, 137,057 frees, 177,245,026b allocated p (https://github.com/nnstreamer/nntrainer/pull/3312)
- [AI Ag|val|description] `allocation` ⟷ `50% allocation reduction` — nce validation**: run search benchmarks to confirm the claimed 30-50% allocation reduction translates to real performance gains - [ ] **ci e (https://github.com/buger/probe/pull/56)
- [Human|val|description] `allocs` ⟷ `80 b` — rk_equalfieldtype-12 361.0 ns/op 80 b/op 9 allocs/op ``` new: ``` benchmark_equalfieldtype-12 (https://github.com/gofiber/fiber/pull/3479)
- [Human|val|commit_messages] `allocs` ⟷ `80 b` — rk_equalfieldtype-12 361.0 ns/op 80 b/op 9 allocs/op ``` new: ``` benchmark_equalfieldtype-12 99.8 (https://github.com/gofiber/fiber/pull/3479)
- [Human|val|description] `allocation` ⟷ `up to 20x` — ses like `bitset(mask)._words[0]`. * removed an unnecessary `string` allocation to skip a `malloc` and reduce memory overhead. (https://github.com/modular/modular/pull/4511)
- [Human|val|issue_comments] `allocs` ⟷ `+563.19%` — new.txt │ │ allocs/op │ allocs/op vs base │ readfile/mapfs_sma (https://github.com/microsoft/typescript-go/pull/218)
- [Human|val|description] `allocated | alloc ratio | |- … s | 0.386 ns | 1.00 | 32 b |` ⟷ `allocated | alloc ratio | |---------------- |---------- |---` — h). | method | toolchain | mean | error | ratio | allocated | alloc ratio | |---------------- |---------- |----------:| (https://github.com/dotnet/runtime/pull/117071)
- [Human|val|issue_comments] `gen0 | allocated | alloc rat … 0 | 0.00 | 0.1736 | 8024 b |` ⟷ `gen0 | allocated | alloc ratio | |------------- |-----------` — length | mean | error | stddev | ratio | ratiosd | gen0 | allocated | alloc ratio | |------------- |----------- |----- (https://github.com/dotnet/fsharp/pull/18509)

### +size-tables (D6) — D1 (89 PR-source hits; 80 in validated PRs)
- [AI Ag|val|description] `30 ns` ⟷ `30 ns` — s v = rand(int, 10) gv = garbagevector{100}(v) @btime collect($v); # 30 ns ( ) -> 30 ns (pr) @btime collect($gv); # 179 ns ( ) -> 30 n (https://github.com/JuliaLang/julia/pull/59071)
- [AI Ag|val|issue_comments] `runtime` ⟷ `3s` — tested. it reduces runtime by 2-3s in debug mode. (https://github.com/nearai/nearai/pull/1179)
- [AI Ag|val|description] `speedup` ⟷ `1.87x speedup` — rithm optimizations** - [x] **`next_power_of_2` optimization**: 1.87x speedup using bit manipulation - [x] **threading integration**: 7 para (https://github.com/microsoft/onnxruntime/pull/25061)
- [AI Ag|val|code_diff] `speedup` ⟷ `1.75x speedup` — + } + return out; + } +} +``` + +**performance impact:** 1.75x speedup measured in microbenchmarks. + +### 2. threading support + +ad (https://github.com/microsoft/onnxruntime/pull/25061)
- [AI Ag|val|description] `faster` ⟷ `0.8 ns` — ng shows dramatic performance improvements: - **row access**: ~6,000x faster (0.8 ns vs 4,841 ns) - **rowroots access**: ~420x faster (2.8 n (https://github.com/celestiaorg/rsmt2d/pull/361)
- [AI Ag|NOV|description] `time` ⟷ `from 10 to 30` — increase websocket reconnect wait time from 10 to 30s ## description this pr increases the wait time be (https://github.com/microsoft/HydraLab/pull/695)
- [AI Ag|NOV|commit_messages] `time` ⟷ `from 10 to 30` — initial plan for issue increase websocket reconnect sleep time from 10 to 30s (https://github.com/microsoft/HydraLab/pull/695)
- [AI Ag|val|description] `took` ⟷ `24min` — late_nd` function. for output shape `(1, 384, 40, 40)`, the operation took approximately **24min** to complete, making it unusable for pract (https://github.com/onnx/onnx/pull/7057)
- [AI Ag|val|issue_comments] `run time` ⟷ `0.004s` — 4911 | 2 | 4909 | 3390 | view the top 2 failed test(s) by shortest run time > stack traces | 0.004s run time > > > onnx\backen (https://github.com/onnx/onnx/pull/7057)
- [AI Ag|val|issue_comments] `1.25us` ⟷ `1.25us` — --------------|------------------|---| | depthmap::from_parent | 60.3±1.25us | 60.6±1.59us | +0.50% | | fix_complex_query | 12.0±0.14ms | ** (https://github.com/quarylabs/sqruff/pull/1749)
- [AI Ag|val|description] `slower` ⟷ `1.41x slower` — formance regressions in wasm compilation mode: - `bitarrayget`: 1.41x slower (183.17 ns -> 259.16 ns) - `bitarrayset`: 1.42x slower (34.17 (https://github.com/dotnet/runtime/pull/117160)
- [AI Ag|val|issue_comments] `slower` ⟷ `1.1x to 5.4x slower` — ver, the performance data shows significant regressions (1.1x to 5.4x slower) in critical bitarray operations specifically in wasm compilati (https://github.com/dotnet/runtime/pull/117160)
- [AI Ag|val|description] `taking` ⟷ `4s` — -datatype element ( (a1) (a2) (a3) ... (a50000) )) ``` the cli was taking 4s for 50,000 constructors and 14s for 100,000 constructors, wh (https://github.com/Z3Prover/z3/pull/7710)
- [AI Ag|NOV|code_diff] `speedup` ⟷ `5x speedup` — | | empty right | 50 | 5 | the optimized hash join yields a ~4-5x speedup over the unoptimized nested-loop approach. @@ -14,7 +14,6 @@ i (https://github.com/mochilang/mochi/pull/3943)
- [AI Ag|val|code_diff] `time` ⟷ `+21050.0%` — | +| python | 846 | +21050.0% | ## math.fact_rec.20 | language | time (µs) | +/- | | --- | ---: | --- | -| c | 6 | best | -| mochi (vm (https://github.com/mochilang/mochi/pull/3948)
- [AI Ag|val|code_diff] `duration` ⟷ `116us` — ## rosetta golden test checklist (53/284) | index | name | status | duration | memory | |------:|------|:-----:|---------:|-------:| | 1 (https://github.com/mochilang/mochi/pull/13059)
- [AI Ag|val|code_diff] `time` ⟷ `+240862.5%` — (interp) | 57831 | +240862.5% | ## math.fact_rec.20 | language | time (µs) | +/- | | --- | ---: | --- | | mochi | 55 | best | -| type (https://github.com/mochilang/mochi/pull/217)
- [AI Ag|val|description] `took` ⟷ `35s` — 22:13:03 [gpu_model_runner.py:2283] graph capturing finished in 35s, took 0.59gb ``` after ``` vllm serve qwen/qwen3-0.6b ... capturing cu (https://github.com/vllm-project/vllm/pull/21146)
- [AI Ag|NOV|code_diff] `duration` ⟷ `30s` — on = system.currenttimemillis() - starttime; + if (duration > 30000) { // warn if cleanup blocks scheduler for >30s + (https://github.com/Stirling-Tools/Stirling-PDF/pull/3992)
- [AI Ag|val|description] `time` ⟷ `13.616 s` — inary && go test -tags test_dep -count=1 -c -o test_binary ./tests time (mean ± σ): 13.616 s ± 0.215 s [user: 13.757 s, system: 3 (https://github.com/temporalio/temporal/pull/7896)
- [AI Ag|NOV|description] `faster` ⟷ `50ms` — when the server fails to start - poll server readiness every 50ms for faster startup ## testing - `python scripts/check_python_deps.py` - ` (https://github.com/MontrealAI/AGI-Alpha-Agent-v0/pull/3666)
- [AI Ag|val|code_diff] `duration` ⟷ `116us` — ## rosetta golden test checklist (53/284) | index | name | status | duration | memory | |------:|------|:-----:|---------:|-------:| | 1 (https://github.com/mochilang/mochi/pull/13057)
- [AI Ag|val|code_diff] `duration` ⟷ `116us` — ## rosetta golden test checklist (105/284) | index | name | status | duration | memory | |------:|------|:-----:|---------:|-------:| | 1 (https://github.com/mochilang/mochi/pull/13066)
- [AI Ag|NOV|code_diff] `时间` ⟷ `缩短 70%` — ** 内存使用率降低 30%+ +- **稳定性提升** 系统可用性达到 99.9%+ +- **可维护性** 通过监控和诊断工具 问题定位时间缩短 70%+ +- **扩展性** 支持更大规模的设备接入 10,000+设备 + +这些优化方案基于对现有代码的深入分析 针对性 (https://github.com/lunasaw/gb28181-proxy/pull/38)
- [AI Ag|val|description] `time` ⟷ `reduce vercel deployment time by 2x` — reduce vercel deployment time by 2x reduce vercel deploy time by offloading typescript checks (https://github.com/different-ai/zero-finance/pull/216)

### +size-tables (D6) — D9 (6 PR-source hits; 6 in validated PRs)
- [AI Ag|val|description] `7 parallel` ⟷ `7 parallel` — 1.87x speedup using bit manipulation - [x] **threading integration**: 7 parallel execution paths added - [x] **memory efficiency**: optimize (https://github.com/microsoft/onnxruntime/pull/25061)
- [AI Ag|val|code_diff] `connection pooling` ⟷ `from 5 to 30` — 3. network layer optimizations + +### networkhelperoptimized.kt +- **connection pooling**: increased max idle connections from 5 to 30 +- * (https://github.com/gmathi/NovelLibrary/pull/246)
- [AI Ag|val|description] `30 workers` ⟷ `30 workers` — ance optimizations #### thread pool optimization - **before**: fixed 30 workers regardless of system capacity - **after**: dynamic 4-8 work (https://github.com/test-zeus-ai/testzeus-hercules/pull/61)
- [AI Ag|val|commit_messages] `thread pool` ⟷ `from 4-8 to 2` — for planner, 10 -> 5 for nav agent (90% reduction) - optimize appium thread pool from 4-8 to 2-3 workers for mobile scenarios - add direct (https://github.com/test-zeus-ai/testzeus-hercules/pull/61)
- [AI Ag|val|code_diff] `thread pool` ⟷ `from 4-8 to 2` — e performance optimizations applied:") + print(" • reduced appium thread pool from 4-8 to 2-3 workers") + print(" • added direct exe (https://github.com/test-zeus-ai/testzeus-hercules/pull/61)
- [Human|val|description] `10,000 users` ⟷ `10,000 users` — sion). ## how has this been tested? this was tested with about 10,000 users, 100 user groups, and 500 personas. ## backporting (c (https://github.com/onyx-dot-app/onyx/pull/4127)

### +size-tables (D6) — D6 (9 PR-source hits; 8 in validated PRs)
- [AI Ag|val|description] `bundle size` ⟷ `3mb` — decrease opennext bundle size to below 3mb this pr reduces the opennext bundle size to w (https://github.com/unibeck/solstatus/pull/55)
- [AI Ag|val|commit_messages] `bundle size` ⟷ `3mb` — s and build configurations for reduced bundle size further optimize bundle size with enhanced configuration successfully reduced opennex (https://github.com/unibeck/solstatus/pull/55)
- [AI Ag|val|code_diff] `bundle size` ⟷ `3mb` — if (compressedsize <= 3 * 1024 * 1024) { + console.log('✓ success: bundle size is below 3mb when compressed!'); + } else { + console. (https://github.com/unibeck/solstatus/pull/55)
- [AI Ag|val|issue_comments] `size-limit report 📦 | path | … /browser.esm.js | 92.28 kb (` ⟷ `size-limit report 📦 | path | size | | ----------------------` — nges with github/github using the [integration workflow]( thanks! ## size-limit report 📦 | path | size (https://github.com/primer/react/pull/6197)
- [AI Ag|val|code_diff] `bundle size` ⟷ `20mb` — ecyclerview + +## 📊 performance metrics + +### before optimization +- bundle size: ~15-20mb +- cold start time: ~3-5s +- database operations (https://github.com/gmathi/NovelLibrary/pull/246)
- [AI Ag|val|issue_comments] `gzip` ⟷ `15 kb` — reduction @magic-ext/web3modal-ethers5 | -~320 b | -~230 b | -15 kb (gzip) / -8 kb (br) | very large iife reduction @magic-sdk/react-nativ (https://github.com/magiclabs/magic-js/pull/874)
- [Human|val|issue_comments] `size=256; gc = concurrent wo … .64 | 0.07 | 0.0038 | 32 b |` ⟷ `size=256; gc = concurrent workstation mean = 29.934 us, stde` — 4.8) [length=10000] runtime = .net framework ( ), x64 ryujit vectorsize=256; gc = concurrent workstation mean = 29.934 us, stderr = 0.06 (https://github.com/dotnet/fsharp/pull/18509)
- [Human|NOV|issue_comments] `size | publisher | |:--- |:- …  none | <a href=" | 120 kb |` ⟷ `size | publisher | |:--- |:--- |:--- |:--- |:--- | | | none ` — ocket for github ↗︎]( | package | new capabilities | transitives | size | publisher | |:--- |:--- |:--- |:--- |:--- | | | none | <a hr (https://github.com/calcom/cal.com/pull/20080)
- [Human|val|code_diff] `file size` ⟷ `50mb` — ilecollecttask } from './workers/filecollectworker.js'; -// maximum file size to process (50mb) -// this prevents out-of-memory errors whe (https://github.com/yamadashy/repomix/pull/309)

### +size-tables (D6) — D5 (8 PR-source hits; 8 in validated PRs)
- [AI Ag|val|description] `api calls` ⟷ `50x fewer` — with up to 50 prs per request) - **typical improvement**: 5-50x fewer api calls depending on release size - **example**: 100 prs now require (https://github.com/mlflow/mlflow/pull/16039)
- [AI Ag|val|code_diff] `database queries` ⟷ `reduces database queries by up to 92%` — sary re-renders +- **reduced server load**: in-memory caching reduces database queries by up to 92% for app registry data +- **better resour (https://github.com/calcom/cal.com/pull/21052)
- [AI Ag|val|code_diff] `database queries` ⟷ `from 2 separate queries to 1` — n team?.members || []; +} +``` + +**performance impact**: +- reduces database queries from 2 separate queries to 1 optimized query with joi (https://github.com/amplication/amplication/pull/9794)
- [AI Ag|val|description] `database queries` ⟷ `50% reduction` — d and repo.name as keys ## performance impact - **50% reduction** in database queries for search operations - eliminates unnecessary round (https://github.com/sourcebot-dev/sourcebot/pull/357)
- [AI Ag|val|commit_messages] `database round trips` ⟷ `by ~50%` — e performance report documenting 5 identified inefficiencies - reduce database round trips for search operations by ~50% (https://github.com/sourcebot-dev/sourcebot/pull/357)
- [Human|val|description] `1b writes` ⟷ `1b` — hing after the recent merges and i need to find a way to simplify the 1b writes more nicely, i don't like all the specializations. --- (https://github.com/bitcoin/bitcoin/pull/31868)
- [Human|val|issue_comments] `1b writes` ⟷ `1b` — is hitting this during ibd during serialization we have very many 1b writes, this change avoids the heavy allocations. but drafting unt (https://github.com/bitcoin/bitcoin/pull/31868)
- [Human|val|code_diff] `payload` ⟷ `4mb` — tion?.metadata || {}); + // this helps to prevent reaching the 4mb payload limit by avoiding base64 and instead passing the avatar url + (https://github.com/calcom/cal.com/pull/21855)

### +size-tables (D6) — D2 (7 PR-source hits; 6 in validated PRs)
- [AI Ag|val|description] `3 million elements per second` ⟷ `3 million elements per second` — (1, 384, 40, 40) | ~24min | ~0.2s | **~7,400x** | processing rate: **3 million elements per second** ### testing: - ✅ correctness verified (https://github.com/onnx/onnx/pull/7057)
- [AI Ag|val|description] `1.92it/s` ⟷ `1.92it/s` — b ... capturing cuda graph shapes: 100%|███████| 67/67 [00:34<00:00, 1.92it/s] info 07-17 22:13:03 [gpu_model_runner.py:2283] graph capturi (https://github.com/vllm-project/vllm/pull/21146)
- [AI Ag|val|issue_comments] `6.23it/s` ⟷ `6.23it/s` — apturing cuda graph shapes: 100%|██████████████| 67/67 [00:10<00:00, 6.23it/s] # n=1 capturing cuda graph shapes: 100%|██████████████| (https://github.com/vllm-project/vllm/pull/21146)
- [AI Ag|NOV|code_diff] `吞吐量` ⟷ `1,000 msg/s` — 性能调优 + +### 7.2 性能测试 + +建议进行以下性能测试 +- 并发连接数测试 目标 10,000+设备 +- 消息处理吞吐量测试 目标 1,000 msg/s +- 内存使用优化验证 +- 长时间稳定性测试 24 小时 + + +### 7.3 监 (https://github.com/lunasaw/gb28181-proxy/pull/38)
- [AI Ag|val|description] `throughput` ⟷ `50%** - ✅ **throughput increase` — **latency reduction**: target 30-50% -> **achieved 30-50%** - ✅ **throughput increase**: target 3-5x -> **achieved 3-5x** - ✅ **mem (https://github.com/nnstreamer/nntrainer/pull/3312)
- [AI Ag|val|code_diff] `60fps` ⟷ `60fps` — memory usage: optimized with better caching +- list scrolling: smooth 60fps performance + +## 🧪 testing + +### performance testing +- ✅ cold (https://github.com/gmathi/NovelLibrary/pull/246)
- [Human|val|description] `fps` ⟷ `from ~1 to ~100` — t want to do this in other places. with this change in a debug build, fps goes from ~1 to ~100 on m4 max ### how did you ver (https://github.com/oven-sh/bun/pull/18585)

### +size-tables (D6) — D8 (3 PR-source hits; 2 in validated PRs)
- [AI Ag|NOV|code_diff] `costs` ⟷ `$ 900` — 1,4 @@ "--- top products (excluding most expensive) ---" -smartphone costs $ 900 -tablet costs $ 600 -monitor costs $ 300 +"smartphone" "co (https://github.com/mochilang/mochi/pull/6522)
- [AI Ag|val|code_diff] `price` ⟷ `+12%` — emand: ↑ +8% │ • vegas market oversaturated │ +│ avg ticket price: ↑ +12% │ • country music declining │ +│ resale veloci (https://github.com/langchain-ai/langchain/pull/31987)
- [AI Ag|val|description] `battery` ⟷ `40% battery improvement` — nference 3-5x faster - ✅ **battery life**: mobile devices see 30-40% battery improvement - ✅ **scalability**: better performance on multi- (https://github.com/nnstreamer/nntrainer/pull/3312)

### +size-tables (D6) — D7 (4 PR-source hits; 4 in validated PRs)
- [AI Ag|val|description] `build time` ⟷ `13.6s` — go - cloud.google.com/go - github.com/jackc/pgx **before: ~13.6s build time** ``` hyperfine --warmup 1 "rm -f test_binary && go tes (https://github.com/temporalio/temporal/pull/7896)
- [AI Ag|val|commit_messages] `ci runs` ⟷ `15min` — with best practices - add concurrency control to prevent overlapping ci runs - add timeout protection (15min ci, 10min publish) - add job s (https://github.com/ltwlf/json-diff-ts/pull/301)
- [AI Ag|val|issue_comments] `run duration` ⟷ `51s` — &nbsp;passed&nbsp; run duration 00m 51s c (https://github.com/ethyca/fides/pull/6310)
- [Human|val|description] `build time` ⟷ `from ~70s/30s/30s (on windows/ubuntu/mac) to ~15` — improve jupyterlab extension build time this pr reduces the build time of the jupyterlab package fr (https://github.com/microsoft/qsharp/pull/2530)

### +size-tables (D6) — D0 (11 PR-source hits; 11 in validated PRs)
- [AI Ag|val|description] `performance` ⟷ `5x overall performance improvement` — focusing on three core areas that collectively provide **3-5x overall performance improvement** across arm v9 and x64 processors. ## 📊 pe (https://github.com/nnstreamer/nntrainer/pull/3312)
- [AI Ag|val|commit_messages] `performance` ⟷ `400% quantization performance improvement` — k calls with simd-optimized versions - achieves 200-400% quantization performance improvement - maintains fallback compatibility for unsuppo (https://github.com/nnstreamer/nntrainer/pull/3312)
- [AI Ag|val|description] `performance` ⟷ `30% reduction` — `_getschedulecount`: convert `for...in` to standard for loops ### 📊 performance impact - **20-30% reduction** in dom query operations - ** (https://github.com/ateliee/jquery.schedule/pull/58)
- [AI Ag|val|commit_messages] `performance` ⟷ `30% - improved` — hensive efficiency analysis report documenting all identified issues performance improvements: - reduced dom query operations by 20-30% - i (https://github.com/ateliee/jquery.schedule/pull/58)
- [AI Ag|val|code_diff] `responsiveness` ⟷ `30% reduction` — d to provide: +- 20-30% reduction in dom query operations +- improved responsiveness during drag/drop operations +- better performance with (https://github.com/ateliee/jquery.schedule/pull/58)
- [AI Ag|val|commit_messages] `performance` ⟷ `from 500 -> 50` — bility across different environments implement comprehensive mobile performance optimizations - reduce llm chat rounds from 500 -> 50 for (https://github.com/test-zeus-ai/testzeus-hercules/pull/61)
- [AI Ag|val|description] `performance` ⟷ `0.1% performance improvement` — alidation of the changes. manual testing is especially important. - **performance vs correctness**: the 0.1% performance improvement is mode (https://github.com/saturday06/VRM-Addon-for-Blender/pull/964)
- [Human|val|issue_comments] `performance` ⟷ `97.44% performance improvement` — s concurrency safety by avoiding global state 4. achieves significant performance improvement as mentioned in the pr objectives the benchma (https://github.com/gofiber/fiber/pull/3329)
- [Human|val|description] `performance` ⟷ `improve performance by about ~30%` — ad times from user-group and persona endpoints; these changes improve performance by about ~30% in some cases (worst i saw was equivalent pe (https://github.com/onyx-dot-app/onyx/pull/4127)
- [Human|val|issue_comments] `performance` ⟷ `improvement of 4,848%` — late_node_levels` function in `flow/utils.py`, achieving a remarkable performance improvement of 4,848%. the changes focus on reducing compu (https://github.com/crewAIInc/crewAI/pull/2136)
- [Human|val|issue_comments] `speed` ⟷ `improvement (672%` — g regex-based parsing with string operations. it claims a significant speed improvement (672%). the changes are performance-driven with impl (https://github.com/crewAIInc/crewAI/pull/2137)

### +size-tables (D6) — D4 (5 PR-source hits; 5 in validated PRs)
- [AI Ag|val|description] `function calls` ⟷ `2.600s` — 関数の最適化 - 角度の閾値を微調整して不要な更新を減少 ## ベンチマーク結果 ### 最適化前 ``` function calls in 2.600s ordered by: internal time ncalls tot (https://github.com/saturday06/VRM-Addon-for-Blender/pull/795)
- [AI Ag|val|description] `function calls` ⟷ `2.715s` — 5. オブジェクト生成を最小限に抑える最適化 ## ベンチマーク結果の比較 ### 最適化前 ``` function calls in 2.715s ordered by: internal time ncalls (https://github.com/saturday06/VRM-Addon-for-Blender/pull/796)
- [AI Ag|val|description] `function calls` ⟷ `2.689s` — をキャッシュ 4. コライダー衝突計算でのベクトル計算を最適化 ## ベンチマーク結果 ### 最適化前 ``` function calls in 2.689s ordered by: internal time ncalls tot (https://github.com/saturday06/VRM-Addon-for-Blender/pull/797)
- [AI Ag|val|description] `function calls` ⟷ `2.629s` — extension(bone).uuid`の呼び出し回数を削減 ## ベンチマーク結果 ### 最適化前 ``` function calls in 2.629s ordered by: internal time ncalls tot (https://github.com/saturday06/VRM-Addon-for-Blender/pull/798)
- [AI Ag|val|description] `function calls` ⟷ `202.010s` — なボーン階層を持つモデルでのパフォーマンスが向上することを期待 ## ベンチマーク結果 ### 最適化前 ``` function calls in 202.010s ordered by: internal time ncalls t (https://github.com/saturday06/VRM-Addon-for-Blender/pull/800)
