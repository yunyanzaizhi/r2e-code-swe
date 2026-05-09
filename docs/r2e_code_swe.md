# R2E Code/SWE Environment

This project now has an R2E-only repository-level code/SWE environment registered as `env.env_name=r2e_code_swe`.

It keeps verl-agent as the training and rollout owner. The model still emits actions during veRL/GiGPO/GRPO rollouts. The adapter parses those actions and calls R2E-Gym `RepoEnv` / `DockerRuntime` for repository workspace setup, Docker execution, patch export, and unit-test reward.

## Data Splits

- `R2E-Gym/R2E-Gym-Lite`, `split=dev_10pr_v1`: smoke test, debugging, and validation only.
- `R2E-Gym/R2E-Gym-Lite`, `split=train`: training.
- `R2E-Gym/R2E-Gym-Subset`, `split=train`: training.
- `R2E-Gym/SWE-Bench-Lite`, `split=test`: final evaluation only.
- `R2E-Gym/R2EGym-SFT-Trajectories`: SFT warm-up or tool-call format learning.

The normalizer refuses `dev_10pr_v1` for training unless `allow_train_on_dev=true`, and always refuses `R2E-Gym/SWE-Bench-Lite split=test` for training.

## Install And Docker Check

```bash
cd /home/caiting/R2E-Gym
uv venv
source .venv/bin/activate
uv sync
uv pip install -e .

cd /home/caiting/verl-agent-exp-copy-from-lab-server-20260505
source .venv/bin/activate
uv pip install -e /home/caiting/R2E-Gym
uv pip install "datasets==4.8.5"

docker version
docker info
docker run --rm hello-world
```

R2E Docker images are large. The first verified dev image pulled during setup was about 1.38GB. Avoid pre-pulling full datasets; use `max_samples` while iterating.

## Smoke Tests

One dev sample:

```bash
cd /home/caiting/verl-agent-exp-copy-from-lab-server-20260505
source .venv/bin/activate
python3 -m examples.r2e_code_swe.smoke_r2e_env --split dev_10pr_v1 --max_samples 1
```

Three dev samples:

```bash
python3 -m examples.r2e_code_swe.smoke_r2e_env --split dev_10pr_v1 --max_samples 3
```

These commands initialize R2E `RepoEnv`, start/pull Docker images as needed, run one simple `/testbed` bash command, submit, and save patches/reward/logs.

## Tiny Rollout/Eval

```bash
python3 -m examples.r2e_code_swe.run_r2e_rollout_eval \
  --dataset_name R2E-Gym/R2E-Gym-Lite \
  --split dev_10pr_v1 \
  --max_samples 3 \
  --max_steps 5 \
  --rollout_n 1
```

This validates adapter/logging behavior. It is not a training command.

## Small LoRA Training On V100

Training must use a train split:

```bash
MODEL_PATH=Qwen/Qwen2.5-Coder-1.5B-Instruct \
TRAIN_DATA_SIZE=20 \
GROUP_SIZE=2 \
R2E_MAX_STEPS=8 \
R2E_TRAIN_DATASET=R2E-Gym/R2E-Gym-Lite \
R2E_TRAIN_SPLIT=train \
bash examples/gigpo_trainer/run_r2e_code_swe_lora_v100.sh vllm
```

Scale gradually with `TRAIN_DATA_SIZE=50`, then `100`, and `R2E_MAX_STEPS=12` after smoke and small rollout logs look healthy. The script uses fp16-compatible V100 settings, LoRA rank 8 by default, tiny micro-batches, and rollout group size 2.

### Action Protocol Tuning

The R2E adapter prompts the model to output exactly one XML tool call per turn and to avoid prose, markdown, and multiple tool calls. The safe examples shown to the model are bash inspection, editor view, and submit:

```xml
<function=bash>
<parameter=cmd>pwd && find . -maxdepth 2 -type f | head -100</parameter>
</function>

<function=str_replace_editor>
<parameter=command>view</parameter>
<parameter=path>/testbed/aiohttp/http_parser.py</parameter>
</function>

<function=submit></function>
```

Placeholder examples are intentionally not shown because small models tend to copy them. Initial prompts also tell the model to inspect the repository with `bash` before submitting.

For short GRPO action-format checks, prefer low-variance sampling:

```bash
TRAIN_TEMPERATURE=0.3 TRAIN_TOP_P=0.9 MAX_RESPONSE_LENGTH=384 \
bash examples/gigpo_trainer/run_r2e_code_swe_lora_v100.sh \
  algorithm.adv_estimator=grpo \
  trainer.total_training_steps=5
```

Use `MAX_RESPONSE_LENGTH=256` when the model keeps writing explanations; use `384` when valid tool calls are truncated.

On V100, keep the rollout dtype at `float16`. The LoRA helper defaults `ROLLOUT_DTYPE=float16`; overriding it to `bfloat16` will fail during vLLM initialization on compute capability 7.0 GPUs.

### Iterative Repair Path

Use one repair per short run. Pick the most blocking or most visible issue, patch only that area, run a short check, then compare metrics before starting the next repair.

Current path:

1. Action protocol: hard one-tool prompt, safe bash-only example, strict multiple-tool rejection, and invalid-action memory cleanup.
2. V100 startup: force rollout dtype to `float16` before judging action quality.
3. Docker startup: ensure the training SSH session has `docker` group access before judging tool behavior.
4. Bash shorthand: accept `<function=bash>cmd</function>` as `cmd` because Qwen often emits this exact one-tool call after seeing the allowed envelope.
5. Final-step submit prompt: when the current prompt is for the last allowed step, instruct the model to emit only `<function=submit>...</function>` so late malformed outputs do not become projection-level invalid actions.
6. Uncertain-step bash fallback: tell the model to use the safe bash inspection action when it is unsure, instead of emitting prose or a no-tool response.
7. Raw invalid response logging: attach clipped raw model text and projection validity to R2E trajectory actions so the next repair can distinguish prose, empty output, and malformed XML instead of guessing from a generic parser error.
8. Markdown-fence parser tolerance: unwrap single fenced `json` tool calls and treat fenced `bash` snippets as bash commands, because the dominant raw invalid responses in longer runs are otherwise well-formed tool attempts wrapped in markdown.
9. Malformed editor parameter tolerance: accept `<parameter>old_str=...</parameter>` and `<parameter>command>view</parameter>` style tags, because the remaining invalid editor calls often use this shape instead of `<parameter=old_str>...</parameter>`.
10. No-prose prompt hardening: explicitly ban planning prose starts such as `To address`, `Here is`, and `step-by-step`, because the remaining invalid raw responses after malformed-parameter tolerance are mostly explanatory text instead of tool calls.
11. Key/value parameter-tag tolerance: accept `<parameter=command=view</parameter>`, `<parameter=path=/testbed/file</parameter>`, and `<parameter>file_text="..."</parameter>` style editor arguments, because these became the dominant invalid editor-tool shape after no-prose prompt hardening.
12. Editor CLI shorthand tolerance: convert bash payloads like `str_replace_editor view /testbed/file.py` and `str_replace_editor replace /testbed/file.py old new` into structured `str_replace_editor` actions, because v17's most frequent execution invalids are model attempts to call the editor with SWE-agent-style positional CLI syntax.
13. Rollout I/O audit logging: write each train-step rollout episode step as JSON under `v_x/train_step_<global_step>/episode_<episode>/step_<step>.json`, including model input, raw output, parsed action, tool observation, reward, done, and clipped info. This is the next repair once action validity is good enough to inspect patch-quality failures.
14. Action-quality gate: compare `episode/valid_action_ratio`, `episode/tool_call_count/mean`, `bash` tool count, and natural-language response count.
15. Action-validity metric split: keep `episode/valid_action_ratio` focused on parser/protocol/safety validity, and record command/editor runtime failures separately as `tool_execution_success=false` with `fail_reason`, matching R2E-Gym's `RepoEnv.step` pattern where execution output is observation rather than an action-format invalid.
16. JSON schema variant tolerance: accept single-call JSON variants such as `{"tool": "str_replace_editor", "command": "view", ...}` and `{"tool_call": {"tool_name": ..., "parameters": ...}}`, because v20 missed the 0.88 gate mainly due to a few already-structured JSON calls using a different schema.
17. XML parameter schema tolerance: accept `str_replace_editor` XML variants that use direct child tags, `<parameters>...</parameters>`, comma-separated key/value bodies, or sequential `<parameter>key</parameter><parameter>value</parameter>` pairs. This remains a protocol repair, not a patch-quality repair.
18. R2E-compatible action normalization: keep verl-agent as the rollout/training owner, but normalize model text using R2E-Gym's action conventions before passing a single `{tool_name, parameters}` object into the env. The adapter now accepts `action` JSON aliases, single OpenAI-style `tool_calls`, prose-wrapped fenced JSON tool calls, `/repo` and relative paths mapped into `/testbed`, `/testbed/file.py:line` path suffixes converted into `view_range`, and symbolic editor `view_range` searches converted into safe `bash grep -RIn ... | head -50` commands. This is an action-protocol bridge, not use of R2E-Gym's Agent for inference.
19. Strict XML prompt and clean R2E history serialization: keep the user's `<|im_start|>user` task prompt focused on exactly one XML tool call, place the strict reminder after the current observation, and serialize previous actions back as canonical XML `Tool call:` blocks instead of Python dicts or `Action N:` labels. This reduces raw-format drift without relaxing the parser or changing R2E-Gym execution semantics.
20. Bash shell execution wrapper: execute bash tool payloads through `bash -lc <quoted command>` inside the R2E Docker workspace. This preserves shell builtins and composition such as `cd ... && ...`, pipes, redirects, and variables, while keeping protocol validity separate from command exit status.
21. Reward shaping and source-edit gating: keep R2E-Gym `_calculate_reward` as the terminal reward, but add small potential-based process reward for repository exploration, issue-relevant search, unique source views, successful source edits, source patches, validation after edit, and clean submit. Test files, reproduction scripts, and R2E/runtime auxiliary files are classified separately; only successful source edits unlock voluntary submit. Test and R2E auxiliary edits are blocked before Docker execution, while reproduction scripts may be created but do not count as a source fix.
22. Next repair is chosen only after the previous short run finishes or fails with a clear blocker.

Observed action-protocol checkpoints:

- Before Docker group access, official action metrics stayed at zero because R2E workspace setup failed even for parsed tool calls.
- After Docker group access, a 3-step short run reached trajectory valid ratio about `0.646`, with bash/editor calls executing in Docker. The remaining dominant parser error was missing `cmd` for `<function=bash>cmd</function>`.
- After accepting bash body shorthand, trajectory valid ratio reached `0.875` on the next short run before the actor update OOMed. This confirms the protocol fix, but official train-step metrics need the V100 memory issue fixed next.
- Do not use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` with the current vLLM path; it conflicts with vLLM's memory pool.
- After the V100 memory repair, dual-GPU `r2e_action_protocol_v9_memfix` completed one GRPO step without OOM. Official `episode/valid_action_ratio` was `0.625`, `episode/tool_call_count/mean` was `3.25`, `prompt_length/clip_ratio` and `response_length/clip_ratio` were both `0.0`, and trajectory parsing showed no prose-start responses. The gap came from projection-level malformed final-step outputs, so the next single repair is the final-step submit prompt above.
- After the final-step submit prompt, `r2e_action_protocol_v10_final_submit` completed without OOM. Official `episode/valid_action_ratio` improved to `0.688`, `episode/tool_call_count/mean` stayed `3.25`, both clip ratios stayed `0.0`, and submit calls appeared on final steps. The remaining single issue is no-tool malformed output on intermediate uncertain steps, so the next repair is the bash fallback rule above.
- After the bash fallback rule, `r2e_action_protocol_v11_bash_fallback` completed without OOM. Official `episode/valid_action_ratio` remained `0.688`, `episode/tool_call_count/mean` stayed `3.25`, both clip ratios stayed `0.0`, and trajectory-derived train validity was `0.8125`. The remaining blocker is that invalid trajectory entries only expose a generic parser error, not the raw model text, so the next single repair is raw invalid response logging above.
- After raw invalid response logging, `r2e_action_protocol_v13_longcfg5` ran the longer-config 5-step check without OOM. Official valid action ratio stayed below the `0.7` target across steps (`0.562`, `0.500`, `0.643`, `0.643`, `0.667`), while trajectory-derived valid ratio was `0.76`. Raw invalid samples showed many markdown-fenced JSON/bash tool attempts and some prose, so the next single repair is markdown-fence parser tolerance above.
- After markdown-fence parser tolerance, `r2e_action_protocol_v14_fence_parser` completed the 5-step longer-config run without OOM. Official valid action ratio improved, reaching `0.714` at step 3 and `0.786` at step 5, with prompt and response clip ratios at `0.0`. The remaining single parser issue is malformed editor parameter tags such as `<parameter>old_str=...</parameter>`, so the next repair is malformed editor parameter tolerance above.
- After malformed editor parameter tolerance, `r2e_action_protocol_v15_param_tag_parser` completed the 5-step longer-config run without OOM. Official valid action ratio was `0.750`, `0.750`, `0.688`, `0.750`, `0.750`; trajectory-derived valid ratio was `0.74`, with `37` bash calls and no Docker setup errors. The remaining most visible issue is residual natural-language planning responses such as `To address...`, so the next single repair is the no-prose prompt hardening above.
- After no-prose prompt hardening, `r2e_action_protocol_v16_no_prose` completed the 5-step longer-config run without OOM, but action quality did not improve: official valid action ratio was `0.750`, `0.688`, `0.500`, `0.562`, `0.643`, with trajectory-derived valid ratio around `0.69`. Prose remained in a few samples, but the most visible single invalid shape was key/value editor parameter tags such as `<parameter=command=view</parameter>` and `<parameter>command=create</parameter>`, so the next repair is key/value parameter-tag tolerance above.
- After key/value parameter-tag tolerance, `r2e_action_protocol_v17_kv_param_parser` completed the 5-step longer-config run without OOM. Official valid action ratio was `0.750`, `0.688`, `0.688`, `0.562`, `0.938`; trajectory-derived valid ratio was `0.75`, with `42` bash calls, `33` editor calls, and no Docker setup errors. The remaining most frequent execution invalid is bash-wrapped editor CLI shorthand such as `str_replace_editor view /testbed/Orange/data.py`, so the next single repair is editor CLI shorthand tolerance above.
- After editor CLI shorthand tolerance, `r2e_action_protocol_v18_editor_cli_shorthand` completed the 5-step longer-config run without OOM. Official valid action ratio was `0.750`, `0.750`, `0.750`, `0.688`, `0.750`; trajectory-derived valid ratio was `0.78`, with `33` bash calls, `43` editor calls, and no Docker setup errors. The next single repair was rollout I/O audit logging so invalid sources could be inspected at the prompt/output/tool-result level.
- After rollout I/O audit logging, `r2e_action_protocol_v19_rollout_io` completed without OOM. Official valid action ratio averaged `0.7376`, with per-step values `0.750`, `0.750`, `0.750`, `0.688`, `0.750`; rollout I/O logs showed `75` valid and `25` invalid steps. The dominant root cause was metric conflation: `14` invalid steps were well-formed bash/editor calls that failed at execution time (`grep` no match, editor path missing, or replacement string missing). R2E-Gym's `RepoEnv.step` reports execution output and timing but does not define these as action-format invalids, so the next single repair is the action-validity metric split above while preserving `tool_execution_success` and `fail_reason` for runtime-quality analysis.
- After the action-validity metric split, `r2e_action_protocol_v20_metric_split` completed without OOM. Official valid action ratio averaged `0.8752`, with per-step values `0.938`, `0.875`, `0.938`, `0.750`, `0.875`; rollout I/O showed `158` valid and `22` invalid steps, with `54` bash calls, `83` editor calls, `22` submit calls, and no Docker setup failures. This narrowly missed the user target of `0.88`. The next single valid-ratio repair is JSON schema variant tolerance, because three invalid steps were already structured tool calls using `tool` or `tool_call` instead of the canonical `tool_name`/`parameters` schema.
- After JSON schema variant tolerance, `r2e_action_protocol_v21_json_schema` completed without OOM but did not clear the gate. Official valid action ratio averaged `0.8626`, with per-step values `0.938`, `0.875`, `0.938`, `0.812`, `0.750`; rollout I/O showed `162` valid and `18` invalid steps, with no Docker setup failures. The largest remaining parseable protocol cluster is malformed XML parameter schemas, so the next single repair is XML parameter schema tolerance above.
- After XML parameter schema tolerance, `r2e_action_protocol_v22_xml_params` completed without OOM and cleared the user's `0.88` valid-action gate. Official valid action ratio averaged `0.9502`, with per-step values `1.000`, `0.875`, `0.938`, `0.938`, `1.000`; rollout I/O still showed zero task success. Manual inspection found two action-protocol quality issues that hurt long-run patch quality despite high validity: markdown-fenced JSON using an `action` key, and editor paths such as `/testbed/tests/test_http_parser.py:272` being treated as literal file paths before immediate submit. The next manual repair is R2E-compatible action normalization above.
- After R2E-compatible action normalization, `r2e_action_protocol_v23_r2e_action_normalizer` completed without OOM and kept the official valid-action gate cleared with mean `0.9378`. Rollout I/O still showed many raw outputs were non-canonical before parser normalization: markdown fences, prose starts, and history-shaped `Action` labels. The next manual repair is strict XML prompt and clean R2E history serialization above, so the model sees canonical XML examples and previous tool calls instead of Python dict history.

- After aggressive pre-longrun v1 step inspection, success remained zero even with high parser validity. The manual repair removes literal Markdown fence tokens from the actor prompt and sanitizes issue/history/observation text so problem statements with code blocks do not prime fenced XML. It also normalizes common R2E/SWE editor variants: `command=view_range` becomes `command=view` with `view_range`, `range/start_line/end_line` are converted to line ranges, malformed `old_str</value>` and three-body `old_str/value/new_value` forms are recovered, and empty or placeholder `old_str` is rejected before Docker execution. Finally, voluntary `submit` is blocked until at least one successful source edit, and identical failed actions are blocked after one repeat to reduce no-edit and local failure loops. These are verl-agent adapter/prompt repairs; R2E-Gym still owns Docker execution, patch export, and reward.
- After aggressive pre-longrun v2 step inspection, `episode/valid_action_ratio` was high but success stayed zero because trajectories often submitted with no source edit, repeatedly viewed files, edited `/testbed/test.py`, or modified `tests/*`. The next repair adds reward shaping and source-edit gating so RL receives small positive feedback for source-directed repair progress and small negative feedback for blocked submit/test-edit/repeated-failure loops. Terminal R2E test reward remains the main objective.

### Reward Shaping

The R2E adapter returns:

```text
step_reward = terminal_r2e_reward + shaping_delta + penalty
```

`terminal_r2e_reward` still comes from R2E-Gym `_calculate_reward` on submit or auto-submit. The shaping component is potential-based and capped so it cannot dominate a real test pass:

- Positive cap: `+0.25`
- Negative cap: `-0.20`
- Repository exploration: `+0.02`
- Issue-relevant search: `+0.03`
- Unique source file view: `+0.02`, capped at `+0.06`
- First successful source edit: `+0.08`
- Additional source edits: `+0.03`, capped at `+0.06`
- Source patch present at submit: `+0.03`
- Validation command after source edit: `+0.04`
- Clean submit after source edit: `+0.02`

Penalties are small and behavior-specific:

- Submit before successful source edit: `-0.04`
- Test file edit attempt: `-0.08`
- R2E/runtime auxiliary edit attempt: `-0.08`
- Repeated failed action: `-0.03`
- Max-step auto-submit without source edit: `-0.06`

Rollout I/O and trajectory info include `terminal_r2e_reward`, `shaping_reward`, `total_reward`, `successful_source_edit_count`, and `reward_breakdown` with `phi_before`, `phi_after`, `events`, `shaping_delta`, and `penalty`.

Rollout I/O audit logging is disabled by default because it can generate many JSON files. Enable it per run with:

- `env.r2e_code_swe.rollout_io.enabled=true`
- `env.r2e_code_swe.rollout_io.version=v19_rollout_io`
- `env.r2e_code_swe.rollout_io.log_dir=experiments/logs/r2e_code_swe/rollout_io`

Each step log includes clipped `model_input`, `raw_model_output`, `parsed_action`, `tool_observation`, `reward`, `done`, and `info`. Reward-only dataset fields such as `expected_output_json`, `gold_patch_optional`, and `parsed_commit_content` are stripped from the logged info.


### HF Proxy And Offline Model Loading

`examples/gigpo_trainer/run_r2e_code_swe_lora_v100.sh` clears loopback proxy variables such as `HTTP_PROXY=http://127.0.0.1:*` before data/model startup. This prevents Codex-local proxy settings from leaking through `sg docker -c` into Ray workers, where PEFT may otherwise retry `huggingface.co/.../config.json` HEAD requests and stall training logs with `ProxyError`.

Before launching `main_ppo`, the script defaults to cached/offline model loading:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

This is intended for long R2E runs where the model has already been downloaded to `/home/caiting/.cache/huggingface`. Set `R2E_HF_MODEL_OFFLINE=false` only when intentionally downloading a missing model. Set `R2E_CLEAR_LOCAL_PROXY=false` only if a working proxy is actually running on the remote server.

V100 memory defaults in `examples/gigpo_trainer/run_r2e_code_swe_lora_v100.sh` are intentionally conservative:

- `ROLLOUT_DTYPE=float16`
- `ACTOR_MODEL_DTYPE=float16`
- `ACTOR_MP_PARAM_DTYPE=fp16`
- `ENABLE_ACTIVATION_OFFLOAD=True`
- `ACTOR_USE_TORCH_COMPILE=False`
- `ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=8192`
- `ROLLOUT_GPU_MEMORY_UTILIZATION=0.45`
- `ROLLOUT_FREE_CACHE_ENGINE=True`

For Qwen2.5-Coder-3B on two V100-32GB cards, prefer `GPU_COUNT=2 TENSOR_MODEL_PARALLEL_SIZE=2` while debugging. Single-card runs are useful for parser checks, but actor updates can OOM after tool calls start producing real repository context.

The actor `fsdp_config.model_dtype` and `fsdp_config.mixed_precision` fields are not present in the base Hydra config, so the helper appends them with `+...` overrides. If those overrides fail with `Key ... is not in struct`, check the script before rerunning; the failure happens before rollout and does not measure action quality.

## SWE-Bench-Lite Test Evaluation

```bash
python3 -m examples.r2e_code_swe.run_r2e_rollout_eval \
  --dataset_name R2E-Gym/SWE-Bench-Lite \
  --split test \
  --max_samples 1 \
  --max_steps 5
```

This uses the R2E-Gym Docker/R2E route and saves generated patches and R2E reward/test output. It is not the official SWE-Bench harness score unless you separately feed patches to the official SWE-Bench evaluation harness.

## Implementation Notes

- Reused from R2E-Gym: `RepoEnv`, `DockerRuntime`, Docker image metadata, `/testbed`, `str_replace_editor`, patch export, and `_calculate_reward`.
- Implemented in verl-agent: dataset normalizer, split guard, prompt/memory truncation, action parser, gym-style vector adapter, trajectory JSONL logging, smoke/eval/training entrypoints.
- Gold patch and `expected_output_json` are kept in task metadata for reward/debugging but are never inserted into prompts.

### Repair Path: v23 Patch Export Filtering

The step40 run `r2e_loop_step40_20260509_143222` completed without OOM or runtime failure, with `episode/valid_action_ratio=1.000`, but `episode/success_rate=0.000`. Rollout I/O showed the next most prominent integration issue was polluted saved patches: R2E Docker runtime exports `git add -A && git diff --cached`, so environment/setup artifacts such as `Makefile`, `install.sh`, `process_*.py`, `r2e_tests`, and dataset symlinks could appear in the adapter patch logs even when the model did not create them.

The adapter now follows the same conservative spirit as R2E-Gym trajectory export (`true_output_patch_only_existing_files`): saved `.patch` files keep only existing source-file edits, while new files and non-source/R2E auxiliary diffs are dropped. The unfiltered patch is still saved as `.raw.patch` for debugging, and submit info records `raw_patch_chars`, `patch_filter_kept_files`, `patch_filter_dropped_files`, and `patch_filter_dropped_file_count`. Terminal reward still comes from R2E-Gym `_calculate_reward`; this change only makes logged/submitted patches and shaping patch checks reflect source edits instead of R2E setup noise.

### Repair Path: v24 R2E Editor Recovery Hint

The completed v23 step40 run had no OOM or runtime crash and kept `episode/valid_action_ratio=1.000`, but success stayed at `0.000`. Rollout I/O showed the most prominent behavior failure was not malformed actions: `editor_error` plus `repeated_failed_action` dominated failed steps, especially repeated `str_replace` calls whose `old_str` matched multiple locations such as `raise IncompatibleContext()`, `TransferEncodingError`, or `BadStatusLine`.

R2E-Gym's own `str_replace_editor` documentation says `old_str` must match exactly one or more consecutive original lines and must be unique; if it is not unique, the replacement is refused. The adapter prompt now includes that R2E rule directly. When the R2E editor returns a `Multiple occurrences` error, the observation now adds a recovery hint telling the model to view the target range and copy a larger consecutive block into `old_str`, and `info.editor_recovery_hint=non_unique_old_str` is logged for rollout analysis. This is a single targeted repair for repeated non-unique `str_replace` failures; action parsing and reward logic are unchanged.

The completed v24 step40 run finished cleanly with `episode/valid_action_ratio=0.990`, no prompt/response clipping, no OOM, and no Docker/R2E setup failure, but `episode/success_rate` remained `0.000`. The active rollout I/O showed that one episode issued repeated `str_replace` edits where `old_str` and `new_str` were identical. R2E accepts this as an edit because the replacement matches exactly once, but it creates no patch; the adapter was incorrectly counting it as `successful_source_edit` and then allowing clean submit shaping with `patch_chars=0`.

The next single repair blocks no-op `str_replace` actions before calling R2E, records `fail_reason=noop_edit`, keeps the action protocol-valid, sets `tool_execution_success=false`, and leaves `successful_source_edit_count` at zero so submit remains gated until a real source edit exists. The prompt also states that identical `old_str` and `new_str` will not count as a source edit. This keeps R2E as the editor/runtime while preventing false positive process rewards in verl-agent.

### Repair Path: v26 Missing Indent `str_replace` Recovery

The completed v25 step40 run finished cleanly with `episode/valid_action_ratio=1.000`, no prompt/response clipping, no OOM, and no Docker/R2E setup failure, but `episode/success_rate` remained `0.000`. The no-op edit false positive was gone (`noop_edit_blocked=0` and no old_str/new_str identical source-edit unlocks), so the next dominant behavior issue moved to R2E editor strict matching: one aiohttp episode alternated successful narrow views with repeated `str_replace` calls whose `old_str` had lost the original Python leading indentation copied from `cat -n` output, producing repeated `No occurrences` editor errors.

R2E-Gym's editor implementation intentionally uses `file_content.count(old_str)` and requires exact text. The adapter now preserves that rule but adds one conservative recovery: when a `str_replace` fails with `No occurrences`, it runs a container-local probe that searches the target file for exactly one line window matching after removing only leading whitespace. If and only if there is exactly one such window, the adapter reconstructs `old_str` and `new_str` with the original file's common leading indentation and retries the same R2E editor command once. Successful recovery records `indent_repair_applied=true`, `indent_repair_line_start`, and `editor_recovery_hint=indent_repaired_old_str`; ambiguous or unmatched cases still return the original editor error plus a hint to copy exact whitespace.

### Repair Path: v27 View-Range Scoped `str_replace`

The completed v26 step40 run finished cleanly with `episode/valid_action_ratio=1.000`, no OOM, and no Docker/R2E setup failure, but `episode/success_rate` was still `0.000`. The v26 indent probe did not trigger on that rollout (`indent_repair_applied=0`). The dominant active failure was again R2E editor strictness, now concentrated in Orange: the model repeatedly emitted `old_str=raise IncompatibleContext()` with `view_range=[1042, 1044]`. R2E correctly rejected the one-line `old_str` because it appears multiple times in the file, and the model then repeated the same failed action until blocked.

The adapter now treats `view_range` on a failed non-unique `str_replace` as a scoped editing intent. It does not relax R2E's final rule. Instead, after a `Multiple occurrences` error, it runs a container-local probe that reads the requested file range. If `old_str` appears inside that range and the expanded range block is unique in the file, the adapter builds a unique `old_str` block from the original file text, applies the requested replacement inside that block, and retries R2E once with the expanded parameters. Successful recovery records `range_repair_applied=true`, `range_repair_line_start`, `range_repair_line_end`, `range_repair_replacement_count`, and `editor_recovery_hint=view_range_scoped_old_str`. Ambiguous ranges still return the original R2E error.

### Repair Path: v28 Bash Pipefail for Pipeline Errors

The completed v27 step40 run finished cleanly with `episode/valid_action_ratio=1.000`, no prompt/response clipping, no OOM, and no Docker/R2E setup failure, but `episode/success_rate` stayed at `0.000`. The v27 scoped range repair did trigger (`range_repair_applied=3`) and removed the previous Orange non-unique edit loop. The next dominant active failure was a bash search loop: one train episode spent all 40 steps on `grep ... /testbed/orange3 | head -50`. The repository path was wrong, but plain shell pipeline semantics returned the status of `head`, so the adapter marked each command as successful even though the observation contained `No such file or directory`.

The adapter now executes user bash through `bash -lc` with `set -o pipefail` prepended. This keeps R2E Docker as the runtime and preserves normal shell features, but makes failures inside pipelines visible as `command_failed`, allowing the existing repeated-failed-action guard and observations to steer the model away from dead search paths.

### Repair Path: v29 Repeated No-Progress View Guard

The completed v28 step40 run finished cleanly with `episode/valid_action_ratio=0.981`, no prompt/response clipping, no OOM, and no Docker/R2E setup failure, but `episode/success_rate` stayed at `0.000` and `episode/length/mean` hit the max of `40.000`. Rollout I/O showed the next most prominent behavior issue was no longer action format: one validation episode viewed `/testbed/aiohttp/http_parser.py` with `view_range=[123, 127]` thirty-seven times in a row, each counted as a successful tool execution even though no source edit happened.

The adapter now records exact `str_replace_editor view` signatures at the current `successful_source_edit_count`. Repeating the same file/range before making a source edit is blocked before calling R2E, remains protocol-valid, records `fail_reason=repeated_no_progress_action`, sets `tool_execution_success=false`, and receives the same small shaping penalty bucket as repeated failed actions. This keeps R2E-Gym's editor/runtime semantics intact while preventing a successful observation-only loop from consuming all 40 steps.
Operational note after v29: the first v29 launcher at `20260509_200523` reached rollout but every workspace setup failed with Docker socket permission denied because the job was started outside the `docker` group. The replacement launcher `r2e_loop_step40_v29b_sgdocker_no_progress_view_20260509_200833` uses `sg docker -c` around the same venv-backed training command so Ray workers inherit Docker access.

### Repair Path: v30 Repeated No-Progress Early Stop

The completed v29b step40 run finished cleanly with `episode/valid_action_ratio=0.985`, no OOM, no Docker/R2E setup failure, and no prompt/response clipping, but `episode/success_rate` remained `0.000`. The v29 view guard worked as intended in the sense that repeated identical views were no longer counted as successful tool executions, but rollout I/O showed the model continued issuing the same blocked view anyway: `repeated_no_progress_action=146`, with several episodes still reaching the 40-step limit.

R2E-Gym's own agent prompt tells the model not to repeat the same failed edit and its agent loop exits when step limits are reached. In the verl-agent adapter, repeated no-progress view loops now escalate instead of burning the full rollout: `max_repeated_no_progress_actions` is configurable (default `3`), and after that many blocked repeats of the same exact `str_replace_editor view` signature at the same source-edit count, the episode ends with `fail_reason=repeated_no_progress_action_limit`, `exit_reason=repeated_no_progress_action_limit`, and `tool_execution_success=false`. The action remains protocol-valid; this is a trajectory-quality/stuck-loop signal, not an action-format penalty.

### Repair Path: v31 Malformed Editor Parameter Recovery

The completed v30 step40 run finished cleanly and the early-stop guard reduced wasted rollout length (`episode/length/mean=13.250` instead of many 40-step loops), with no OOM, no Docker/R2E setup failure, and no prompt/response clipping. `episode/success_rate` was still `0.000`. Rollout I/O showed the next most prominent active blocker was not Docker or reward execution: after viewing the right file range, several episodes attempted a `str_replace` but emitted malformed editor parameters such as `<parameter>old_str</parameter>` with no actual old text or `new_str`, then fell back to repeated views and hit the no-progress limit.

R2E-Gym's editor requires `old_str` and `new_str` content, and verl-agent must keep a strict one-tool-call action protocol. The adapter now makes that failure explicit in the parser error and history: malformed `str_replace` parameter attempts tell the model not to write bare `<parameter>old_str</parameter>` or `<parameter>new_str</parameter>`, and to use named tags with content copied from the viewed file: `<parameter=old_str>...</parameter>` and `<parameter=new_str>...</parameter>`. Invalid-action history now includes the concrete parser error instead of only a generic invalid-response sentence, so the next prompt preserves the actionable recovery signal without exposing reward-only fields or changing R2E execution semantics.

### Repair Path: v32 Repeated Failed-Action Early Stop

The completed v31b step40 run finished cleanly with `episode/valid_action_ratio=0.958`, no OOM, no Docker/R2E setup failure, and no prompt/response clipping, but `episode/success_rate` remained `0.000`. The v31 parser/history hint preserved the concrete malformed `str_replace` error and the run produced more real source edits/submits, but the dominant remaining waste was a repeated failed bash loop: one episode spent all 40 steps around the same failed `grep`, with 37 `repeated_failed_action` blocks. This is analogous to the v30 no-progress view loop, but for failed commands/actions.

The adapter now adds `max_repeated_failed_action_blocks` (default `3`). After the same failed action has already been blocked that many times, the episode ends with `fail_reason=repeated_failed_action_limit`, `exit_reason=repeated_failed_action_limit`, `repeated_failed_action_block_count`, and `tool_execution_success=false`. The action remains protocol-valid; this is a stuck-trajectory signal that prevents long rollouts from spending the full `max_steps` on an action the adapter is already refusing to execute.

### Repair Path: v33 Repeated No-Progress Bash Guard

The completed v32 step40 run finished cleanly with `episode/valid_action_ratio=0.962`, no OOM, no Docker/R2E setup failure, and no prompt/response clipping. The v32 failed-action early stop worked: `episode/length/mean` dropped to `6.500` and the prior 40-step failed grep loop ended after the configured limit. However `episode/success_rate` stayed `0.000` and no patch files were saved. Rollout I/O showed the next single loop class: successful but repeated bash searches, such as the same `grep -RIn -- 'TransferEncodingError' /testbed/aiohttp | head -50`, were counted as successful observations even when repeated before any source edit, then the trajectory moved into repeated view/no-op patterns.

The no-progress guard now covers exact repeated successful bash commands at the same `successful_source_edit_count`, not only repeated `str_replace_editor view` calls. Repeating the same bash command before a source edit is blocked before re-execution, remains protocol-valid, records `fail_reason=repeated_no_progress_action`, `no_progress_tool_name=bash`, and escalates through the existing `max_repeated_no_progress_actions` limit. This keeps R2E command execution unchanged while preventing successful search output from being treated as progress when it is an exact duplicate.

### Repair Path: v34 Workspace Root Anchoring

The completed v33 step40 run finished cleanly with `episode/valid_action_ratio=0.972`, no OOM, no Docker/R2E setup failure, no prompt/response clipping, and shorter stuck trajectories (`episode/length/mean=8.750`). `episode/success_rate` was still `0.000`, and no patch files were saved. Rollout I/O showed that the most prominent active blocker had moved to path anchoring: Orange episodes repeatedly guessed paths such as `/testbed/orange3` or `/testbed/orange3/context_handler.py`, but R2E images place the repository root directly at `/testbed` and the actual package path is discovered from the workspace contents, for example `/testbed/Orange/...`.

R2E-Gym's own prompts state that the repository is at `/testbed` and that the current working directory is already `/testbed`. The adapter now carries that same assumption explicitly. On reset, after R2E `RepoEnv` starts and the `str_replace_editor` command is installed, the initial observation includes a configurable root preview produced by R2E's own editor directory view: `str_replace_editor view /testbed`. The prompt and repeated-failure hints now tell the model not to assume a subdirectory named after `repo_name` exists, and to search from `.` or from paths shown by the workspace preview. The preview length is controlled by `env.r2e_code_swe.runtime.workspace_overview_max_chars` and defaults to `4000`; set it to `0` to disable the extra initial context.

### Repair Path: v35 Post-Edit Validation Before Submit

The completed v34b step40 run finished cleanly with `episode/valid_action_ratio=1.000`, no OOM, no Docker/R2E setup failure, no prompt/response clipping, and no `/testbed/orange3` wrong-path repeats. The workspace preview repair worked: the first Orange action searched `/testbed/Orange`, and the run produced a real filtered source patch. However `episode/success_rate` remained `0.000`. Rollout I/O showed the next single behavior problem: after one successful source edit, the model immediately called `submit` without running any reproduction or unit test. The terminal R2E tests then failed, but the episode had already ended, so the model never received failure details as an observation it could use to revise the patch.

R2E-Gym's official prompts explicitly require reproduce/verify/unit-test steps and warn not to submit until confident. The adapter now enforces that workflow while keeping R2E as the reward/runtime: `require_validation_before_submit` defaults to `true`, and a manual submit after a successful source edit is blocked until the agent has run a focused bash validation command after that edit. Validation commands are recognized with the same test-like patterns used by shaping (`pytest`, `python -m pytest`, `tox`, `unittest`, `r2e_tests`, `run_tests`, etc.). A failing test command still counts as validation feedback because its output is the signal needed for iterative repair. Auto-submit at the hard max-step limit still goes directly to R2E reward so the episode can terminate. The prompt now states the same rule: edit, run focused validation, inspect failures, then submit.

### Repair Path: v36 Concrete Validation Action After Blocked Submit

The completed v35 step40 run finished cleanly with `episode/valid_action_ratio=1.000`, no OOM, no Docker/R2E setup failure, and no prompt/response clipping. The validation-before-submit gate worked mechanically: premature submit after a source edit was blocked with `fail_reason=submit_before_validation_after_source_edit`. However, rollout I/O showed `validation_after_source_edit_count=0` for every episode and `validation_cmds=0`: after the blocked submit, the model either repeated `submit` or drifted back into search/view/edit loops instead of running a test command. One part of the problem was that the guidance still contained a placeholder-like command (`python -m pytest <relevant_test>`), which violates the action-protocol hardening rule to avoid placeholder examples.

The adapter now makes the blocked-submit recovery concrete and copy-safe. The prompt no longer contains `<relevant_test>`; it suggests `python -m pytest -q`, `pytest -q`, or `python reproduce_issue.py`. If the model repeats `submit` after a validation-required block, the repeated-failure guard no longer replaces the message with a generic "do not repeat failed action" hint. Instead it returns the same validation-specific observation, including an exact XML bash action:

```text
<function=bash>
<parameter=cmd>python -m pytest -q</parameter>
</function>
```

This keeps `submit` protocol-valid but blocked, preserves R2E as the eventual terminal reward path, and gives the actor a clear next tool action that can produce post-edit failure output for iterative repair.

### v37 Validate Tool And Submit Action Mask

The completed v35 run showed `episode/valid_action_ratio=1.000` but `success_rate=0.000`: after successful source edits, the model kept calling `submit` or drifting back into search/view loops instead of running validation. The adapter now exposes `validate` as a first-class XML tool and adds a state-aware action mask to both prompt text and step `info`. Initial prompts allow only `bash` and `str_replace_editor`; after a successful source edit, `validate` becomes allowed and `submit` remains masked; after a post-edit validation command reaches the R2E bash runtime, `submit` becomes allowed. The final forced step still allows submit.

Canonical validation format:

<function=validate>
<parameter=cmd>python -m pytest -q</parameter>
</function>

The parser accepts `validate` with the same strict XML/JSON action normalization as other tools, and env execution runs it inside the R2E Docker workspace via the existing bash runtime. Non-test-like validate commands are protocol-valid but execution-masked with `fail_reason=validate_requires_test_command`. Validate before any source edit is execution-masked with `fail_reason=validate_before_successful_source_edit`. Existing bash-based test commands still count as validation for backward compatibility, but the prompt now guides the model to use the explicit validate tool.
