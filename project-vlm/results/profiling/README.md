# Understanding the profiling results

Each folder is one run. Its name contains the model, task, number of measured examples, seed and date.

| File | What it contains |
| --- | --- |
| `per_example_measurements.csv` | One row for each measured example. |
| `summary.csv` | Averages and other summary measurements for the run. |
| `run_settings.json` | Settings, selection seed, GPU and software information. |

Warm-up examples and model-loading time are excluded from reported latency.

## Units

- **ms** means milliseconds: 1,000 ms = 1 second. Lower latency means faster responses.
- Memory columns ending in **`_mb`** actually use **MiB**: 1,024 MiB = 1 GiB.

## Measurements for each example

| Column | Meaning |
| --- | --- |
| `e2e_latency_ms` | Total time for one example: image processing, preparing the model input, GPU generation and converting the output into text. E2E means “end to end.” |
| `prefill_latency_ms` | Time for the initial processing of the image and question, ending after the first model pass. It excludes preparing the input outside the model. |
| `memory_after_load_mb` | GPU memory used by PyTorch just after loading the model, before warm-up. This is the baseline for the run. |
| `peak_allocated_mb` | Highest amount of GPU memory actively used by PyTorch during this example, including the loaded model. |
| `peak_reserved_mb` | Highest amount of GPU memory PyTorch held during this example, including unused space kept for reuse. |
| `additional_inference_memory_mb` | Extra memory above the baseline: `peak_allocated_mb - memory_after_load_mb`. |
| `input_tokens` | Number of input tokens passed to generation. A token is a small piece of text or an image marker. Models represent images differently, so these counts are not directly equivalent across models. |
| `generated_tokens` | Number of tokens generated for the answer, including any end-of-answer token. Tokens are not the same as words. |

Prefill timing starts at slightly different points in LLaVA and Qwen. It is not the exact time until the first answer word appears on screen. Enabling it adds a small amount of measurement overhead.

Reserved memory can include space kept from warm-up or earlier examples. Memory is not cleared between examples. These counters measure PyTorch memory in this process; they do not include all GPU memory used by the driver or other programs.

## Summary measurements

These use **all measured examples** in the run. For example, a 100-example run uses all 100 latencies.

| Column | Meaning |
| --- | --- |
| `num_samples` | Number of measured examples; excludes warm-up. |
| `mean_latency_ms` | Average E2E latency. |
| `median_latency_ms` | Middle E2E latency after sorting the times; with an even number of examples, the average of the two middle times. |
| `p95_latency_ms` | Estimated time within which about 95% of measured examples finish. Calculated from the sorted times, using values between neighboring measurements when needed. More examples give a more reliable estimate. |
| `mean_prefill_latency_ms` | Average prefill latency. |
| `memory_after_load_mb` | The same model-loaded baseline described above. |
| `max_peak_allocated_mb` | Largest `peak_allocated_mb` across measured examples. |
| `max_peak_reserved_mb` | Largest `peak_reserved_mb` across measured examples. |
| `max_additional_inference_memory_mb` | Largest additional memory usage across measured examples. |

## Other columns and settings

| Column | Meaning |
| --- | --- |
| `model` | Model name. |
| `task` | Task used, such as `mme`. |
| `split` | Dataset section used, such as `test`. |
| `sample_id` | Example's row number in that dataset section. |
| `response` | Answer returned by the model. |
| `batch_size` | Examples processed together; this profiler uses 1. |
| `max_new_tokens` | Maximum allowed answer length. The model can stop earlier. |
| `generation_json` | Saved answer-generation settings. |
| `dtype` | Number format used for model weights; `float16` uses 16 bits per number. |
| `attention` | Attention implementation; this setup uses `eager`, the standard implementation. |
| `prefill_instrumented` | Whether prefill timing was enabled. |
| `thinking_enabled` | Whether the model's thinking mode was enabled. It is off for these comparisons. |
| `qwen_max_pixels` | Image-size limit passed to the Qwen wrapper. Empty for LLaVA. |

Some columns may be absent in older runs or when prefill timing is disabled.

The selection seed is saved in `run_settings.json`. To repeat the selection, use the same seed, task, example counts and dataset version. Images are not copied into these result folders.

For comparisons, keep these settings and the answer-length limit the same. Also check `generated_tokens`: longer answers usually take longer to generate.
