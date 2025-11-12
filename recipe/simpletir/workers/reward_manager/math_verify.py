# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import signal
from typing import Any, Callable, Dict, List

import ray
import torch
from ray.exceptions import GetTimeoutError
from pathlib import Path

from recipe.simpletir.utils.reward_score import _default_compute_score
from verl import DataProto

import os
import json
import re


# Keep this outside the main wrapper function for clarity and efficiency.
def _timeout_handler(signum, frame):
    """Signal handler function to raise a TimeoutError."""
    # print("Signal handler called!") # Debugging
    raise TimeoutError("Operation timed out!")


@ray.remote
def reward_func_timeout_ray(
    func: Callable, timeout_seconds: int, *args: Any, **kwargs: Any
):
    """A decorator that applies a timeout to the decorated function using signal.

    Args:
        timeout_seconds (int): Number of seconds before timing out the decorated function.
            Defaults to 10 seconds.

    Notes:
        Only works on Unix systems as it uses signal.alarm.
    """
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return func(*args, **kwargs)
    except TimeoutError:
        return {"score": 0.0, "extra_info": {"is_filter": "1"}}
    finally:
        # cancel alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class MathRewardManager:
    """
    The Reward Manager is borrowed from https://github.com/PRIME-RL/PRIME
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, record_dir=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.step = None
        self.timeout_seconds = 100
        self.record_dir = Path(record_dir) / "step_records"
        self.record_dir.mkdir(parents=True, exist_ok=True)

    def math_compute_score_parallel_with_ray(
        self, data_sources, solution_strs, ground_truths, extra_infos
    ):
        scores: List[float] = [0.0] * len(solution_strs)
        extra_info_dict: Dict[
            str, List[float]
        ] = {}  # Key -> list of values for the batch
        print(
            f"Scoring process started over {len(solution_strs)} samples, waiting for results..."
        )

        futures = []
        for i in range(len(solution_strs)):
            ground_truth = ground_truths[i]
            solution_str = solution_strs[i]
            data_source = data_sources[i]
            extra_info = extra_infos[i]

            future = reward_func_timeout_ray.remote(
                self.compute_score,
                self.timeout_seconds,
                data_source,
                solution_str,
                ground_truth,
                extra_info,
            )
            futures.append(future)

        default_fail_score = {
            "score": 0.0,
            "extra_info": {"is_filter": 1},
        }  # Default on error which should be filtered

        for i, future in enumerate(futures):
            try:
                task_result = ray.get(future, timeout=self.timeout_seconds)

                if isinstance(task_result, dict):
                    assert "extra_info" in task_result, (
                        f"Extra info missing in task_result dict for item {i}. Result: {task_result}"
                    )
                    score_result = task_result
                    if "is_filter" not in task_result["extra_info"]:
                        score_result["extra_info"].update({"is_filter": 0})
                elif isinstance(task_result, (int, float)):
                    score_result = {
                        "score": float(task_result),
                        "extra_info": {"is_filter": 0},
                    }
                else:
                    print(
                        f"Unexpected task_result type for item {i}: {type(task_result)}. Using default score. Result: {task_result}"
                    )
                    ray.cancel(future, force=True)
                    score_result = default_fail_score
            except GetTimeoutError:
                print(
                    f"Timeout processing item {i} (gold='{str(ground_truths[i])[:50]}...', target='{str(solution_strs[i])[:50]}...'). Using default score."
                )
                score_result = default_fail_score
            except Exception as e:
                print(
                    f"Error processing item {i} (gold='{str(ground_truths[i])[:50]}...', target='{str(solution_strs[i])[:50]}...'): {e}"
                )
                import traceback

                traceback.print_exc()
                ray.cancel(future, force=True)
                score_result = default_fail_score

            scores[i] = float(score_result.get("score", 0.0))

            if "extra_info" in score_result and isinstance(
                score_result["extra_info"], dict
            ):
                for key, value in score_result["extra_info"].items():
                    if key not in extra_info_dict:
                        extra_info_dict[key] = [0.0] * len(solution_strs)
                    extra_info_dict[key][i] = value

        return scores, extra_info_dict

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        # check the last step index
        if self.step is None:
            last_step_idx = 0
            for file in os.listdir(self.record_dir):
                if self.num_examine == 1:
                    if re.search(r"step-val-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
                else:
                    if re.search(r"step-\d+\.json", file):
                        step_idx = int(file[:-len(".json")].split("-")[-1])
                        if step_idx > last_step_idx:
                            last_step_idx = step_idx
            self.step = last_step_idx + 1
        if data.meta_info.get('global_step', None) is not None:
            self.step = data.meta_info['global_step']

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        already_print_data_sources = {}
        to_save_records = []

        response_ids = data.batch["responses"]
        sequences_strs = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )
        ground_truths = [
            data_item.non_tensor_batch["reward_model"]["ground_truth"]
            for data_item in data
        ]
        data_sources = data.non_tensor_batch["data_source"]
        extra_infos = [
            data_item.non_tensor_batch.get("extra_info", None) for data_item in data
        ]

        assert len(sequences_strs) == len(ground_truths) == len(data_sources)

        # it is very important to use ray to compute score in parallel!
        scores, extra_info_dict = self.math_compute_score_parallel_with_ray(
            data_sources, sequences_strs, ground_truths, extra_infos
        )

        # batched scoring
        prompt_ids = data.batch["prompts"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(
            dim=-1
        )
        data_sources = data.non_tensor_batch["data_source"]

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            prompt_str = self.tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
            response_str = sequences_strs[i]
            ground_truth = ground_truths[i]
            score = scores[i]

            sample_extra_info = {}
            for key, value_list in extra_info_dict.items():
                if i < len(value_list):
                    sample_extra_info[key] = value_list[i]

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                
                # Print sample information
                
                print("=" * 80)
                print(f"[data_source] {data_source}")
                print(f"[prompt] {prompt_str}")
                print(f"[response] {response_str}")
                print(f"[ground_truth] {ground_truth}")
                print(f"[score] {score}")
                
                # Print extra info if available
                for key in extra_info_dict:
                    if i < len(extra_info_dict[key]):
                        print(f"[{key}] {extra_info_dict[key][i]}")
                print("=" * 80)

            to_save_records.append({
                "data_source": data_source,
                "prompt": prompt_str,
                "response": response_str,
                "ground_truth": ground_truth,
                "score": score,
                "extra_info": sample_extra_info,
            })
        
        save_record=True
        if save_record:
            # Save the records to a file
            if self.num_examine == 1:
                temp_file = self.record_dir / f"step-val-{self.step}.json"
            else:
                temp_file = self.record_dir / f"step-{self.step}.json"  
            self.step += 1
            if temp_file.exists():
                with open(temp_file, "r") as f:
                    existing_records = json.load(f)
                existing_records.extend(to_save_records)
                with open(temp_file, "w") as f:
                    json.dump(existing_records, f, indent=4)
            else:
                with open(temp_file, "w") as f:
                    json.dump(to_save_records, f, indent=4)
            print(f"Saved records to {temp_file}")          

        return {"reward_tensor": reward_tensor, "extra_info": extra_info_dict}
