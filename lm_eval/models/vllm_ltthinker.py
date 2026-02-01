import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance
from lm_eval.models.utils import normalize_gen_kwargs
# from lm_eval.models.utils_ltthinker import parse_model_output
from vllm import SamplingParams
from lm_eval.models.vllm_causallms import VLLM

eval_logger = logging.getLogger(__name__)

class ReasoningStep:
    def __init__(self, detailed_reasoning: str, step_id: int, summary: str):
        self.step_id = step_id
        self.detailed_reasoning = detailed_reasoning
        self.summary = summary
        self.expanded: bool = False

    def mark_expanded(self) -> None:
        self.expanded = True
    
    def mark_folded(self) -> None:
        self.expanded = False

    def is_expanded(self) -> bool:
        return self.expanded
    
class ScratchPad:
    """Store intermediate reasoning steps."""
    def __init__(self):
        self.steps: List[ReasoningStep] = []

    def add_step(self, detailed_reasoning: str, summary: str) -> None:
        if not detailed_reasoning or not summary:
            return
        id = len(self.steps) + 1
        step = ReasoningStep(detailed_reasoning, id, summary)
        self.steps.append(step)

    def expand_step(self, step_id: int) -> None:
        if 0 < step_id <= len(self.steps):
            self.steps[step_id - 1].mark_expanded()
    
    def fold_step(self, step_id: int) -> None:
        if 0 < step_id <= len(self.steps):
            self.steps[step_id - 1].mark_folded()

    def build_history_text_for_prompt(self) -> str:
        history_lines = []
        for step in self.steps:
            if step.is_expanded():
                history_lines.append(f"[Expanded Step {step.step_id}] {step.detailed_reasoning}\n")
            else:
                history_lines.append(f"[Step {step.step_id}] {step.summary}\n")
        return "".join(history_lines).strip()

def parse_model_output(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parses the model output text to extract structured information.
    
    The function looks for specific XML-like tags in the text:
    - "... \n\n<commit> ... </commit>"
    - "... \n\n<expand> Step N </expand>"
    - "... \n\n<fold> Step N </fold>"
    - "\n\n<final_answer> ... </final_answer>"

    Depending on the tag found, it extracts and parses the outer and inner content.

    Args:
        text (str): The model output text.
    Returns:
        Tuple[str, dict]: A tuple containing the type of content extracted 
                         ("commit", "expand", "fold", "final_answer") and the parsed content as a dictionary. 
                         ("detailed_reasoning": outer raw content, "summary": inner raw content) for "commit", 
                         ("step_id": N) for "expand" and "fold", 
                         and {"final_answer": raw inner content} for "answer".    
    """ 
    # 定义匹配模式，注意顺序（通常 answer 优先级最高，其次是结构化操作，最后是通用提交）
    patterns = {
        "final_answer": r"<answer>\s*(.*?)\s*(?:</answer>|$)",
        "expand": r"<expand>\s*(.*?)\s*(?:</expand>|$)",
        "fold": r"<fold>\s*(.*?)\s*(?:</fold>|$)",
        "commit": r"<commit>\s*(.*?)\s*(?:</commit>|$)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            raw_inner_content = match.group(1).strip()
            
            # Case 1: Commit
            if key == "commit":
                detailed_reasoning = text[:match.start()].strip()
                
                if not detailed_reasoning and not raw_inner_content:
                    eval_logger.warning(f"Found {key} tag but content is empty.")
                    return key, {}

                eval_logger.info(f"Extracted {key}: reasoning_len={len(detailed_reasoning)}, summary_len={len(raw_inner_content)}")
                return key, {
                    "detailed_reasoning": detailed_reasoning, 
                    "summary": raw_inner_content
                }
            
            # Case 2: Expand / Fold
            # 需要从 "Step N" 中解析出 N
            elif key in ["expand", "fold"]:
                try:
                    # 使用正则提取数字，支持 "Step 1", "Step1", 或者仅仅是 "1"
                    id_match = re.search(r"(?:Step\s*)?(\d+)", raw_inner_content, re.IGNORECASE)
                    if id_match:
                        step_id = int(id_match.group(1))
                        eval_logger.info(f"Extracted {key} id: {step_id}")
                        return key, {"step_id": step_id}
                    else:
                        eval_logger.warning(f"Failed to extract Step ID from {key}. Content: '{raw_inner_content}'")
                        return key, {}
                except Exception as e:
                    eval_logger.error(f"Error parsing ID for {key}: {e}", exc_info=True)
                    return key, {}
            
            # Case 3: Final Answer
            # 直接返回内部原始内容
            else: 
                eval_logger.info(f"Extracted {key} content: ... {raw_inner_content[-50:]}")
                return key, {"final_answer": raw_inner_content}

    # Default fallback (如果没找到任何标签)
    # 默认行为是将所有文本视为 detailed_reasoning，但在 commit 状态下等待 summary
    key = "commit"
    eval_logger.warning(f"No XML tags found in model output. Defaulting to '{key}' with all text as reasoning.")
    return key, {
        "detailed_reasoning": text.strip(), 
        "summary": ""
    }

@register_model("vllm_ltthinker")
class VLLMLTThinker(VLLM):
    def __init__(
        self,
        max_turns: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_turns = max_turns    
        self.interactive_stop_words = ["</commit>", "</expand>", "</fold>", "</answer>"]
        
        eval_logger.info(f"Initialized VLLMLTThinker with max_turns={self.max_turns}")

    
    def _inject_scratchpad(self, original_prompt: str, scratchpad_text: str, force_instruction: str = "") -> str:
        """Inject the scratchpad history into the original prompt."""
        
        injection_content = ""
        if scratchpad_text:
            injection_content += f"\n### Current Scratchpad:\n{scratchpad_text}"
        if force_instruction:
            injection_content += f"\n{force_instruction}"

        if not injection_content:
            return original_prompt
        
        # "Unapply" chat template if any
        # 1. ChatML format
        if "<|im_end|>" in original_prompt:            
            parts = original_prompt.rsplit("<|im_end|>", 1)            
            return f"{parts[0]}{injection_content}<|im_end|>{parts[1]}"
        
        # 2. Llama-3
        elif "<|eot_id|>" in original_prompt:
            parts = original_prompt.rsplit("<|eot_id|>", 1)
            return f"{parts[0]}{injection_content}<|eot_id|>{parts[1]}"
        
        # 3. fallback: direct append
        return f"{original_prompt}{injection_content}"

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
                        
        # 1. Initialize states for all requests
        # context, all_gen_kwargs = zip(*(req.args for req in requests), strict=True)
        states = []
        for i, req in enumerate(requests):
            prompt_text, gen_kwargs = req.args

            gen_kwargs = normalize_gen_kwargs(gen_kwargs=gen_kwargs, default_max_gen_toks=self.max_gen_toks)

            user_until = gen_kwargs.get("until", [])
            if isinstance(user_until, str):
                user_until = [user_until]
            combined_until = list(set(user_until + self.interactive_stop_words))

            states.append({
                "req_idx": i,
                "original_prompt": prompt_text,  # 保存原始请求的 Prompt
                "gen_kwargs": gen_kwargs,
                "until": combined_until,
                "scratchpad": ScratchPad(),
                "turn": 1,
                "finished": False,
                "final_text": "",
                "forcing": False
            })

        active_indices = list(range(len(states)))

        pbar = tqdm(
            total=len(requests), 
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Interactive Inference"
        )

        # 2. Main generation loop
        while active_indices:
            batch_token_ids = []
            batch_map = []
            batch_sampling_params = []

            for state_idx in active_indices:
                state = states[state_idx]

                # A. Build the prompt with scratch pad history
                scratchpad_text = state["scratchpad"].build_history_text_for_prompt()
                force_text = ""

                if state["forcing"]:
                    force_text = "SYSTEM NOTIFICATION: Max turns reached. Output <answer>...\\boxed{RESULT}</answer> immediately."

                current_prompt = state["original_prompt"]
                
                if state["turn"] == 1:
                    current_prompt_text = state["original_prompt"]
                else:
                    current_prompt_text = self._inject_scratchpad(
                        original_prompt=current_prompt,
                        scratchpad_text=scratchpad_text,
                        force_instruction=force_text
                    )
                
                # B. Tokenize the current prompt
                token_ids = self.tok_encode(current_prompt_text)

                # C. truncate if necessary
                # eval_logger.info(f"max length: {self.max_length}, current prompt length: {len(token_ids)}")
                max_ctx = self.max_length - state["gen_kwargs"].get("max_tokens", 10240)
                if len(token_ids) > max_ctx:
                    token_ids = token_ids[-max_ctx:]

                batch_token_ids.append(token_ids)
                batch_map.append(state_idx)

                # D. Sampling params

                kwargs, _, max_gen = self.modify_gen_kwargs(
                    state["gen_kwargs"], 
                    eos=self.tokenizer.decode(self.eot_token_id),
                    default_max_gen_toks=10240
                )

                params = SamplingParams(
                    max_tokens=max_gen, 
                    stop=state["until"], 
                    **kwargs
                )

                batch_sampling_params.append(params)

            if not batch_token_ids:
                break
                
            # 3. Generate outputs for the batch
            outputs = self._model_generate(
                requests=batch_token_ids,
                generate=True,
                sampling_params=batch_sampling_params
            )

            # 4. Update states based on outputs
            next_active_indices = []

            for batch_i, output in enumerate(outputs):
                state_idx = batch_map[batch_i]
                state = states[state_idx]

                generated_text = output.outputs[0].text.strip()

                # Parse the model output
                action, content = parse_model_output(generated_text)
                is_finished = False

                if action == "final_answer":
                    if isinstance(content, dict) and "final_answer" in content:
                        state["final_text"] = content["final_answer"]
                    else:
                        state["final_text"] = generated_text
                    is_finished = True

                elif action == "commit":
                    if isinstance(content, dict):
                        state["scratchpad"].add_step(
                            detailed_reasoning=content.get("detailed_reasoning", ""),
                            summary=content.get("summary", "")
                        )
                
                elif action == "expand":
                    if isinstance(content, dict) and "step_id" in content:
                        state["scratchpad"].expand_step(content["step_id"])
                
                elif action == "fold":
                    if isinstance(content, dict) and "step_id" in content:
                        state["scratchpad"].fold_step(content["step_id"])

                elif state["forcing"]:
                    # In forcing mode, any output is treated as final answer
                    state["final_text"] = generated_text
                    is_finished = True

                state["turn"] += 1

                if not is_finished:
                    if state["turn"] > self.max_turns and not state["forcing"]:
                        state["forcing"] = True
                        eval_logger.warning(f"Request {state_idx} reached max turns. Forcing final answer in next turn.")
                        next_active_indices.append(state_idx)
                    elif state["turn"] > self.max_turns + 2:
                        # deadloop after forcing, terminate directly
                        state["final_text"] = generated_text
                        state["finished"] = True
                        pbar.update(1)
                    else:
                        next_active_indices.append(state_idx)
                else:
                    state["finished"] = True
                    pbar.update(1)

            active_indices = next_active_indices

        pbar.close()

        # 5. Collect final outputs
        results = [state["final_text"] for state in states]
        return results
                        
                    






                








        
        
    