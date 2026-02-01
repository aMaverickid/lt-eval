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
                    logging.warning(f"Found {key} tag but content is empty.")
                    return key, {}

                logging.info(f"Extracted {key}: reasoning_len={len(detailed_reasoning)}, summary_len={len(raw_inner_content)}")
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
                        logging.info(f"Extracted {key} id: {step_id}")
                        return key, {"step_id": step_id}
                    else:
                        logging.warning(f"Failed to extract Step ID from {key}. Content: '{raw_inner_content}'")
                        return key, {}
                except Exception as e:
                    logging.error(f"Error parsing ID for {key}: {e}", exc_info=True)
                    return key, {}
            
            # Case 3: Final Answer
            # 直接返回内部原始内容
            else: 
                logging.info(f"Extracted {key} content: ... {raw_inner_content[-50:]}")
                return key, {"final_answer": raw_inner_content}

    # Default fallback (如果没找到任何标签)
    # 默认行为是将所有文本视为 detailed_reasoning，但在 commit 状态下等待 summary
    key = "commit"
    logging.warning(f"No XML tags found in model output. Defaulting to '{key}' with all text as reasoning.")
    return key, {
        "detailed_reasoning": text.strip(), 
        "summary": ""
    }