"""
方案B: Qwen3-0.6B 微调模型推理
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class CorrectionPredictor:
    """端到端纠错预测器"""

    def __init__(self, model_dir: str):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
        self.model.eval()

    def predict(self, text: str) -> dict:
        """
        输入用户文本，输出标准化实体
        返回: {"cable_name": "AAE-1", "segment": "S1.8", "city": "中国香港"}
        """
        prompt = f"<|im_start|>system\n你是海缆领域实体纠错助手。从用户输入中提取海缆名称、缆段、城市实体，并纠正为标准名称。输出JSON格式。<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
        )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        try:
            # 提取 JSON
            if "{" in response:
                json_str = response[response.index("{"):response.rindex("}") + 1]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        return {"cable_name": None, "segment": None, "city": None}


if __name__ == "__main__":
    model_dir = os.path.join(os.path.dirname(__file__), "saved_model")
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        print("请先运行: python finetune.py --action train")
        sys.exit(1)

    predictor = CorrectionPredictor(model_dir)
    test_texts = [
        "AAE1在S1-8段有故障吗？",
        "APG海缆香港到釜山方向有影响吗？",
    ]
    for text in test_texts:
        result = predictor.predict(text)
        print(f"输入: {text}")
        print(f"输出: {result}\n")
