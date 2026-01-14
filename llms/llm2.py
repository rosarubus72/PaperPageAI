import torch
from pathlib import Path
import os
import warnings
from openai import OpenAI

class Qwen_2_5_txt:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = torch.device("cuda:1")
        model_path = "/home/gaojuanru/mnt_link/gaojuanru/twittergenerate/cache/huggingface/Qwen/Qwen2.5-7B-Instruct"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map={"": device},
            torch_dtype=torch.float16
        )            
        self.device = device

    def generate(self, query, images=None):
        # 对于纯文本模型，忽略 images 参数
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            messages = [{"role": "user", "content": query}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=4096*2,
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
            output_text = self.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            del inputs, generated_ids
            torch.cuda.empty_cache()
            return output_text[0]


# class LLM2:
#     def __init__(self, model_name):
#         self.model_name = model_name
#         if 'Qwen2.5-7B-Instruct' in self.model_name:
#             self.model = Qwen_2_5_txt(model_name)
#         elif model_name.startswith('gpt'):
#             from openai import OpenAI
#             self.key = ''  # 替换为你的实际API密钥
#             self.model = OpenAI(base_url="https://api.ai.cs.ac.cn", api_key=self.key)
#         else:
#             raise ValueError(f"不支持的模型: {model_name}")

#     def generate(self,** kwargs):
#         query = kwargs.get('query', '')
#         image = kwargs.get('image', None)  # 纯文本场景可忽略
#         model_name = kwargs.get('model_name', self.model_name)

#         if 'Qwen2.5-7B-Instruct' in model_name:
#             return self.model.generate(query)
#         elif model_name.startswith('gpt'):
#             # 纯文本处理：仅保留文本内容，忽略image参数
#             content = [{"type": "text", "text": query}]
            
#             # 调用gpt-4o-mini生成响应
#             completion = self.model.chat.completions.create(
#                 model="gpt-4o-mini",  # 指定使用gpt-4o-mini
#                 messages=[{"role": "user", "content": content}]
#             )
#             return completion.choices[0].message.content

#     # 新增：定义图像编码函数（即使纯文本场景不用，也避免报错）
#     @staticmethod
#     def _encode_image(filepath):
#         import base64
#         with open(filepath, "rb") as image_file:
#             return base64.b64encode(image_file.read()).decode('utf-8')
class BaseModel:
    """基础模型类"""
    def __init__(self, model_type, model_config_dict=None, url=None):
        self.model_type = model_type
        self.model_config_dict = model_config_dict or {}
        self.url = url
    
    def generate(self, messages, max_tokens=4096*2, temperature=0.7):
        """基础生成方法，需要子类实现"""
        raise NotImplementedError
class ETChatModel(BaseModel):
    """ETChat API模型类"""
    def __init__(self, api_key, model_config_dict=None):
        super().__init__("etchat", model_config_dict)
        self.api_key = api_key
        # 使用指定的URL
        self.client = OpenAI(
            base_url="https://api.ai.cs.ac.cn/v1",  # 使用完整URL，包括/v1路径
            api_key=api_key,
        )
    
    def generate(self, messages, max_tokens=4096*2, temperature=0.7):
        """使用OpenAI客户端调用ETChat API"""
        try:
            # 使用O3模型
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",  # 使用gpt-4o-mini模型
                max_completion_tokens=max_tokens,
            )
            
            content = chat_completion.choices[0].message.content
            
            # 计算token使用量
            usage = chat_completion.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            
            return content, input_tokens, output_tokens
            
        except Exception as e:
            print(f"❌ ETChat API调用失败: {e}")
            raise
class LLM2:
    def __init__(self, model_name = 'Qwen2.5-7B-Instruct'):
        self.model_name = model_name
        if 'Qwen2.5-7B-Instruct' in self.model_name:
            self.model = Qwen_2_5_txt(model_name)
        elif model_name.startswith('gpt'):
            from openai import OpenAI
            self.key = ''  # 替换为你的实际API密钥
            self.model = OpenAI(base_url="https://api.ai.cs.ac.cn", api_key=self.key)
        elif model_name == 'etchat':  # 添加ETChat支持
            self.key = ''  # 替换为你的ETChat API密钥
            self.model = ETChatModel(api_key=self.key)
        else:
            raise ValueError(f"不支持的模型: {model_name}")

    def generate(self, **kwargs):
        query = kwargs.get('query', '')
        image = kwargs.get('image', None)  # 纯文本场景可忽略
        model_name = kwargs.get('model_name', self.model_name)
        max_tokens = kwargs.get('max_tokens', 4096*2)
        temperature = kwargs.get('temperature', 0.7)

        if 'Qwen2.5-7B-Instruct' in model_name:
            return self.model.generate(query)
        elif model_name.startswith('gpt'):
            # 纯文本处理：仅保留文本内容，忽略image参数
            content = [{"type": "text", "text": query}]
            
            # 调用gpt-4o-mini生成响应
            completion = self.model.chat.completions.create(
                model="gpt-4o-mini",  # 指定使用gpt-4o-mini
                messages=[{"role": "user", "content": content}]
            )
            return completion.choices[0].message.content
        elif model_name == 'etchat':  # ETChat处理分支
            # 构造messages格式
            messages = [{"role": "user", "content": query}]
            
            # 调用ETChat API
            content, input_tokens, output_tokens = self.model.generate(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return content

    # 新增：定义图像编码函数（即使纯文本场景不用，也避免报错）
    @staticmethod
    def _encode_image(filepath):
        import base64
        with open(filepath, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')