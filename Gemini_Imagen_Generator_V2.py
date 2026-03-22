import json
import requests
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
import re
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading
import os

class GeminiOpenAIProxyNodeV2:
    """
    ComfyUI节点 V2: Gemini图像生成 - 支持图像尺寸选择
    真并发+流式返回+错误处理+1K/2K/4K选择
    直接从节点输入获取API Key
    """
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "text")
    FUNCTION = "generate_images"
    OUTPUT_NODE = True
    CATEGORY = "image/ai_generation"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "display_name": "API Key"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Hello nano banana!"
                }),
                "api_base_url": ("STRING", {
                    "default": "https://api.qianhai.online"
                }),
                "model_type": ("STRING", {
                    "default": "gemini-3-pro-image-preview"
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 8
                }),
                "aspect_ratio": (["Auto", "1:1", "9:16", "16:9", "21:9", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4"], {
                    "default": "Auto"
                }),
                "image_size": (["Auto", "1K", "2K", "4K"], {
                    "default": "2K"
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": -1, "min": -1, "max": 102400
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01
                }),
                "max_workers": ("INT", {
                    "default": 4, "min": 1, "max": 8
                }),
                "input_image_1": ("IMAGE",),
                "input_image_2": ("IMAGE",),
                "input_image_3": ("IMAGE",),
                "input_image_4": ("IMAGE",),
                "input_image_5": ("IMAGE",),
            }
        }
    
    def tensor_to_base64(self, tensor: torch.Tensor) -> str:
        """将tensor转换为base64"""
        img_array = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def base64_to_tensor_single(self, b64_str: str) -> np.ndarray:
        """将单个base64转换为numpy数组"""
        try:
            img_data = base64.b64decode(b64_str)
            img = Image.open(BytesIO(img_data)).convert('RGB')
            img_array = np.array(img).astype(np.float32) / 255.0
            return img_array
        except Exception as e:
            print(f"⚠️ 图片解码失败: {str(e)}")
            return np.zeros((64, 64, 3), dtype=np.float32)

    def base64_to_tensor_parallel(self, base64_strings: List[str]) -> torch.Tensor:
        """并发解码多张图片"""
        if not base64_strings:
            return torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        
        decode_start = time.time()
        images = []
        
        with ThreadPoolExecutor(max_workers=min(4, len(base64_strings))) as executor:
            future_to_index = {executor.submit(self.base64_to_tensor_single, b64): i 
                             for i, b64 in enumerate(base64_strings)}
            
            results = [None] * len(base64_strings)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    print(f"⚠️ 图片{index+1}解码异常: {str(e)}")
                    results[index] = np.zeros((64, 64, 3), dtype=np.float32)
            
            images = [r for r in results if r is not None]
        
        decode_time = time.time() - decode_start
        print(f"✓ 并发解码 {len(images)} 张图片完成,耗时: {decode_time:.2f}s")
        
        return torch.from_numpy(np.stack(images))
    
    def create_request_data(self, prompt: str, seed: int, aspect_ratio: str, 
                          image_size: str, top_p: float = 0.65, 
                          input_images: List[torch.Tensor] = None) -> Dict:
        """构建请求数据"""
        if seed != -1:
            np.random.seed(seed)
            random.seed(seed)
            style_variations = [
                "detailed, high quality",
                "masterpiece, ultra detailed", 
                "photorealistic, stunning",
                "artistic, beautiful composition",
                "vibrant colors, sharp focus"
            ]
            style = style_variations[seed % len(style_variations)]
            final_prompt = f"{prompt}, {style}"
        else:
            final_prompt = prompt
            
        parts = [{"text": final_prompt}]
        
        if input_images:
            for image_tensor in input_images:
                if image_tensor is not None:
                    base64_image = self.tensor_to_base64(image_tensor)
                    parts.append({
                        "inlineData": {
                            "mimeType": "image/png",
                            "data": base64_image
                        }
                    })
        
        generation_config = {
            "responseModalities": ["IMAGE", "TEXT"],
            "temperature": 0.8,
            "topP": top_p,
            "maxOutputTokens": 8192,
        }
        
        # 构建imageConfig - 必须同时配置才生效
        image_config = {}
        
        # 设置aspectRatio
        if aspect_ratio and aspect_ratio != "Auto":
            image_config["aspectRatio"] = aspect_ratio
        else:
            # 如果没有指定比例，使用默认的1:1
            image_config["aspectRatio"] = "1:1"
        
        # 设置imageSize
        if image_size and image_size != "Auto":
            image_config["imageSize"] = image_size
        
        # imageConfig必须要有，否则imageSize不生效
        generation_config["imageConfig"] = image_config
        
        if seed != -1:
            generation_config["seed"] = seed
        
        request_data = {
            "contents": [{
                "role": "user", 
                "parts": parts
            }],
            "generationConfig": generation_config
        }
        
        # 调试输出
        print(f"🔧 请求配置: aspectRatio={image_config.get('aspectRatio')}, imageSize={image_config.get('imageSize', 'Auto')}")
        
        return request_data

    def send_request(self, api_key: str, request_data: Dict, model_type: str, 
                    api_base_url: str, timeout: int = 180) -> Dict:
        """发送API请求"""
        endpoint = "generateContent"
        
        if "generativelanguage.googleapis.com" in api_base_url:
            url = f"{api_base_url.rstrip('/')}/v1beta/models/{model_type}:{endpoint}?key={api_key}"
            headers = {'Content-Type': 'application/json'}
        else:
            url = f"{api_base_url.rstrip('/')}/v1beta/models/{model_type}:{endpoint}"
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}',
            }
        
        headers['User-Agent'] = 'ComfyUI-Gemini-Node/2.2'
        
        session = requests.Session()
        session.headers.update(headers)
        
        try:
            response = session.post(url, json=request_data, timeout=timeout)
            
            if response.status_code != 200:
                raise Exception(f"API返回 {response.status_code}: {response.text[:200]}")
            
            return response.json()
            
        except requests.exceptions.Timeout:
            raise Exception(f"请求超时({timeout}秒)")
        except requests.exceptions.RequestException as e:
            raise Exception(f"网络错误: {str(e)}")
        finally:
            session.close()

    def extract_content(self, response_data: Dict) -> Tuple[List[str], str]:
        """提取响应中的图像和文本"""
        print(f"🔍 原始响应结构: {list(response_data.keys())}")
        base64_images = []
        text_content = ""
        
        candidates = response_data.get('candidates', [])
        if not candidates:
            raise ValueError("API响应中没有candidates字段")
        print(candidates)
        
        content = candidates[0].get('content', {})
        
        if content is None or content.get('parts') is None:
            return base64_images, text_content
        
        parts = content.get('parts', [])
        
        for part in parts:
            if 'text' in part:
                text_content += part['text']
            elif 'inlineData' in part and 'data' in part['inlineData']:
                base64_images.append(part['inlineData']['data'])
        
        if not base64_images and text_content:
            patterns = [
                r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)',
                r'!\[.*?\]\(data:image/[^;]+;base64,([A-Za-z0-9+/=]+)\)',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, text_content)
                if matches:
                    base64_images.extend(matches)

        return base64_images, text_content.strip()

    def generate_single_image(self, args):
        """生成单张图片(用于并发)"""
        i, current_seed, api_key, prompt, model_type, aspect_ratio, image_size, top_p, \
        input_images, api_base_url, timeout, stagger_delay = args
        
        if stagger_delay > 0:
            delay = i * stagger_delay
            if delay > 0:
                time.sleep(delay)
        
        thread_id = threading.current_thread().name
        task_start = time.time()
        
        print(f"[{thread_id}] 批次 {i+1} 开始请求...")
        
        try:
            request_data = self.create_request_data(prompt, current_seed, aspect_ratio, 
                                                   image_size, top_p, input_images)
            
            request_start = time.time()
            response_data = self.send_request(api_key, request_data, model_type, api_base_url, timeout)
            request_time = time.time() - request_start
            
            base64_images, text_content = self.extract_content(response_data)
            
            task_time = time.time() - task_start
            
            print(f"[{thread_id}] 批次 {i+1} ✅ 完成 - "
                  f"生成 {len(base64_images)} 张图片 - "
                  f"耗时 {task_time:.2f}s (API: {request_time:.2f}s)")
            
            return {
                'index': i,
                'success': True,
                'images': base64_images,
                'text': text_content,
                'seed': current_seed,
                'time': task_time,
                'request_time': request_time
            }
        except Exception as e:
            task_time = time.time() - task_start
            error_msg = str(e)[:200]
            print(f"[{thread_id}] 批次 {i+1} ❌ 失败 - 耗时 {task_time:.2f}s")
            print(f"  错误: {error_msg}")
            return {
                'index': i,
                'success': False,
                'error': error_msg,
                'seed': current_seed,
                'time': task_time
            }

    def generate_images(self, api_key, prompt, api_base_url, model_type, batch_size, aspect_ratio,
                       image_size, seed=-1, top_p=0.65, max_workers=4,
                       input_image_1=None, input_image_2=None, input_image_3=None,
                       input_image_4=None, input_image_5=None):
        
        # 验证API Key
        if not api_key or api_key.strip() == "":
            error_msg = "❌ 请输入有效的API Key"
            print(f"\n{error_msg}")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), error_msg)
        
        start_time = time.time()
        input_images = [img for img in [input_image_1, input_image_2, input_image_3,
                                       input_image_4, input_image_5] if img is not None]
        
        concurrent_mode = True
        stagger_delay = 0.0
        request_timeout = 1000
        continue_on_error = True
        
        if seed == -1:
            base_seed = random.randint(0, 102400)
        else:
            base_seed = seed
        
        all_b64_images = []
        all_texts = []
        
        print(f"\n{'='*60}")
        print(f"🎨 Gemini 图像生成 V2 (支持尺寸选择)")
        print(f"{'='*60}")
        print(f"API Key: {api_key[:10]}...{api_key[-10:] if len(api_key) > 20 else ''}")
        print(f"批次: {batch_size} 张")
        print(f"比例: {aspect_ratio}")
        print(f"尺寸: {image_size}")
        if seed != -1:
            print(f"种子: {seed}")
        if top_p != 0.65:
            print(f"Top-P: {top_p}")
        print(f"{'='*60}\n")
        
        if concurrent_mode and batch_size > 1:
            concurrent_start = time.time()
            
            tasks = []
            for i in range(batch_size):
                current_seed = base_seed + i if seed != -1 else -1
                tasks.append((i, current_seed, api_key, prompt, model_type, aspect_ratio, 
                            image_size, top_p, input_images, api_base_url, request_timeout, 
                            stagger_delay))
            
            results = []
            actual_workers = min(max_workers, batch_size)
            completed = 0
            
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                future_to_index = {executor.submit(self.generate_single_image, task): task[0] 
                                 for task in tasks}
                
                overall_timeout = request_timeout + 30
                
                try:
                    for future in as_completed(future_to_index, timeout=overall_timeout):
                        try:
                            result = future.result(timeout=5)
                            results.append(result)
                            completed += 1
                            
                            status = "✅" if result['success'] else "❌"
                            print(f"{status} [{completed}/{batch_size}] 批次 {result['index']+1} 完成\n")
                            
                        except TimeoutError:
                            index = future_to_index[future]
                            print(f"❌ 批次 {index+1} 获取结果超时\n")
                            results.append({
                                'index': index,
                                'success': False,
                                'error': '结果获取超时',
                                'time': 0
                            })
                        except Exception as e:
                            index = future_to_index[future]
                            print(f"❌ 批次 {index+1} 异常: {str(e)}\n")
                            results.append({
                                'index': index,
                                'success': False,
                                'error': str(e),
                                'time': 0
                            })
                            
                except TimeoutError:
                    print(f"\n⚠️ 整体超时!已完成 {completed}/{batch_size} 个任务")
                    for future in future_to_index:
                        future.cancel()
            
            concurrent_time = time.time() - concurrent_start
            
            results.sort(key=lambda x: x['index'])
            
            successful_results = [r for r in results if r['success']]
            failed_count = len(results) - len(successful_results)
            
            print(f"\n📊 并发统计:")
            print(f"  总耗时: {concurrent_time:.2f}s")
            print(f"  成功: {len(successful_results)}/{batch_size}")
            if failed_count > 0:
                print(f"  失败: {failed_count}/{batch_size}")
            
            if successful_results:
                total_request_time = sum(r.get('request_time', 0) for r in successful_results)
                max_request_time = max(r.get('request_time', 0) for r in successful_results)
                avg_request_time = total_request_time / len(successful_results)
                
                print(f"  API请求: 平均 {avg_request_time:.2f}s, 最长 {max_request_time:.2f}s")
                
                if total_request_time > 0:
                    speedup = total_request_time / concurrent_time
                    print(f"  加速比: {speedup:.2f}x")
            
            for result in results:
                if result['success']:
                    all_b64_images.extend(result['images'])
                    if result.get('text'):
                        all_texts.append(f"[批次 {result['index']+1}] {result['text']}")
                else:
                    error_msg = f"[批次 {result['index']+1}] ❌ {result.get('error', '未知错误')}"
                    all_texts.append(error_msg)
                    if not continue_on_error:
                        print(f"\n⚠️ 遇到错误且未开启容错,停止处理")
                        break
        else:
            for i in range(batch_size):
                current_seed = base_seed + i if seed != -1 else -1
                print(f"生成第 {i+1}/{batch_size} 张图片")
                if current_seed != -1:
                    print(f"种子: {current_seed}")
                
                try:
                    request_data = self.create_request_data(prompt, current_seed, aspect_ratio, 
                                                          image_size, top_p, input_images)
                    response_data = self.send_request(api_key, request_data, model_type, 
                                                     api_base_url, request_timeout)
                    base64_images, text_content = self.extract_content(response_data)
                    
                    if base64_images:
                        all_b64_images.extend(base64_images)
                        print(f"✓ 成功生成 {len(base64_images)} 张图片\n")
                    
                    if text_content:
                        all_texts.append(f"[批次 {i+1}] {text_content}")
                        
                except Exception as e:
                    error_msg = f"❌ 第 {i+1} 张失败: {str(e)}"
                    print(f"{error_msg}\n")
                    all_texts.append(error_msg)
                    if not continue_on_error:
                        break
        
        total_time = time.time() - start_time
        
        if not all_b64_images:
            error_text = f"⚠️ 未生成任何图像\n总耗时: {total_time:.2f}s\n\n" + "\n".join(all_texts)
            print(f"\n{error_text}")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), error_text)
        
        print(f"\n🖼️ 解码 {len(all_b64_images)} 张图片...")
        image_tensor = self.base64_to_tensor_parallel(all_b64_images)
        
        actual_count = len(all_b64_images)
        ratio_text = "自动" if aspect_ratio == "Auto" else aspect_ratio
        size_text = "自动" if image_size == "Auto" else image_size
        success_info = f"✅ 成功生成 {actual_count} 张图像(比例: {ratio_text}, 尺寸: {size_text})"
        
        avg_time = total_time / actual_count if actual_count > 0 else 0
        time_info = f"总耗时: {total_time:.2f}s,平均 {avg_time:.2f}s/张"
        
        if actual_count != batch_size:
            time_info += f" ⚠️ 请求{batch_size}张,实际生成{actual_count}张"
        
        combined_text = f"{success_info}\n{time_info}"
        if all_texts:
            combined_text += "\n\n" + "\n".join(all_texts)
        
        print(f"\n{'='*60}")
        print(f"✅ 完成!实际生成 {actual_count} 张图片(请求 {batch_size} 张)")
        print(f"{'='*60}\n")
        
        return (image_tensor, combined_text)

# 注册节点 - V2版本
NODE_CLASS_MAPPINGS = {
    "GeminiOpenAIProxyNodeV2": GeminiOpenAIProxyNodeV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiOpenAIProxyNodeV2": "Hello nano banana V2! 🍌⚡🎯"
}