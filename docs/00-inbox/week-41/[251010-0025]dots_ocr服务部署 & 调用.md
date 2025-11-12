## 项目目录
> https://github.com/rednote-hilab/dots.ocr/tree/master

## 步骤
1. conda环境 `vllm2`，安装`vllm 0.9.1`
2. deploy
``` bash

export hf_model_path=./huing_ocr
export PYTHONPATH=$(dirname "$hf_model_path"):$PYTHONPATH
sed -i '/^from vllm\.entrypoints\.cli\.main import main$/a\
from huing_ocr import modeling_ocr_vllm' `which vllm`

export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup vllm serve ${hf_model_path} --tensor-parallel-size 1 --gpu-memory-utilization 0.9  --chat-template-content-format string --served-model-name model --trust-remote-code --port 8080 > vllm.log 2>&1 &

vllm serve ${hf_model_path} --tensor-parallel-size 1 --gpu-memory-utilization 0.9  --chat-template-content-format string --served-model-name model --trust-remote-code --port 8080



```

# 接口调用格式
**接口地址**
`http://hz.mindflow.com.cn:28080/v1`

## 请求方法
POST

## 请求体（JSON）
```json
{
  "model": "{model_name}",  // 默认为"model"
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/{format};base64,{base64_str}"  // 图片Base64编码
          }
        },
        {
          "type": "text",
          "text": "<|img|><|imgpad|><|endofimg|>{prompt}"  // prompt为布局分析指令
        }
      ]
    }
  ],
  "temperature": 0.1,
  "top_p": 0.9,
  "max_completion_tokens": 32768
}
```

## 响应体（JSON）
```json
{
  "choices": [
    {
      "message": {
        "content": "{布局分析结果JSON字符串}"  // 包含bbox、category、text等字段
      }
    }
  ]
}
```

## 示例脚本：
``` python 
import argparse
import requests
from openai import OpenAI
import os
from io import BytesIO
import base64


from PIL import Image

# from dots_ocr.model.inference import inference_with_vllm

def PILimage_to_base64(image, format='PNG'):
    buffered = BytesIO()
    image.save(buffered, format=format)
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{base64_str}"

def inference_with_vllm(
        image,
        prompt, 
        ip="localhost",
        port=8000,
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=4096,
        model_name='model',
        ):
    
    addr = f"http://{ip}:{port}/v1"
    print(f"[debug]Addr: {addr}")
    client = OpenAI(api_key="{}".format(os.environ.get("API_KEY", "0")), base_url=addr)
    messages = []
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url":  PILimage_to_base64(image)},
                },
                {"type": "text", "text": f"<|img|><|imgpad|><|endofimg|>{prompt}"}  # if no "<|img|><|imgpad|><|endofimg|>" here,vllm v1 will add "\n" here
            ],
        }
    )
    try:
        response = client.chat.completions.create(
            messages=messages, 
            model=model_name, 
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            top_p=top_p)
        response = response.choices[0].message.content
        return response
    except requests.exceptions.RequestException as e:
        print(f"request error: {e}")
        return None



PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.\n\n1. Bbox format: [x1, y1, x2, y2]\n\n2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n\n3. Text Extraction & Formatting Rules:\n    - Picture: For the 'Picture' category, the text field should be omitted.\n    - Formula: Format its text as LaTeX.\n    - Table: Format its text as HTML.\n    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n4. Constraints:\n    - The output text must be the original text from the image, with no translation.\n    - All layout elements must be sorted according to human reading order.\n\n5. Final Output: The entire output must be a single JSON object.\n"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal vLLM OCR demo client")
    parser.add_argument("--ip", type=str, default="hz.mindflow.com.cn", help="vLLM server host")
    parser.add_argument("--port", type=int, default=8080, help="vLLM server port")
    parser.add_argument("--model-name", type=str, default="model", help="Remote model name")
    parser.add_argument(
        "--image",
        type=str,
        default="demo_image1.jpg",
        help="Path to the image used for OCR",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image)

    response = inference_with_vllm(
        image=image,
        prompt=PROMPT,
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
    )

    print(response)


if __name__ == "__main__":
    main()

```