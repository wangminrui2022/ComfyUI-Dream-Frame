---
license: mit
base_model:
- microsoft/Florence-2-large
datasets:
- Ejafa/ye-pop
tags:
- art
pipeline_tag: image-to-text
language:
- en
library_name: transformers
---

# microsoft/Florence-2-large tuned on Ejafa/ye-pop captioned with CogVLM2

This repository contains a fine-tuned version of the `microsoft/Florence-2-large` model. The model has been tuned on a 38,000 image subset of the `Ejafa/ye-pop` dataset, with captions generated using `THUDM/cogvlm2-llama3-chat-19B`.

## Training Details

- **Vision Encoder**: The vision encoder was frozen during training.
- **Batch Size**: 32
- **Gradient Accumulation Steps**: 8
- **Learning Rate**: 4.2667e-5
- **Optimizer**: AdamW
- **Scheduler**: linear
- **Epochs**: 7

## Dataset

The fine-tuning process utilized a 38,000 image subset from the `Ejafa/ye-pop` dataset. This dataset contains a wide array of images with varying subjects, providing a robust training ground for improving the model's captioning abilities.

## Captioning

The captions were generated using `THUDM/cogvlm2-llama3-chat-19B`.

## Usage

To use this model, you can load it directly from the Hugging Face Model Hub:

```python
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("thwri/CogFlorence-2-Large-Freeze", trust_remote_code=True).to(device).eval()
processor = AutoProcessor.from_pretrained("thwri/CogFlorence-2-Large-Freeze", trust_remote_code=True)

# Function to run the model on an example
def run_example(task_prompt, image):
    prompt = task_prompt

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
        do_sample=True
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer

from PIL import Image
import requests
import copy

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)
result = run_example("<MORE_DETAILED_CAPTION>" , image)
print(result)

# {'<MORE_DETAILED_CAPTION>': 'a turquoise volkswagen beetle parked on a cobblestone street in front of a yellow wall with two wooden doors. the car's body is painted in a vibrant shade of teal, with a glossy finish that reflects the sunlight, and the wheels are polished with a silver hubcap. the building behind the car has a weathered, aged appearance, with visible cracks and peeling paint. the sky above is clear and blue, suggesting a sunny day.'}
```