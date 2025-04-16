import os
import pandas as pd
import re
from tqdm import tqdm
import jsonlines
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration, BlipProcessor, BlipForQuestionAnswering, Blip2Processor, Blip2ForConditionalGeneration, Qwen2VLForConditionalGeneration, BitsAndBytesConfig, AutoConfig, AutoModel, AutoTokenizer
from qwen_vl_utils import process_vision_info
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

DATASET_PATH = "/raid/gurukul/vlm4bio/easy_data/Easy" 
RESULTS_DIR = "/raid/gurukul/vlm4bio/Dataset/task1/results_exp2" 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="BLIP2-Flan-T5-XXL",
                    help="Choose a model from the list: BLIP2-Flan-T5-XXL, Qwen2.5-VL-7B, LLaVA v1.5-7B, Qwen2-VL-2B")
parser.add_argument("--dataset", "-d", type=str, default="bird",
                    help="Dataset options: bird, fish, butterfly")
parser.add_argument("--question_type", "-q", type=str, default="dense",
                    help="Choose between 'dense', 'contextual', 'cot', 'fct', 'nota', 'easy', 'medium', 'hard'") #easy, medium, hard
parser.add_argument("--num_queries", "-n", type=int, default=10,
                    help="Number of images to process")

args = parser.parse_args()


IMAGE_FOLDER = f"{DATASET_PATH}/{args.dataset}"

csv_file = f"{DATASET_PATH}/{args.dataset}.csv"
df = pd.read_csv(csv_file)

mcq_column = "MCQ_Prompt"
open_column = "OpenEnded_Prompt"
image_column = "fileNameAsDelivered"
answer_column = "scientificName"

denseCaption_col = "denseCaption"
densePrompt_col = "densePrompt"

cotCaptionA = "optionA"
cotCaptionB = "optionB"
cotCaptionC = "optionC"
cotCaptionD = "optionD"

fctOption = "fct"
nota = "nota"

#  Limit Queries for Kaggle
df = df.sample(n=args.num_queries, random_state=42)

#  Model-Specific Configurations
MODEL_DICT = {
    "BLIP2-Flan-T5-XXL": "Salesforce/blip2-flan-t5-xxl",
    "BLIP-VQA-Base": "Salesforce/blip-vqa-base",
    "Qwen2-VL-7B-Instruct":"Qwen/Qwen2-VL-7B-Instruct",
    "llava-1.5-7b-hf": "llava-hf/llava-1.5-7b-hf",
    "Qwen2-VL-2B-Instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "blip-vqa-capfilt-large":"Salesforce/blip-vqa-capfilt-large",

}


# Function to extract open-ended questions
def convert_to_open_ended(mcq_prompt):
    # Remove everything after "Options:"
    open_ended = re.sub(r"Options:.*", "", mcq_prompt).strip()
    return open_ended

#  Create a new column with open-ended questions
df["OpenEnded_Prompt"] = df["MCQ_Prompt"].apply(convert_to_open_ended)


#torch.cuda.set_device(2)
device = "cuda:7"
print(f"CUDA device set to {device}")

if args.model in MODEL_DICT:
    model_name = MODEL_DICT[args.model]
else:
    raise ValueError(f"Invalid model name: {args.model}. Please choose from the list or 'qwen'.")



model_name = MODEL_DICT[args.model]
print(model_name)

if "blip2" in model_name:
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map={"": 7}  #  Load everything on cuda:2
    ).to(device)

elif "blip-vqa" in model_name:
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name, device_map = "auto")
elif "blip-vqa-capfilt-large" in model_name:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large", device_map = "auto")

elif "llava-1.5-7b-hf" in model_name:
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from transformers import BitsAndBytesConfig
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", quantization_config = quantization_config, device_map = "auto")
elif "Qwen2-VL-2B-Instruct" in model_name:
    processor = AutoProcessor.from_pretrained(model_name)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, 
    quantization_config=quant_config, 
    device_map={"": 7}  #  Load all model components on cuda:2
).to("cuda:7")  
    print("model loaded successfully")
    print("model name: ", model_name)
elif "Qwen2-VL-7B" in model_name:
    processor = AutoProcessor.from_pretrained(model_name)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, 
    quantization_config=quant_config, 
    device_map={"": 7}  #  Load all model components on cuda:2
).to("cuda:7")  
#to update
elif "mplug-owl3" in model_name.lower():
    model_path = 'mPLUG/mPLUG-Owl3-1B-241014'
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half, trust_remote_code=True)
    model.eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = model.init_processor(tokenizer)
# else:
#     processor = AutoProcessor.from_pretrained(model_name)
#     model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)


#  Output File
out_file_name = f"{RESULTS_DIR}/classification_{args.model}_{args.dataset}_{args.question_type}.jsonl"
writer = jsonlines.open(out_file_name, mode="w")


#  Iterate Over Images
for _, row in tqdm(df.iterrows(), total=len(df)):
    
    image_path = os.path.join(IMAGE_FOLDER, row[image_column])
    print(image_path)

    #  Ensure Image Exists
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found, skipping.")
        continue
    #  Load Image
    image = Image.open(image_path).convert("RGB")

    #  Choose the correct question format
    # instruction = row[mcq_column] if args.question_type == "mcq" else row[open_column]
    # print(instruction)
    if args.question_type == "mcq":
        instruction = row[mcq_column]
    elif args.question_type == "easy":
        instruction = row[mcq_column]
    elif args.question_type == "medium":
        instruction = row[mcq_column]
    elif args.question_type == "hard":
        instruction = row[mcq_column]
    elif args.question_type == "open":
        instruction = row[open_column]
    elif args.question_type == "dense":
        caption = row[denseCaption_col].strip()
        question_part = row[mcq_column].strip()
        instruction = f"Dense Caption: {caption}\n\n Question: {question_part}"
    elif args.question_type == "contextual":
        caption = "Each biological species has a unique scientific name composed of two parts: the first for the genus and the second for the species within that genus."
        question_part = row[mcq_column].strip()
        instruction = f"question: {caption}\n\n{question_part}"
    elif args.question_type == "cot":
        
        option_a = row[cotCaptionA].strip()
        option_b = row[cotCaptionB].strip()
        option_c = row[cotCaptionC].strip()
        option_d = row[cotCaptionD].strip()
        question_part = row[mcq_column].strip()
        instruction = (
            f"Question: {question_part}\n\nPlease consider the following reasoning to formulate your answer: \n Reasoning: To identify the fish in the image, we need to compare its physical characteristics with the descriptions of the four given options \n"
            f"Option A: {option_a}\n"
            f"Option B: {option_b}\n"
            f"Option C: {option_c}\n"
            f"Option D: {option_d}\n\n"
        )
    elif args.question_type == "fct":
        
        fct_text = row[fctOption].strip()
        question_part = row[mcq_column].strip()
        instruction = f"Question:\n{question_part}\n\nFCT Option: {fct_text}"
    elif args.question_type == "nota":
        
        nota_text = row[nota].strip()
        
        instruction = f"Question:\n{nota_text}"
    else:
        raise ValueError(f"Invalid question_type: {args.question_type}")

    target_species = row[answer_column]

    #  Model-Specific Processing & Inference
    if "blip2" in model_name:
        img_url = image_path
        raw_image = Image.open(img_url)

        question = instruction
        inputs = processor(image, instruction, return_tensors="pt").to("cuda:7")  #  Move inputs to correct device


        out = model.generate(**inputs)
        response = processor.decode(out[0], skip_special_tokens=True)
        

    elif "blip-vqa" in model_name:
        url = image_path
        image = Image.open(url)

        question = instruction
        inputs = processor(image, question, return_tensors="pt").to("cuda:7", torch.float32)

        out = model.generate(**inputs, max_new_tokens=80)
        response = processor.decode(out[0], skip_special_tokens=True)
    elif "blip-vqa-capfilt-large" in model_name:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large", device_map = "auto")
        url = "/kaggle/input/bird-data/bird.jpeg"
        image = Image.open(image_path)

        question = instruction
        inputs = processor(image, question, return_tensors="pt").to("cuda:7")

        out = model.generate(**inputs, max_new_tokens=80)
        response = processor.decode(out[0], skip_special_tokens=True)


    elif "llava" in model_name:
    
        from PIL import Image
        import torch
        import requests
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": instruction},
            ],
        },
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, torch.float16)

        # Generate
        generate_ids = model.generate(**inputs, max_new_tokens=100)
        response = processor.batch_decode(generate_ids, skip_special_tokens=True)

        

    elif "Qwen2-VL-2B" in model_name:
    
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:7")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0]
        print(output_text)

    elif "Qwen2-VL-7B-Instruct" in model_name:
        messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        },
        ]

        image = Image.open(image_path)

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image])

        # Generate
        generate_ids = model.generate(torch.tensor(inputs.input_ids).to(device), max_new_tokens=80)
        processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


    elif "mplug-owl3" in model_name.lower():
        question = instruction
        url = image_path
        image = Image.open(url)
        messages = [
            {"role": "user", "content": "<|image|> \n"+question},
            {"role": "assistant", "content": ""}
        ]
        inputs = processor(messages, images=[image], videos=None)
        inputs.to('cuda:7')
        print(hasattr(model, "generate"))
        output = model(**inputs)
        # Get the predicted token IDs (argmax over logits)
        predicted_ids = torch.argmax(output.logits, dim=-1)
        response = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

    else:
        response = "could not find model"
  #  Save Result
    result = {
    "question": instruction,
    "target-class": target_species,  # Ground truth species name
    "output": response,  # Model prediction
    "image-path": image_path,
    "correct-answer": target_species,  # Ensures the correct answer is saved
    "answer-options": row[mcq_column] if args.question_type == "mcq" else None,  # Stores MCQ options
    "is-correct": 1 if response.strip().lower() == str(target_species).lower() else 0

    }   

    writer.write(result)

writer.close()
print(f" Results saved to {out_file_name}")


