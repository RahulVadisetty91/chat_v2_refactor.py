import os
import torch
import json
from PIL import Image
import base64
import io
from accelerate import load_checkpoint_and_dispatch, init_empty_weights, infer_auto_device_map
from transformers import AutoTokenizer, AutoModel
from omnilmm.utils import disable_torch_init
from omnilmm.model.omnilmm import OmniLMMForCausalLM
from omnilmm.model.utils import build_transform
from omnilmm.train.train_utils import omni_preprocess

# Constants for repeated string literals
IMAGE_DECODE_ERROR = "Image decode error"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def init_omni_lmm(model_path):
    torch.backends.cuda.matmul.allow_tf32 = True
    disable_torch_init()
    model_name = os.path.expanduser(model_path)
    print(f'Loading OmniLMM model and tokenizer from {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=2048)

    with init_empty_weights():
        model = OmniLMMForCausalLM.from_pretrained(model_name, tune_clip=True, torch_dtype=torch.bfloat16)
    model = load_checkpoint_and_dispatch(model, model_name, dtype=torch.bfloat16,
                                         device_map="auto", no_split_module_classes=['Eva', 'MistralDecoderLayer', 'ModuleList', 'Resampler'])

    image_processor = build_transform(is_train=False, input_size=model.model.config.image_size, std_mode='OPENAI_CLIP')

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    assert mm_use_im_start_end

    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_config = model.model.vision_config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    image_token_len = model.model.config.num_query

    return model, image_processor, image_token_len, tokenizer

def expand_question_into_multimodal(question_text, image_token_len, im_st_token, im_ed_token, im_patch_token):
    if '<image>' in question_text[0]['content']:
        question_text[0]['content'] = question_text[0]['content'].replace(
            '<image>', im_st_token + im_patch_token * image_token_len + im_ed_token)
    else:
        question_text[0]['content'] = im_st_token + im_patch_token * \
            image_token_len + im_ed_token + '\n' + question_text[0]['content']
    return question_text

def wrap_question_for_omni_lmm(question, image_token_len, tokenizer):
    question = expand_question_into_multimodal(
        question, image_token_len, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN)

    conversation = question
    data_dict = omni_preprocess(sources=[conversation], tokenizer=tokenizer, generation=True)

    data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
    return data_dict

class OmniLMM12B:
    def __init__(self, model_path) -> None:
        model, img_processor, image_token_len, tokenizer = init_omni_lmm(model_path)
        self.model = model
        self.image_token_len = image_token_len
        self.image_transform = img_processor
        self.tokenizer = tokenizer
        self.model.eval()

    def decode(self, image, input_ids):
        with torch.no_grad():
            output = self.model.generate_vllm(
                input_ids=input_ids.unsqueeze(0).cuda(),
                images=image.unsqueeze(0).half().cuda(),
                temperature=0.6,
                max_new_tokens=1024,
                do_sample=True,
                output_scores=True,
                return_dict_in_generate=True,
                repetition_penalty=1.1,
                top_k=30,
                top_p=0.9,
            )
            response = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True).strip()
            return response

    def chat(self, input):
        try:
            image = Image.open(io.BytesIO(base64.b64decode(input['image']))).convert('RGB')
        except Exception:
            return IMAGE_DECODE_ERROR

        msgs = json.loads(input['question'])
        input_ids = wrap_question_for_omni_lmm(msgs, self.image_token_len, self.tokenizer)['input_ids']
        input_ids = torch.tensor(input_ids)
        image = self.image_transform(image)
        out = self.decode(image, input_ids)
        return out

def img2base64(file_name):
    with open(file_name, 'rb') as f:
        encoded_string = base64.b64encode(f.read()).decode('utf-8')
        return encoded_string

class MiniCPM:
    def __init__(self, model_path) -> None:
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval().cuda()

    def chat(self, input):
        try:
            image = Image.open(io.BytesIO(base64.b64decode(input['image']))).convert('RGB')
        except Exception:
            return IMAGE_DECODE_ERROR

        msgs = json.loads(input['question'])
        answer, context, _ = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
        )
        return answer

class MiniCPMV2_5:
    def __init__(self, model_path) -> None:
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.eval().cuda()

    def chat(self, input):
        try:
            image = Image.open(io.BytesIO(base64.b64decode(input['image']))).convert('RGB')
        except Exception:
            return IMAGE_DECODE_ERROR

        msgs = json.loads(input['question'])
        answer = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            temperature=0.7
        )
        return answer

class MiniCPMV2_6:
    EMBED_TOKENS = "llm.model.embed_tokens"

    def __init__(self, model_path, multi_gpus=False) -> None:
        if multi_gpus:
            with init_empty_weights():
                model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16)

            device_map = infer_auto_device_map(model, max_memory={0: "10GB", 1: "10GB"},
                no_split_module_classes=['SiglipVisionTransformer', 'Qwen2DecoderLayer'])
            device_map[self.EMBED_TOKENS] = device_map[self.EMBED_TOKENS]
            device_map["vpm"] = device_map[self.EMBED_TOKENS]
            device_map["resampler"] = device_map[self.EMBED_TOKENS]
            device_id2 = device_map["llm.model.layers.26"]
            for i in range(8, 17):
                device_map[f"llm.model.layers.{i}"] = device_id2
            self.model = load_checkpoint_and_dispatch(model, model_path, dtype=torch.bfloat16, device_map=device_map)
            self.model.eval()
        else:
            self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                attn_implementation='sdpa', torch_dtype=torch.bfloat16).cuda()
            self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def chat(self, input):
        image = None
        if "image" in input and len(input["image"]) > 10:
            try:
                image = Image.open(io.BytesIO(base64.b64decode(input['image']))).convert('RGB')
            except Exception:
                return IMAGE_DECODE_ERROR

        msgs = json.loads(input["question"])

        for msg in msgs:
            contents = msg.pop('content')
            if isinstance(contents, str):
                contents = [contents]

            new_cnts = []
            for c in contents:
                if isinstance(c, dict):
                    if c['type'] == 'text':
                        c = c['pairs']
                    elif c['type'] == 'image':
                        c = Image.open(io.BytesIO(base64.b64decode(c["pairs"]))).convert('RGB')
                    else:
                        raise ValueError("Content type only supports text and image.")
                new_cnts.append(c)
            msg['content'] = new_cnts
        answer = self.model.chat(image=image, msgs=msgs, tokenizer=self.tokenizer)
        return answer

class MiniCPMVChat:
    def __init__(self, model_path, multi_gpus=False) -> None:
        if '12B' in model_path:
            self.model = OmniLMM12B(model_path)
        elif 'MiniCPM' in model_path:
            self.model = MiniCPM(model_path)
        elif 'MiniCPM-Llama3-V2' in model_path:
            self.model = MiniCPMV2_5(model_path)
        elif 'MiniCPM-Llama3-V2-6' in model_path:
            self.model = MiniCPMV2_6(model_path, multi_gpus=multi_gpus)
        else:
            raise ValueError("Unsupported model path")

    def chat(self, input):
        try:
            return self.model.chat(input)
        except Exception as e:
            return f"Error: {str(e)}"
