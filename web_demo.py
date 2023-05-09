import os
import sys

import gradio as gr
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)

from ui import chat, knowledge

folder_path = os.path.abspath('chatglm_6b')
sys.path.append(folder_path)


def ui(model, tokenizer):
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">RolePlayGLM</h1>""")
        with gr.Tab('Chat'):
            chat.ui(model, tokenizer)
        with gr.Tab('Knowledge'):
            knowledge.ui()
    demo.queue().launch(share=False, inbrowser=True, server_name='0.0.0.0')


def main():
    from chatglm_6b.ptuning.arguments import ModelArguments
    parser = HfArgumentParser((ModelArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        model_args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True)

    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    if model_args.ptuning_checkpoint is not None:
        print(f"Loading prefix_encoder weight from {model_args.ptuning_checkpoint}")
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)

    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model = model.half().cuda()
        model.transformer.prefix_encoder.float().cuda()

    model = model.eval()
    ui(model, tokenizer)


if __name__ == "__main__":
    main()
