"""Small localhost Qwen-VL inference server; run in a separate Transformers environment."""
import argparse
import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen3.5-9B')
    p.add_argument('--checkpoint', help='Optional local checkpoint; --model remains the served model ID')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--device', default='cuda:0', help='Logical CUDA device or auto to shard over CUDA_VISIBLE_DEVICES')
    p.add_argument('--gpu-memory', default='20GiB', help='Per-GPU weight budget when --device auto; reserve space for activations')
    args = p.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    checkpoint = args.checkpoint or args.model
    processor = AutoProcessor.from_pretrained(checkpoint)
    from cross_episode_sim.evaluation.video_decode import install_streaming_decoder
    install_streaming_decoder(processor)
    placement = {}
    if args.device == 'auto':
        placement['max_memory'] = {i: args.gpu_memory for i in range(torch.cuda.device_count())}
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="balanced" if args.device == "auto" else args.device, attn_implementation='sdpa', **placement).eval()
    guard = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, payload):
            encoded = json.dumps(payload).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded))); self.end_headers(); self.wfile.write(encoded)

        def do_GET(self):
            if self.path in ('/health', '/v1/models'):
                self.reply(200, dict(data=[dict(id=args.model)]))
            else:
                self.reply(404, dict(error='Unknown route'))

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.reply(404, dict(error='Unknown route')); return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= 50_000_000:
                    raise ValueError('Request must contain at most 50 MB')
                request = json.loads(self.rfile.read(length))
                if request.get('model') != args.model:
                    raise ValueError('Requested model is not loaded')
                messages = request['messages']
                for message in messages:
                    if isinstance(message['content'], list):
                        for i, block in enumerate(message['content']):
                            if block['type'] == 'video_url':
                                parsed = urlparse(block['video_url']['url'])
                                if parsed.scheme != 'file' or parsed.netloc not in ('', 'localhost'):
                                    raise ValueError('Local helper requires shared filesystem file:// videos')
                                path = Path(unquote(parsed.path)).resolve(strict=True)
                                message['content'][i] = dict(type='video', path=str(path))
                            elif block['type'] == 'image_url':
                                url = block['image_url']['url']
                                if not url.startswith('data:image/'):
                                    raise ValueError('Only inline images are supported')
                                image = Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB')
                                message['content'][i] = dict(type='image', image=image)
                video_options = request.get('mm_processor_kwargs', {})
                if set(video_options) - {'fps', 'do_sample_frames'}:
                    raise ValueError('Unsupported video processor override')
                queued = time.monotonic()
                print("QWEN_STAGE waiting_for_inference_lock", flush=True)
                with guard, torch.inference_mode():
                    started = time.monotonic()
                    print(f"QWEN_STAGE preprocessing queue_seconds={started-queued:.1f}", flush=True)
                    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                          return_dict=True, return_tensors='pt', enable_thinking=False, **video_options).to(model.device)
                    print(f'QWEN_INPUT tokens={inputs.input_ids.shape[1]} video_grid={inputs.get("video_grid_thw")}', flush=True)
                    prepared = time.monotonic()
                    print(f"QWEN_STAGE generation preprocessing_seconds={prepared-started:.1f}", flush=True)
                    output = model.generate(**inputs, max_new_tokens=min(int(request.get('max_tokens', 256)), 1024),
                                            do_sample=False)
                    print(f"QWEN_STAGE complete generation_seconds={time.monotonic()-prepared:.1f}", flush=True)
                    generated = output[0, inputs['input_ids'].shape[1]:]
                    text = processor.decode(generated, skip_special_tokens=True)
                    usage = dict(prompt_tokens=int(inputs['input_ids'].shape[1]), completion_tokens=len(generated))
                self.reply(200, dict(model=args.model, choices=[dict(message=dict(role='assistant', content=text))], usage=usage))
            except (BrokenPipeError, ConnectionResetError):
                print("QWEN_CLIENT_DISCONNECTED", flush=True)
            except Exception as exc:
                print(f"QWEN_ERROR {type(exc).__name__}: {exc}", flush=True)
                self.reply(500, dict(error=f'{type(exc).__name__}: {exc}'))

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'QWEN_READY http://{args.host}:{args.port}/v1 model={args.model}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
