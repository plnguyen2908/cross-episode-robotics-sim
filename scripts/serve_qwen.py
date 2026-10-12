"""Small localhost Qwen-VL inference server; run in a separate Transformers environment."""
import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='Qwen/Qwen3.5-9B')
    p.add_argument('--checkpoint', help='Optional local checkpoint; --model remains the served model ID')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--device', default='cuda:0', help='Logical CUDA device; set CUDA_VISIBLE_DEVICES to choose physical GPU')
    args = p.parse_args()
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    checkpoint = args.checkpoint or args.model
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map=args.device, attn_implementation='sdpa').eval()
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
                            if block['type'] == 'image_url':
                                url = block['image_url']['url']
                                if not url.startswith('data:image/'):
                                    raise ValueError('Only inline images are supported')
                                image = Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB')
                                message['content'][i] = dict(type='image', image=image)
                with guard, torch.inference_mode():
                    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                          return_dict=True, return_tensors='pt', enable_thinking=False).to(model.device)
                    output = model.generate(**inputs, max_new_tokens=min(int(request.get('max_tokens', 256)), 1024),
                                            do_sample=False)
                    generated = output[0, inputs['input_ids'].shape[1]:]
                    text = processor.decode(generated, skip_special_tokens=True)
                    usage = dict(prompt_tokens=int(inputs['input_ids'].shape[1]), completion_tokens=len(generated))
                self.reply(200, dict(model=args.model, choices=[dict(message=dict(role='assistant', content=text))], usage=usage))
            except Exception as exc:
                self.reply(500, dict(error=f'{type(exc).__name__}: {exc}'))

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'QWEN_READY http://{args.host}:{args.port}/v1 model={args.model}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
