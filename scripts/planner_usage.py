#!/usr/bin/env python3
"""Total the planner's API token usage and cost of an evaluation batch.

    python3 scripts/planner_usage.py <batch folder> [--input-price 2 --cached-price 0.10 --output-price 10]

Reads every <episode>/episode/qwen_plan.json (plan requests, repairs and
replans; GPT planners record each request's usage there). Prices are US$ per
million tokens; the defaults are GPT-6.1 Sol's published API rates at the time
of writing (input $2, cached input $0.10, output incl. reasoning $10). Check the
current rates at https://developers.openai.com/api/docs/pricing.
"""
import argparse
import json
from pathlib import Path


def requests_of(record):
    for attempt in record.get('attempts') or []:yield attempt
    for replan in record.get('replans') or []:
        for attempt in replan.get('attempts') or []:yield attempt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('batch', type=Path)
    p.add_argument('--input-price', type=float, default=2.0)
    p.add_argument('--cached-price', type=float, default=0.10)
    p.add_argument('--output-price', type=float, default=10.0)
    a = p.parse_args()
    total = dict(episodes=0, requests=0, input_tokens=0, cached_input_tokens=0, output_tokens=0, reasoning_tokens=0)
    for path in sorted(a.batch.glob('*/episode/qwen_plan.json')):
        if '_failedstart_' in str(path):continue
        record = json.loads(path.read_text());total['episodes'] += 1
        for attempt in requests_of(record):
            usage = attempt.get('usage') or {}
            total['requests'] += 1
            for key in ('input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_tokens'):
                total[key] += int(usage.get(key) or 0)
    uncached = total['input_tokens'] - total['cached_input_tokens']
    cost = (uncached * a.input_price + total['cached_input_tokens'] * a.cached_price
            + total['output_tokens'] * a.output_price) / 1e6
    total['cost_usd'] = round(cost, 2)
    total['cost_per_episode_usd'] = round(cost / max(1, total['episodes']), 3)
    total['prices_usd_per_million'] = dict(input=a.input_price, cached_input=a.cached_price, output=a.output_price)
    (a.batch / 'planner_usage.json').write_text(json.dumps(total, indent=2))
    print(json.dumps(total, indent=2))


if __name__ == '__main__':
    main()
