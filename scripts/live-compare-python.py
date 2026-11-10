"""Compare reasoning history with Strands' OpenAIModel; opt-in live API test."""
from __future__ import annotations

import asyncio
import os
import runpy
import ssl

import httpx
import truststore
from strands import Agent
from openai import AsyncOpenAI
from strands.models.openai import OpenAIModel

Probe = runpy.run_path('scripts/live-smoke-python.py')['Probe']

async def main() -> None:
    key = os.environ.get('EXAONE_API_KEY')
    if not key:
        raise SystemExit('Set EXAONE_API_KEY to run this opt-in live test')
    probe = Probe()
    context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    async with httpx.AsyncClient(verify=context, event_hooks={'request': [probe.on_request]}, timeout=120) as client:
        model = OpenAIModel(
            model_id='k-exaone_v2.5',
            client=AsyncOpenAI(api_key=key, base_url='https://api.exaone.ai/v1', max_retries=0, http_client=client),
            params={'max_tokens': 512, 'extra_body': {'reasoning_effort': 'low', 'chat_template_kwargs': {'preserve_thinking': True}}},
        )
        agent = Agent(model=model, callback_handler=None)
        await agent.invoke_async('Remember the code word amber. Reply with one short sentence.')
        reasoning_received = any('reasoningContent' in block for message in agent.messages for block in message.get('content', []))
        await agent.invoke_async('Repeat the code word I asked you to remember.')
        probe.report('openai_baseline_reasoning_history', requests=len(probe.requests), reasoning_received=reasoning_received,
                     forwarded=probe.requests[-1]['reasoning_history'], preserve_thinking=probe.requests[-1]['preserve_thinking'])

if __name__ == '__main__':
    asyncio.run(main())
