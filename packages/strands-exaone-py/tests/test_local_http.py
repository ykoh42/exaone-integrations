"""Exercise the provider over a real loopback connection with synthetic responses."""

import asyncio
import json

import httpx
import pytest

from strands_exaone import ExaoneModel


@pytest.mark.parametrize("ending", ["complete", "disconnect", "cancel"])
async def test_stream_over_local_http(ending: str) -> None:
    requests = []
    disconnected = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            content_length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            requests.append(json.loads(await reader.readexactly(content_length)))
            event = {"choices": [{"index": 0, "delta": {"content": "hello"}}]}
            frame = f"data: {json.dumps(event)}\n\n".encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n")
            if ending == "disconnect":
                writer.write(b"Content-Length: 99999\r\n")
            writer.write(b"\r\n" + frame[:10])
            await writer.drain()
            writer.write(frame[10:])
            await writer.drain()
            if ending == "cancel":
                await asyncio.wait_for(reader.read(), timeout=5)
                disconnected.set()
            elif ending == "complete":
                stop = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                writer.write(f"data: {json.dumps(stop)}\n\ndata: [DONE]\n\n".encode())
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    observed = []
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            model = ExaoneModel(api_key="fixture", base_url=f"http://127.0.0.1:{port}/v1", http_client=client)
            stream = model.stream([{"role": "user", "content": [{"text": "hello"}]}])
            if ending == "disconnect":
                with pytest.raises(httpx.RemoteProtocolError):
                    async for event in stream:
                        observed.append(event)
            else:
                async for event in stream:
                    observed.append(event)
                    if ending == "cancel" and event.get("contentBlockDelta", {}).get("delta", {}).get("text"):
                        await stream.aclose()
                        break
                if ending == "cancel":
                    await asyncio.wait_for(disconnected.wait(), timeout=5)
        assert any(event.get("contentBlockDelta", {}).get("delta", {}).get("text") == "hello" for event in observed)
        assert len(requests) == 1
        assert requests[0]["stream"] is True
    finally:
        server.close()
        await server.wait_closed()
