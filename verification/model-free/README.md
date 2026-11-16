# Reproducing isolated server-code checks

These optional, CPU-only probes exercise selected upstream source code with **synthetic inputs and test doubles**. They do not start vLLM or SGLang, load model weights, call an API, or verify real model inference. They are deliberately separate from the provider test suites and are not required to install any package.

The probes expect the exact source revisions below. Two probes execute selected functions extracted from those checkouts; only run them against source you trust. An assertion failure on another revision may mean the server changed; investigate the new behavior rather than treating the old observation as a requirement.

Using Python 3.11, create an isolated environment and install the probe dependencies:

```bash
python3.11 -m venv /tmp/exaone-parser-check
/tmp/exaone-parser-check/bin/python -m pip install -r verification/model-free/requirements.txt
```

From a separate working directory, obtain the pinned source checkouts:

```bash
git clone https://github.com/lkm2835/sglang.git /tmp/exaone-sglang-k2
git -C /tmp/exaone-sglang-k2 checkout 918653d22b2f1ae853b81fc0f5539b7c9311487e
git clone https://github.com/lkm2835/sglang.git /tmp/exaone-sglang-45
git -C /tmp/exaone-sglang-45 checkout 6757c9f904cdb8ae9028a394a2108d079b9e088c
git clone https://github.com/vllm-project/vllm.git /tmp/exaone-vllm
git -C /tmp/exaone-vllm checkout 4bdc8a788d2e2ce9165d552b3d4d8b72604626bf
```

From this repository's root, run:

```bash
SGLANG_PATH=/tmp/exaone-sglang-k2 /tmp/exaone-parser-check/bin/python verification/model-free/sglang-qwen3-null.py
SGLANG_PATH=/tmp/exaone-sglang-45 /tmp/exaone-parser-check/bin/python verification/model-free/sglang-hermes-finish.py
VLLM_PATH=/tmp/exaone-vllm /tmp/exaone-parser-check/bin/python verification/model-free/vllm-hermes.py
```

The first probe checks the observed `qwen3_coder` conversion of the text `null` to JSON null for a string parameter, both in one response and in six fragment sizes. The second checks the observed `hermes` parser and extracted finish check when a complete synthetic tool call arrives in one chunk. The third checks the vLLM Hermes parser on two synthetic calls, in one response and seven fragment sizes. The vLLM wire types and unrelated dependencies are test doubles; the SGLang finish probe extracts one server function. None reproduces a full server's scheduling, deployment, or model output. See the [compatibility matrix](../../COMPATIBILITY.md) for the wider evidence and limitations.

## Parser-to-provider round trips

The following check generates synthetic reasoning and tool calls, runs the model-card parsers, and feeds their results to all four providers. Both streaming and non-streaming cases check the tool arguments and the next request's reasoning, answer and tool result. Strands executes a synthetic tool in its SDK; LangChain checks the returned call and supplies a synthetic tool result. HTTP responses are in-memory fixtures; no service, GPU, weights, credentials, or real prompts are used.

K-EXAONE 2.0 uses the model-card vLLM and SGLang forks with Qwen3 reasoning and XML tools. EXAONE 4.5 uses official vLLM 0.27.0 and the model-card SGLang fork with Qwen3 reasoning and Hermes tools. The vLLM 4.5 check runs the reasoning and tool parsers separately and constructs the wire response; it does not execute their server integration. The vLLM parsers use dependency and wire-type stubs. SGLang runs its detectors in isolation; the wire envelope and call ID are synthetic, not server output. The 4.5 SGLang single-final-chunk limitation described above is **not** reproduced by these successful round trips; `sglang-hermes-finish.py` tests that case separately. None of these checks establishes actual model quality, endpoint behavior, scheduler chunking, or deployment support.

Install the repository dependencies described in [CONTRIBUTING.md](../../CONTRIBUTING.md), including both Python environments and the root npm workspace. The replay commands below assume the Strands Python environment from that guide is active and LangChain's environment is at `packages/langchain-exaone-py/.venv`. Build the TypeScript workspace with `npm run build`. Use the parser environment and the three checkouts above, then obtain the vLLM model-card fork and the two pinned tokenizer files (not model weights):

```bash
git clone https://github.com/lkm2835/vllm.git /tmp/exaone-vllm-k2
git -C /tmp/exaone-vllm-k2 checkout 33fa995c084e1bf652b4a20130ed0f04d99c9381
mkdir -p /tmp/exaone-roundtrip-input /tmp/exaone-roundtrip-fixtures
curl -fL 'https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B/resolve/fec0c8dd25fdd20ffd09869d5af5277966d5523d/tokenizer.json' -o /tmp/exaone-roundtrip-input/k2-tokenizer.json
curl -fL 'https://huggingface.co/LGAI-EXAONE/EXAONE-4.5-33B/resolve/570aa4b15a4f45ba1133072b45f50198f6e3b4fd/tokenizer.json' -o /tmp/exaone-roundtrip-input/45-tokenizer.json
```

Generate all four fixtures from the repository root. The generator verifies the source commit and tokenizer checksum. Use an empty fixture directory; the replay checks require exactly these four JSON files.

```bash
/tmp/exaone-parser-check/bin/python verification/model-free/roundtrip/fixtures.py vllm k2 /tmp/exaone-vllm-k2 /tmp/exaone-roundtrip-input/k2-tokenizer.json /tmp/exaone-roundtrip-fixtures
/tmp/exaone-parser-check/bin/python verification/model-free/roundtrip/fixtures.py sglang k2 /tmp/exaone-sglang-k2 /tmp/exaone-roundtrip-input/k2-tokenizer.json /tmp/exaone-roundtrip-fixtures
/tmp/exaone-parser-check/bin/python verification/model-free/roundtrip/fixtures.py vllm 45 /tmp/exaone-vllm /tmp/exaone-roundtrip-input/45-tokenizer.json /tmp/exaone-roundtrip-fixtures
/tmp/exaone-parser-check/bin/python verification/model-free/roundtrip/fixtures.py sglang 45 /tmp/exaone-sglang-45 /tmp/exaone-roundtrip-input/45-tokenizer.json /tmp/exaone-roundtrip-fixtures
```

Replay them using the two independently installed Python providers and the built TypeScript workspaces:

```bash
python verification/model-free/roundtrip/replay-strands-py.py /tmp/exaone-roundtrip-fixtures
packages/langchain-exaone-py/.venv/bin/python verification/model-free/roundtrip/replay-langchain-py.py /tmp/exaone-roundtrip-fixtures
node verification/model-free/roundtrip/replay-strands-ts.mjs /tmp/exaone-roundtrip-fixtures
node verification/model-free/roundtrip/replay-langchain-ts.mjs /tmp/exaone-roundtrip-fixtures
```

The source revision check reads Git `HEAD`; use clean checkouts to ensure local edits do not change the parser. The test inputs and output files are synthetic. Never substitute real model responses or credentials in these fixtures.
