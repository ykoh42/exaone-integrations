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
