"""reclaim_serve: the serving half of reclaim — stand up a vLLM endpoint.

reclaim splits into two halves that couple only through a published URL:

* the **client half**: the provider-agnostic agents (classifier, auditor,
  reproduction) that run no model themselves and only POST chat completions to a
  base URL, exactly like Codex / Claude Code / opencode; and
* the **serving half** (this package): it boots a vLLM server on a GPU node
  (e.g. 4 GPUs, tensor-parallel 4), binds a routable address, and publishes the
  URL so any other node on the cluster can attach.

Because the only contract is ``vllm_endpoint.json``, swapping the model is a URL
change. This package must therefore not import the client packages.

Run it::

    PYTHONPATH=src python -m reclaim_serve --model /path/to/MiniMax-M2.7 --port 8000

It prints the base URL and writes ``vllm_endpoint.json``; a consumer points
``--vllm-server-url`` / ``$RECLAIM_SERVER_URL`` / ``$RECLAIM_ENDPOINT_FILE`` at it.
"""

__version__ = "0.1.0"
