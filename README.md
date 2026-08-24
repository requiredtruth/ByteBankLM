# ByteBankLM

ByteBankLM makes deterministic admission decisions for several local-language-model jobs sharing one RAM budget. It works in bytes, separates weights, KV cache, runtime overhead, and operator reserve, then admits jobs by explicit priority.

```bash
python -m bytebanklm examples/plan.json
python -m bytebanklm examples/plan.json --fail-on-reject
```

KV bytes are calculated as `2 * layers * kv_heads * head_dim * context * element_bytes` for keys and values. Every other backend allocation must be supplied as `runtime_overhead_bytes`; ByteBankLM deliberately does not invent a percentage. It is a planner, not a runtime measurement or performance promise.

The JSON result is stable for scripts and records why each job was admitted or how many bytes it lacked. `--fail-on-reject` makes the plan usable as a deployment gate.

## Test

`python -m unittest discover -s tests -v`

## Fund more development

Donations increase RequiredTruth development production. See [SUPPORT.md](SUPPORT.md); confirmed donors may claim a transaction hash in an issue and request a specific direction.

Apache-2.0 licensed.


## Install and run

```sh
chmod +x install.sh run.sh
./install.sh
./run.sh --help
```
