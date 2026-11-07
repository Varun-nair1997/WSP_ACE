# LofSRCR / Weight-Space Programming (WSP)

A modular test bed for investigating short-range compression behaviour in ACE interatomic potentials and experimenting with coefficient-space interventions guided by a local Ministral model.

## Repository layout

```text
lofsrcr-wsp/
├── lofsrcr_utils/                 # ACE test-bed and WSP utilities
│   ├── __init__.py
│   ├── compression.py              # BCC reference, ACE loading, compression scan
│   ├── diagnostics.py              # Monotonicity and artifact diagnostics
│   └── wsp.py                      # Baseline/candidate coefficient operations
├── ministral_runtime/              # Ollama + Ministral initialization/inference
│   ├── __init__.py
│   └── client.py
├── notebooks/
│   └── 01_lofsrcr_wsp_demo.ipynb   # End-to-end demonstration
├── requirements.txt
└── README.md
```

(The package directory is named `lofsrcr_utils` in the actual filesystem.)

## 1. Environment

Use the Python environment and PyACE build supported by the target machine/cluster. PyACE can have platform- and compiler-specific installation requirements; if `pip install pyace` is not available for your environment, install PyACE using your site's documented method and then install the remaining requirements.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. External model and permutation files

This repository does not redistribute ACE potentials or the verified coefficient permutation. Place the YAML files in an accessible directory, then configure:

```bash
export ACE_POTENTIAL_DIR=/path/to/potentials/Fe_ACE/yamls
export LOFSRCR_OUTPUT_DIR=/path/to/results/compression_scans
export WSP_PERMUTATION_PATH=/path/to/wsp/coefficient_permutation.npz
```

Expected potential filenames in the demo:

- `ACE_Fe-O.yaml` (model key `FeO_ACE`)
- `ACE_Fe-O-H.yaml` (model key `FeOH_ACE`)

The permutation NPZ must contain two arrays:

- `permutation`
- `inverse_permutation`

The code checks shape, bijectivity, and inverse consistency. It does not generate or infer the mapping.

## 3. Ministral / Ollama

Install Ollama on the compute node according to local policy and make the requested model available. The demo defaults to `ministral-3:8b`, matching the supplied notebook's runtime configuration.

```bash
export OLLAMA_MODEL=ministral-3:8b
# Optional: connect to an already-running Ollama service
export OLLAMA_URL=http://hostname:11434
# Optional: provide a non-standard Ollama binary location
export OLLAMA_BIN=/path/to/ollama
```

If `OLLAMA_URL` is unset, the runtime derives the URL from the current node's fully qualified hostname and `OLLAMA_PORT` (default `11434`).

In the notebook:

```python
from ministral_runtime import OllamaConfig, OllamaRuntime

runtime = OllamaRuntime(OllamaConfig(model="ministral-3:8b"))
runtime.wait_until_ready()
runtime.load_model()
answer = runtime.generate("WSP operational.", options={"temperature": 0})
print(answer["response"])
```

If the server is not already running, `runtime.start_server()` can start it. On HPC systems, ensure this is compatible with the scheduler, node allocation, and network policy. Do not expose an Ollama port beyond the intended compute environment.

## 4. Run the demo

Start Jupyter from the repository root:

```bash
jupyter lab
```

Open `notebooks/01_lofsrcr_wsp_demo.ipynb` and run the cells in order.

The demo performs:

1. Ollama readiness check, model warm-up, and smoke test.
2. BCC Fe compression scan with the configured ACE models.
3. Compression-response diagnostics and violation-region extraction.
4. Structured diagnostic prompt construction and Ministral inference.
5. Baseline coefficient extraction, verified permutation loading, candidate construction, and patch validation.

The default scan uses a BCC Fe conventional cell repeated 3×3×3, a lattice constant of 2.8665 Å, 80% maximum compression, and 300 scan intervals.

## 5. Importing the utilities

Example:

```python
from lofsrcr_utils import run_compression_test
from lofsrcr_utils import diagnose_compression_curve

scan = run_compression_test(
    models=["FeO_ACE"],
    model_paths={"FeO_ACE": "/path/to/ACE_Fe-O.yaml"},
    run_name="FeO_baseline",
)
results = scan["results"]
report = diagnose_compression_curve(results, "FeO_ACE")
print(report)
```

## Scientific cautions

- A decreasing energy increment during compression is a curve-level diagnostic, not a basis-level causal explanation.
- Large basis contributions do not automatically imply responsibility for non-monotonic behaviour. Evaluate contributions and their changes along the same trajectory.
- The patch validator's maximum step is a configurable guardrail, not a physically calibrated trust region.
- Always compare candidate and baseline on identical structures, distances, and numerical settings. Preserve the baseline model and store candidate artifacts separately.
- The permutation file must be generated and verified by the coefficient-ordering workflow appropriate to the PyACE version and model. Never assume coefficient order from index labels alone.

## Current scope

This repository modularizes the workflow in `WSP2.txt`. It does not yet implement basis-resolved energy decomposition or coefficient sensitivity analysis; those are natural next additions to the LofSRCR test bed.
