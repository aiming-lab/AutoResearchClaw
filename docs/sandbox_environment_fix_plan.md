# AutoResearchClaw — Docker Sandbox Environment Enhancement Plan

> Created: 2026-03-15
> Status: **DONE** — All 10 issues fixed, 1128/1128 tests passing
> Priority: **CRITICAL** — Without these fixes, experiments fall back to synthetic data, producing meaningless papers

---

## 0. Problem Statement

When a user requests experiments on real datasets (e.g., ImageNet, OGB, HuggingFace benchmarks),
the pipeline fails to use them because:

1. The LLM is **not told** which packages are actually available in the Docker image
2. The Docker sandbox **cannot install packages at runtime** (default `network_policy: "none"`)
3. Phase 1 pip install is **broken** — packages install in Container A, but experiment runs in Container B (packages lost)
4. Only **4 small datasets** are pre-cached (CIFAR-10, FashionMNIST); prompt incorrectly claims CIFAR-100 and MNIST are cached too
5. **No dataset download mechanism** exists — no setup phase for downloading data before experiment execution
6. The Dockerfile is **missing key ML packages** (timm, einops, torchmetrics, ogb, etc.)

**Result:** The LLM generates `torch.randn()` "ImageNet-like" synthetic data as a fallback, making all experiment results meaningless.

---

## 1. Reference Solutions Analysis

### 1.1 AI-Scientist (SakanaAI)
- **Approach:** "Fat image" — ALL dependencies and datasets baked into Docker image at build time
- **Dataset handling:** Pre-download scripts run during `docker build` (enwik8, shakespeare, text8)
- **Runtime pip install:** None — not supported
- **Network:** No isolation (user's responsibility)
- **Lesson:** Pre-caching is the most reliable strategy for reproducibility

### 1.2 AutoResearch (Karpathy)
- **Approach:** End-to-end automation in local environment
- **Dataset handling:** Direct downloads via standard APIs
- **Lesson:** Simplicity — don't over-engineer isolation if it breaks functionality

### 1.3 OpenHands (formerly OpenDevin)
- **Approach:** Most sophisticated sandbox architecture
- **Key feature:** `runtime_extra_deps` config for pre-declaring packages
- **Agent autonomy:** Agent can run `pip install` via `CmdRunAction` inside the container
- **Three-tag Docker image caching system** for build optimization
- **Lesson:** Allow agent (LLM) to declare and install its own dependencies

### 1.4 MLCommons Training Benchmarks
- **Approach:** Host-download, container-mount pattern
- **Three phases:** Download on host → Build Docker image → Mount data volumes
- **Lesson:** Large datasets should NEVER be inside Docker images — always volume-mount

### 1.5 Docker Desktop Sandboxes
- **Network policies:** HTTP/HTTPS proxy allowlists per host
- **Example:** Allow `*.pypi.org`, `*.huggingface.co`, `download.pytorch.org`, block everything else
- **Lesson:** Fine-grained network control is better than all-or-nothing

---

## 2. Issues Identified

### E1: `pkg_hint` doesn't list most installed packages [CRITICAL]
**File:** `researchclaw/pipeline/executor.py:2062-2073`
**Current:**
```python
pkg_extras = ", torchdiffeq, gymnasium, networkx, and pip-installable packages"
# Resulting prompt: "AVAILABLE PACKAGES: Python stdlib, numpy, torch, sklearn, scipy, pandas, torchdiffeq, gymnasium, networkx"
```
**Missing from list:** torchvision, torchaudio, matplotlib, seaborn, PyYAML, tqdm
**Impact:** LLM thinks torchvision isn't available → avoids it → generates synthetic data instead of using CIFAR-10

### E2: Phase 1/Phase 2 container isolation BUG [CRITICAL]
**File:** `researchclaw/experiment/docker_sandbox.py:169-181, 317-354`
**Bug:** Phase 1 runs `docker run --rm` (installs packages, then container is removed). Phase 2 runs a NEW `docker run --rm` from the same base image. Packages installed in Phase 1 are **completely lost** because the container was deleted.
**Impact:** `auto_install_deps` and `pip_pre_install` features are entirely non-functional. Any package not in the base Docker image is unavailable during experiment execution.

### E3: Default `network_policy` is `"none"` [HIGH]
**File:** `researchclaw/config.py:163`
**Current:** `network_policy: str = "none"`
**Impact:** Even with `auto_install_deps: True`, Phase 1 never executes because it requires `network_policy == "pip_only"`. No runtime installation or download is possible by default.

### E4: No dataset download phase [HIGH]
**File:** `researchclaw/experiment/docker_sandbox.py` (no implementation exists)
**Missing:** There is no mechanism for downloading datasets before experiment execution. The only datasets available are the 4 pre-cached in the Docker image.
**Impact:** Experiments requiring any dataset beyond CIFAR-10/100, MNIST, FashionMNIST cannot use real data.

### E5: Pre-cached dataset list inconsistent [MEDIUM]
**File:** `researchclaw/docker/Dockerfile:27-30` vs `researchclaw/prompts.py:328-332`
**Bug:** Dockerfile only pre-caches CIFAR-10 and FashionMNIST, but the `dataset_guidance` prompt also lists CIFAR-100 and MNIST as pre-cached. If LLM uses `download=False` for CIFAR-100/MNIST, it will get a FileNotFoundError.

### E6: Dockerfile missing commonly-needed ML packages [MEDIUM]
**File:** `researchclaw/docker/Dockerfile:20-24`
**Missing packages:**
- Vision: `timm`, `albumentations`, `kornia`, `Pillow`
- General ML: `einops`, `torchmetrics`, `lightning`
- Graph: `ogb`, `torch-geometric` (optional, large)
- HuggingFace: `transformers`, `datasets`, `accelerate`, `peft` (needed for LLM fine-tuning tasks)
- Utilities: `h5py`, `tensorboard`, `wandb`

### E7: `dataset_guidance` prompt is misleading [MEDIUM]
**File:** `researchclaw/prompts.py:333`
**Current:** "For other torchvision datasets: use `download=True` (network available during setup)"
**Reality:** With default `network_policy: "none"`, network is NOT available at any point. This guidance causes the LLM to generate code with `download=True` that fails with DNS resolution errors.

### E8: No `requirements.txt` generation or processing [LOW]
**File:** `researchclaw/pipeline/executor.py` (code_generation stage)
**Missing:** The LLM is not guided to declare its package requirements. No `requirements.txt` is generated alongside experiment code.

### E9: No LLM-generated setup script support [LOW]
**File:** `researchclaw/experiment/docker_sandbox.py`
**Missing:** No support for a `setup.py` or `download_data.py` script that runs before `main.py` to prepare the environment (download datasets, install packages, etc.).

### E10: No dataset registry / availability matrix [LOW]
**File:** `researchclaw/prompts.py`
**Missing:** The LLM has no knowledge of which datasets are downloadable (and how), which are too large, and what fallback alternatives exist. It should know: "ImageNet is 168GB — use Tiny-ImageNet (200 classes, 500/class) or ImageNet-1k subset instead."

---

## 3. Solution Design

### Architecture: Unified Container with Setup Phase

Replace the broken two-container model with a **single container** running a **wrapper entrypoint script** that handles three phases:

```
┌─────────────────────────────────────────────────────────────────┐
│  Single Docker Container                                         │
│                                                                  │
│  Phase 0: pip install (requirements.txt + auto-detected deps)    │
│           ↓ (network enabled for this phase)                     │
│  Phase 1: setup.py (dataset download, preprocessing)             │
│           ↓ (network enabled for this phase)                     │
│  Phase 2: main.py (experiment execution)                         │
│           ↓ (network optionally disabled via iptables)           │
│                                                                  │
│  Network policy:                                                 │
│  - "none": skip Phase 0/1, no network in Phase 2                │
│  - "setup_only" (NEW default): network in Phase 0+1, disabled   │
│    via iptables before Phase 2                                   │
│  - "pip_only": network in Phase 0 only                          │
│  - "full": network throughout                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Single vs. multi container | Single | Fixes E2; packages survive between phases |
| Network isolation method | iptables drop inside container | Docker doesn't support mid-run network changes |
| Default network policy | `"setup_only"` | Allows pip install + dataset download, but experiment runs isolated |
| Large dataset strategy | Volume-mount + download + fallback hierarchy | ImageNet on host, smaller sets downloadable |
| Entrypoint | Wrapper bash script → python scripts | Separates concerns, easy to debug |
| LLM guidance | Dataset availability matrix in prompt | LLM makes informed decisions about data |

---

## 4. Implementation Plan

### Task E1: Fix `pkg_hint` to list all installed packages [CRITICAL]
**File:** `researchclaw/pipeline/executor.py:2062`
**Change:** Update `pkg_extras` string for docker mode to include ALL pre-installed packages.
```python
# Before:
pkg_extras = ", torchdiffeq, gymnasium, networkx, and pip-installable packages"

# After:
pkg_extras = (
    ", torchvision, torchaudio, matplotlib, seaborn, scipy, "
    "tqdm, torchdiffeq, gymnasium, networkx, PyYAML, "
    "and pip-installable packages (auto-detected from imports)"
)
```
**Effort:** 10 min

### Task E2: Fix Phase 1/Phase 2 container isolation — single-container execution [CRITICAL]
**Files:**
- `researchclaw/docker/entrypoint.sh` (NEW) — wrapper script
- `researchclaw/docker/Dockerfile` (MODIFY) — new entrypoint
- `researchclaw/experiment/docker_sandbox.py` (MODIFY) — refactor execution model

**E2.1: Create wrapper entrypoint script** (`researchclaw/docker/entrypoint.sh`)
```bash
#!/bin/bash
set -e

WORKSPACE="/workspace"
SETUP_ONLY_NETWORK="${RC_SETUP_ONLY_NETWORK:-0}"

# --- Phase 0: Install additional pip packages ---
if [ -f "$WORKSPACE/requirements.txt" ]; then
    echo "[RC] Phase 0: Installing packages from requirements.txt..."
    pip install --no-cache-dir --break-system-packages \
        -r "$WORKSPACE/requirements.txt" 2>&1 | tail -5
    echo "[RC] Phase 0: Package installation complete."
fi

# --- Phase 1: Run setup script (dataset downloads, etc.) ---
if [ -f "$WORKSPACE/setup.py" ]; then
    echo "[RC] Phase 1: Running setup.py (dataset download/preparation)..."
    python3 -u "$WORKSPACE/setup.py"
    echo "[RC] Phase 1: Setup complete."
fi

# --- Network cutoff (if setup_only policy) ---
if [ "$RC_SETUP_ONLY_NETWORK" = "1" ]; then
    echo "[RC] Disabling network for experiment phase..."
    # Drop all outbound traffic (requires NET_ADMIN capability)
    iptables -A OUTPUT -j DROP 2>/dev/null || \
    ip route del default 2>/dev/null || \
    echo "[RC] Warning: Could not disable network (no NET_ADMIN cap)"
fi

# --- Phase 2: Run experiment ---
ENTRY_POINT="${1:-main.py}"
echo "[RC] Phase 2: Running experiment ($ENTRY_POINT)..."
exec python3 -u "$WORKSPACE/$ENTRY_POINT"
```

**E2.2: Update Dockerfile** to use new entrypoint
```dockerfile
# Add entrypoint script
COPY entrypoint.sh /usr/local/bin/rc-entrypoint.sh
RUN chmod +x /usr/local/bin/rc-entrypoint.sh

# Change entrypoint from python3 to wrapper script
ENTRYPOINT ["/usr/local/bin/rc-entrypoint.sh"]
CMD ["main.py"]
```

Note: For `setup_only` network policy, the container needs `--cap-add=NET_ADMIN` for iptables,
or we use `ip route del default` as a fallback (doesn't require capabilities).

**E2.3: Refactor `docker_sandbox.py`**
- Remove separate `_install_deps()` method
- Update `_build_run_command()` to pass entry point as CMD argument
- Handle `requirements.txt` generation in staging dir
- Add `RC_SETUP_ONLY_NETWORK` env var for network cutoff

**E2.4: Update `_build_run_command()` for new model**
```python
def _build_run_command(self, staging_dir, *, entry_point, container_name, network_disabled):
    cfg = self.config
    cmd = [
        "docker", "run",
        "--name", container_name,
        "--rm",
        "-v", f"{staging_dir}:/workspace",
        "-w", "/workspace",
        f"--memory={cfg.memory_limit_mb}m",
        f"--shm-size={cfg.shm_size_mb}m",
    ]

    # For setup_only: container starts with network, then disables it internally
    if cfg.network_policy == "setup_only":
        cmd.extend(["-e", "RC_SETUP_ONLY_NETWORK=1"])
        cmd.extend(["--cap-add=NET_ADMIN"])
        # Don't add --network none (need network for setup phases)
    elif cfg.network_policy == "none":
        cmd.extend(["--network", "none"])
    # "pip_only" and "full" keep normal network

    # ... (volume mounts, GPU, etc. unchanged) ...

    # New: generate requirements.txt from auto-detected deps
    if cfg.network_policy in ("pip_only", "setup_only", "full"):
        if cfg.auto_install_deps or cfg.pip_pre_install:
            self._write_requirements_txt(staging_dir)

    cmd.append(cfg.image)
    cmd.append(entry_point)  # Passed as CMD to entrypoint.sh
    return cmd
```

**Effort:** 3-4 hours

### Task E3: Change default `network_policy` to `"setup_only"` [HIGH]
**File:** `researchclaw/config.py:163`
```python
# Before:
network_policy: str = "none"

# After:
network_policy: str = "setup_only"
```
Also update docstring and config examples.
**Effort:** 15 min

### Task E4: Add LLM-generated `setup.py` for dataset downloads [HIGH]
**Files:**
- `researchclaw/prompts.py` — add `setup_script_guidance` block
- `researchclaw/pipeline/executor.py` — code generation stage generates setup.py alongside main.py

**E4.1: Add `setup_script_guidance` prompt block**
```
## Setup Script (setup.py)
In addition to main.py, generate a setup.py script that handles:
1. Downloading required datasets
2. Any data preprocessing needed before the experiment

The setup.py will run WITH network access before main.py runs without network.
Use standard APIs:
- torchvision.datasets.X(root='/workspace/data', download=True)
- datasets.load_dataset('name', cache_dir='/workspace/data/hf')
- ogb.nodeproppred.PygNodePropPredDataset(name='ogbg-molhiv', root='/workspace/data')
- urllib.request.urlretrieve(url, '/workspace/data/filename')

If all datasets are pre-cached (CIFAR-10, CIFAR-100, MNIST, FashionMNIST, STL-10, SVHN),
you may omit setup.py entirely.
```

**E4.2: Update executor code generation to emit `setup.py`**
In the code generation prompt, instruct the LLM to produce a second file `setup.py` when datasets need downloading. The executor parses the response for both `main.py` and `setup.py` and writes both to the staging directory.

**Effort:** 2 hours

### Task E5: Fix pre-cached dataset list — expand + sync with prompt [MEDIUM]
**File:** `researchclaw/docker/Dockerfile:27-30`

**Add to Dockerfile:**
```dockerfile
# Pre-cache standard datasets for offline use
RUN mkdir -p /opt/datasets && \
    python3 -c "\
import torchvision; \
torchvision.datasets.CIFAR10(root='/opt/datasets', train=True, download=True); \
torchvision.datasets.CIFAR10(root='/opt/datasets', train=False, download=True); \
torchvision.datasets.CIFAR100(root='/opt/datasets', train=True, download=True); \
torchvision.datasets.CIFAR100(root='/opt/datasets', train=False, download=True); \
torchvision.datasets.MNIST(root='/opt/datasets', train=True, download=True); \
torchvision.datasets.MNIST(root='/opt/datasets', train=False, download=True); \
torchvision.datasets.FashionMNIST(root='/opt/datasets', train=True, download=True); \
torchvision.datasets.FashionMNIST(root='/opt/datasets', train=False, download=True); \
torchvision.datasets.STL10(root='/opt/datasets', split='train', download=True); \
torchvision.datasets.STL10(root='/opt/datasets', split='test', download=True); \
torchvision.datasets.SVHN(root='/opt/datasets', split='train', download=True); \
torchvision.datasets.SVHN(root='/opt/datasets', split='test', download=True); \
" && chmod -R a+r /opt/datasets
```

**Update `dataset_guidance` prompt** to match actual pre-cached datasets.

**Effort:** 30 min

### Task E6: Expand Dockerfile with commonly-needed ML packages [MEDIUM]
**File:** `researchclaw/docker/Dockerfile`

**Add package groups:**
```dockerfile
# Extended ML ecosystem
RUN python3 -m pip install \
    timm einops torchmetrics Pillow \
    transformers datasets accelerate peft \
    bitsandbytes sentencepiece protobuf safetensors tokenizers \
    trl evaluate rouge-score \
    h5py tensorboard

# Optional heavy packages (uncomment if needed)
# RUN python3 -m pip install torch-geometric ogb
# RUN python3 -m pip install albumentations kornia
```

**Update `builtin` set in `docker_sandbox.py:372-383`** to include new packages.
**Update `import_to_pip` dict** with new mappings.

**Effort:** 30 min

### Task E7: Fix misleading `dataset_guidance` prompt [MEDIUM]
**File:** `researchclaw/prompts.py:323-369`

**Changes:**
1. Accurately reflect which datasets are pre-cached vs. need downloading
2. Add dataset availability matrix with size info
3. Add fallback hierarchy for large datasets
4. Remove misleading "network available during setup" statement
5. Add guidance based on actual `network_policy`

**New `dataset_guidance` block structure:**
```
## Dataset Availability

### Tier 1: Pre-cached (ALWAYS available, use download=False)
CIFAR-10, CIFAR-100, MNIST, FashionMNIST, STL-10, SVHN
→ Root: /workspace/data

### Tier 2: Downloadable (available if setup.py runs with network)
Any torchvision dataset, HuggingFace datasets, OGB benchmarks
→ Generate a setup.py to download before experiment runs

### Tier 3: Large datasets (require host-side preparation)
ImageNet (168GB), LAION (>1TB), etc.
→ Use smaller alternatives: Tiny-ImageNet (237MB, 200 classes),
  ImageNet-1k subset, or CIFAR-100 as proxy

### ANTI-PATTERNS (NEVER do these):
✗ torch.randn() "ImageNet-like" data → Use real datasets
✗ download=True in main.py → Use setup.py for downloads
✗ download=False for non-cached datasets → Will FileNotFoundError
```

**Effort:** 1 hour

### Task E8: Add `requirements.txt` generation support [LOW]
**Files:**
- `researchclaw/prompts.py` — add requirement to code_generation prompt
- `researchclaw/experiment/docker_sandbox.py` — auto-generate from detected imports

**E8.1: LLM generates `requirements.txt`**
Add to code_generation prompt:
```
If your experiment requires packages not in the standard Docker image,
include a requirements.txt file listing them (one per line with versions).
```

**E8.2: Auto-generate fallback**
In `docker_sandbox.py`, before container execution, auto-detect imports and write `requirements.txt` to staging dir if the LLM didn't provide one.

**Effort:** 1 hour

### Task E9: Add dataset registry for LLM guidance [LOW]
**File:** `researchclaw/data/dataset_registry.yaml` (NEW)
**Content:** Structured registry of common ML datasets with:
- Name, domain, size, download method
- Availability tier (pre-cached / downloadable / host-only)
- Fallback alternatives for large datasets

**Usage:** Injected into experiment_design and code_generation prompts so the LLM makes informed decisions.

**Effort:** 1 hour

### Task E10: Add `entrypoint.sh` non-root pip install support [LOW]
**File:** `researchclaw/docker/Dockerfile`
**Issue:** Container runs as `researcher` (non-root) but pip install needs root (or `--break-system-packages --user`).
**Fix:** In `entrypoint.sh`, use `--user` flag for pip install, or run as root and drop privileges before Phase 2.
**Alternative:** Use `--user 0:0` for the container and run experiment code under `researcher` via `su -c`.

**Effort:** 30 min

---

## 5. Implementation Priority & Dependencies

```
E1 (pkg_hint fix)          ─────────── Immediate, no deps
    ↓
E5 (pre-cache datasets)    ─────────── Dockerfile change
E6 (expand packages)       ─────────── Dockerfile change (parallel with E5)
    ↓
E2 (single-container)      ─────────── Core architecture fix
E3 (default policy)        ─────────── After E2
E10 (non-root pip)         ─────────── After E2
    ↓
E4 (setup.py generation)   ─────────── After E2+E3
E7 (fix prompt guidance)   ─────────── After E5
E8 (requirements.txt)      ─────────── After E2
    ↓
E9 (dataset registry)      ─────────── Last (enhancement)
```

**Phase A (Critical, Day 1):** E1 + E2 + E3 + E5 + E6 + E7 + E10
**Phase B (Important, Day 2):** E4 + E8 + E9

---

## 6. Testing Plan

### Unit Tests
- [ ] `test_docker_sandbox.py`: Test single-container execution with entrypoint.sh
- [ ] `test_docker_sandbox.py`: Test requirements.txt auto-generation
- [ ] `test_docker_sandbox.py`: Test setup.py execution in container
- [ ] `test_docker_sandbox.py`: Test network cutoff in `setup_only` mode
- [ ] `test_prompts.py`: Verify pkg_hint includes all installed packages
- [ ] `test_prompts.py`: Verify dataset_guidance matches Dockerfile pre-cached list

### Integration Tests
- [ ] Docker build succeeds with expanded packages and datasets
- [ ] Experiment using CIFAR-10 (pre-cached) runs without network
- [ ] Experiment using STL-10 (newly pre-cached) runs without network
- [ ] Experiment requiring timm installs it via requirements.txt
- [ ] Experiment with setup.py downloads HuggingFace dataset
- [ ] Network is actually disabled after setup in `setup_only` mode

### E2E Regression
- [ ] Run pipeline with topic "Image Classification on CIFAR-10" → uses real CIFAR-10
- [ ] Run pipeline with topic "Vision Transformer on STL-10" → uses real STL-10
- [ ] Run pipeline with topic "Sentiment Analysis on IMDB" → downloads IMDB via setup.py
- [ ] No experiment produces synthetic/random data as a dataset substitute

---

## 7. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| iptables in container needs NET_ADMIN cap | Medium | Fallback to `ip route del default`; document cap requirement |
| entrypoint.sh changes break existing Docker images | High | Version the image tag (`:v2`); test with both entrypoints |
| Large pre-cached datasets bloat Docker image | Medium | Use multi-stage build; keep optional packages commented |
| HuggingFace download timeouts | Low | Set timeout in setup.py; retry logic |
| LLM generates malicious pip packages | Low | Existing code security validation catches subprocess/network calls |

---

## 8. Files to Modify/Create

| Action | File | Tasks |
|--------|------|-------|
| CREATE | `researchclaw/docker/entrypoint.sh` | E2 |
| CREATE | `researchclaw/data/dataset_registry.yaml` | E9 |
| MODIFY | `researchclaw/docker/Dockerfile` | E2, E5, E6, E10 |
| MODIFY | `researchclaw/experiment/docker_sandbox.py` | E2, E3, E8 |
| MODIFY | `researchclaw/config.py` | E3 |
| MODIFY | `researchclaw/pipeline/executor.py` | E1, E4 |
| MODIFY | `researchclaw/prompts.py` | E4, E7 |
| MODIFY | `tests/test_docker_sandbox.py` | Tests |

---

## 9. Comparison: Before vs. After

| Aspect | Before | After |
|--------|--------|-------|
| Available packages in prompt | 9 packages listed | 15+ packages listed |
| Runtime pip install | Broken (Phase 1/2 isolation bug) | Working (single container) |
| Default network policy | `"none"` (no install, no download) | `"setup_only"` (install+download, then isolated) |
| Pre-cached datasets | 2 (CIFAR-10, FashionMNIST) | 6 (+ CIFAR-100, MNIST, STL-10, SVHN) |
| Dataset download support | None | setup.py with network access |
| Dockerfile ML packages | ~15 packages | ~30+ packages |
| Large dataset handling | Falls back to synthetic | Fallback hierarchy + alternatives |
| Requirements declaration | None | requirements.txt + auto-detect |
