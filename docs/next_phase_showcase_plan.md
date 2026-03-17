# AutoResearchClaw — Phase 5: Showcase Website & Sample Papers

> Created: 2026-03-15
> Status: **Website Built** — static site deployed, showcase papers pending generation
> Prerequisites: Phase 3 regression tests complete, all fixes pushed to origin/main

---

## 1. Goals

1. **Generate representative showcase papers** across diverse research domains to demonstrate pipeline capabilities
2. **Build a static website** to publicly present AutoResearchClaw's pipeline, features, and sample outputs
3. **Establish a paper gallery** with downloadable PDFs and code for each showcase paper

---

## 2. Showcase Paper Generation

### 2.1 Test Case Selection Strategy

Select 5-6 topics across different ML subfields, difficulty levels, and experiment types to maximize diversity:

| # | Topic | Domain | Experiment Type | Why |
|---|-------|--------|-----------------|-----|
| S1 | "Curriculum Learning with Adaptive Difficulty Scheduling for Image Classification" | CV + Training Strategy | CIFAR-10/100, standard benchmark | Accessible, clear baselines, testable |
| S2 | "Prompt-Length-Aware Routing for Mixture-of-LoRA Experts in Instruction-Following" | NLP + PEFT | QLoRA + Qwen-2.5-3B fine-tuning | Showcases LLM fine-tuning capability |
| S3 | "Graph Attention Networks with Learnable Edge Features for Molecular Property Prediction" | GNN + Chemistry | OGB-MolHIV benchmark | Cross-domain application |
| S4 | "Entropy-Guided Exploration Bonuses for Sparse-Reward Continuous Control" | RL | MuJoCo locomotion tasks | Complex multi-algorithm comparison |
| S5 | "Spectral Normalization Effects on Mode Collapse in Conditional GANs for CIFAR-10" | Generative Models | GAN training on CIFAR-10 | Visual results + quantitative metrics |
| S6 | "Test-Time Adaptation via Batch Normalization Statistics for Distribution Shift" | Domain Adaptation | CIFAR-10-C corruption benchmark | Practical, real-world relevance |

### 2.2 Selection Criteria for Showcase

From the generated papers, select 3-4 best ones based on:

- **Paper Quality Score**: >= 7/10 from the built-in quality assessment
- **Experiment Completeness**: All methods ran, ablations show differentiation
- **Visual Quality**: Charts are clean, metrics are meaningful
- **Topic Diversity**: No two showcase papers from the same subfield
- **Narrative Quality**: Clear story from motivation through results to conclusions

### 2.3 Configuration Template

```yaml
# showcase_config_template.yaml
llm:
  provider: "azure_openai"
  model: "gpt-5.1"
  max_tokens: 16384

experiment:
  backend: "docker"
  timeout_sec: 3600  # generous budget for quality
  docker:
    gpu_enabled: true
    memory_limit_mb: 40960
    network_policy: "full"

pipeline:
  target_conference: "iclr_2026"
  max_refinement_iterations: 3
  enable_code_review: true
```

---

## 3. Static Website Design

### 3.1 Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Static Site Generator | **Astro** or **Next.js (static export)** | Modern, fast, Markdown-native |
| Styling | **Tailwind CSS** | Utility-first, rapid prototyping |
| Hosting | **GitHub Pages** or **Vercel** | Free, auto-deploy from repo |
| PDF Rendering | **PDF.js** embedded viewer | In-browser paper viewing |
| Domain | `autoresearchclaw.github.io` or custom | GitHub Pages default |

**Alternative (simpler):** Pure HTML/CSS/JS with no build step — suitable if we want zero dependencies and maximum portability.

### 3.2 Site Structure

```
/                          → Landing page (hero, pipeline overview, CTA)
/pipeline                  → Interactive pipeline visualization (23 stages)
/papers                    → Gallery of showcase papers
/papers/{paper-id}         → Individual paper page (PDF viewer + metadata)
/features                  → Feature highlights and comparison
/getting-started           → Quick start guide
```

### 3.3 Page Designs

#### Landing Page (`/`)
- **Hero section**: Logo, tagline ("Chat an Idea. Get a Paper."), demo GIF/video
- **Pipeline overview**: Animated or scrollable 23-stage diagram
- **Key stats**: "1039 tests passed", "23 autonomous stages", "GPU-accelerated experiments"
- **Paper carousel**: 3-4 showcase papers with thumbnails
- **CTA**: GitHub link, quickstart command

#### Pipeline Page (`/pipeline`)
- Interactive visualization of the 23-stage pipeline
- Each stage clickable → shows description, inputs, outputs, example
- Stage groups: Topic Discovery → Literature → Experiment Design → Code → Execution → Writing → Review
- Highlight: Docker sandbox, multi-agent review, citation verification

#### Paper Gallery (`/papers`)
- Grid/card layout with paper thumbnails
- Each card shows: title, topic domain, quality score badge, abstract preview
- Filter by domain (CV, NLP, RL, etc.)
- Sort by quality score

#### Individual Paper Page (`/papers/{id}`)
- Embedded PDF viewer (PDF.js)
- Metadata sidebar: topic, quality score, stages completed, runtime, GPU used
- Download buttons: PDF, LaTeX source, experiment code
- Quality assessment summary (strengths, weaknesses from internal review)
- Experiment charts gallery

#### Features Page (`/features`)
- Feature cards with icons:
  - Real literature search (arXiv + Semantic Scholar)
  - Docker-sandboxed experiments with GPU passthrough
  - Multi-agent peer review
  - Iterative refinement loop
  - Conference-ready LaTeX output
  - Hardware-aware experiment design
  - Citation verification
  - LLM fine-tuning support (QLoRA/LoRA)

### 3.4 Assets Needed

| Asset | Source | Status |
|-------|--------|--------|
| Logo | Existing (`image/logo.png`) | Done |
| Framework diagram | Existing (`image/framework.png`) | Done |
| Pipeline stage icons | Need to create | TODO |
| Paper thumbnails | Generate from LaTeX PDFs | TODO |
| Demo video/GIF | Screen recording of pipeline run | TODO |
| Quality score badges | SVG badges | TODO |

---

## 4. Repository Structure for Website

```
website/
├── public/
│   ├── papers/
│   │   ├── paper-01/
│   │   │   ├── paper.pdf
│   │   │   ├── paper.tex
│   │   │   ├── code/
│   │   │   ├── charts/
│   │   │   └── metadata.json
│   │   └── paper-02/
│   │       └── ...
│   └── assets/
│       ├── logo.png
│       ├── framework.png
│       └── icons/
├── src/
│   ├── pages/
│   │   ├── index.astro        (or .html)
│   │   ├── pipeline.astro
│   │   ├── papers/
│   │   │   ├── index.astro
│   │   │   └── [id].astro
│   │   ├── features.astro
│   │   └── getting-started.astro
│   ├── components/
│   │   ├── Header.astro
│   │   ├── PipelineStage.astro
│   │   ├── PaperCard.astro
│   │   └── QualityBadge.astro
│   ├── layouts/
│   │   └── Base.astro
│   └── styles/
│       └── global.css
├── astro.config.mjs
├── tailwind.config.js
└── package.json
```

---

## 5. Paper Metadata Format

Each showcase paper includes a `metadata.json`:

```json
{
  "id": "curriculum-learning-cifar",
  "title": "Curriculum Learning with Adaptive Difficulty Scheduling...",
  "domain": "Computer Vision",
  "tags": ["curriculum-learning", "image-classification", "CIFAR-10"],
  "quality_score": 7.5,
  "verdict": "accept",
  "target_conference": "ICLR 2026",
  "generated_date": "2026-03-15",
  "runtime_minutes": 45,
  "gpu": "NVIDIA RTX 6000 Ada (49GB)",
  "stages_completed": 23,
  "abstract": "...",
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "files": {
    "pdf": "paper.pdf",
    "latex": "paper.tex",
    "code": "code/",
    "charts": "charts/",
    "references": "references.bib"
  }
}
```

---

## 6. Implementation Timeline

| Phase | Task | Estimated Effort | Dependencies |
|-------|------|-----------------|--------------|
| 5.1 | Generate 5-6 showcase papers | 1 day (parallel runs) | Phase 3 complete |
| 5.2 | Review & select 3-4 best papers | 2 hours | 5.1 |
| 5.3 | Compile LaTeX → PDF for selected papers | 1 hour | 5.2 |
| 5.4 | Set up website repo structure | 1 hour | — |
| 5.5 | Build landing page + pipeline visualization | 4 hours | 5.4 |
| 5.6 | Build paper gallery + individual pages | 3 hours | 5.2, 5.4 |
| 5.7 | Build features page | 2 hours | 5.4 |
| 5.8 | Deploy to GitHub Pages | 30 min | 5.5-5.7 |
| 5.9 | Create demo video/GIF | 2 hours | Pipeline working |

**Total estimated**: ~2 days

---

## 7. Deployment Options

### Option A: GitHub Pages (Recommended)
- Free hosting on `autoresearchclaw.github.io`
- Auto-deploy via GitHub Actions on push to `website` branch or `docs/` folder
- No server costs, CDN included

### Option B: Vercel
- Free tier supports static sites
- Faster builds, preview deployments for PRs
- Custom domain support

### Option C: Netlify
- Similar to Vercel, free tier available
- Form handling if needed later

**Recommendation**: Start with GitHub Pages for simplicity, migrate to Vercel if we need preview deployments or custom domain.

---

## 8. Content Checklist

- [x] Finalize showcase paper topics (Section 2.1)
- [ ] Run all showcase experiments
- [ ] Review and select best 3-4 papers
- [ ] Compile PDFs from LaTeX
- [ ] Create paper metadata.json for each
- [x] Design pipeline visualization (interactive or static) — interactive click-to-expand
- [x] Write feature descriptions — 16 feature cards + comparison table
- [x] Create getting-started guide (adapted from README) — 7-step guide
- [ ] Record demo video/GIF
- [x] Build and deploy website — pure HTML/CSS, GitHub Pages via Actions
- [x] Test on mobile/tablet — responsive CSS with nav toggle
- [ ] Add analytics (optional, e.g., Plausible)

---

## 9. Open Questions

1. **Custom domain?** — Do we want a custom domain (e.g., `autoresearchclaw.com`) or is `github.io` sufficient?
2. **Video demo?** — Should we include a screen recording of a full pipeline run, or is a GIF of key stages enough?
3. **Interactive pipeline?** — Full interactive SVG/Canvas pipeline diagram vs. static image with tooltips?
4. **Paper format** — Show papers as embedded PDFs, or convert to HTML for better web rendering?
5. **Localization** — Website in English only, or mirror the multi-language READMEs?

---

## 10. Success Metrics

- At least 3 showcase papers with quality score >= 7/10
- Website loads in < 2 seconds on 3G connection
- All showcase paper PDFs downloadable
- Pipeline visualization accurately represents all 23 stages
- GitHub stars / traffic increase after website launch (track via GitHub Insights)
