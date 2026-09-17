# OrcaRouter provider

[OrcaRouter](https://www.orcarouter.ai) is an OpenAI-compatible AI gateway
built for both models and agents: adaptive routing, automatic failover,
zero-markup inference, observability, guardrails, and agent-tool governance.
It is a first-class provider here — it appears in the provider presets, the
`researchclaw init` wizard, the server's provider API and settings page, and
the model catalogue, exactly like every other preset.

There are **two explicit ways to authenticate**, and both produce the same
kind of OrcaRouter API key:

| Choice | Provider id | Label | Where the key comes from |
| --- | --- | --- | --- |
| Existing key | `orcarouter` | OrcaRouter — API | You paste an `sk-orca-…` key. |
| Account login | `orcarouter-oauth` | OrcaRouter — Auth | OAuth 2.0 + PKCE mints one for your account. |

They are separate entries rather than one ambiguous button, because they
have different failure modes: an existing-key user must not be pushed into a
browser flow, and a revoked account grant must not be masked by a stale
pasted key.

## Inference endpoint

```yaml
llm:
  provider: "orcarouter"          # or "orcarouter-oauth"
  base_url: "https://api.orcarouter.ai/v1"
  api_key_env: "ORCAROUTER_API_KEY"
  primary_model: "orcarouter/auto"
  fallback_models:
    - "deepseek/deepseek-v4-pro"
    - "deepseek/deepseek-v4-flash"
```

The wire format is OpenAI-compatible, so the existing `LLMClient` is used
unchanged with `Authorization: Bearer <key>`. Model ids keep their
`vendor/model` namespace.

## Authentication and inference use different origins

- consent screen: `https://www.orcarouter.ai/auth`
- code exchange: `https://www.orcarouter.ai/api/v1/auth/keys`
- inference and model list: `https://api.orcarouter.ai/v1`

`https://api.orcarouter.ai/v1/auth/keys` is a 404 — the relay lives at `/v1`
on the API origin, the auth endpoints do not. The two origins are configured
independently and are never derived from one another:

| Variable | Purpose |
| --- | --- |
| `ORCA_AUTH_BASE_URL` | Auth origin override (explicit wins) |
| `ORCA_API_BASE_URL` | Inference origin override (explicit wins) |
| `ORCA_BASE_URL` | Shared self-hosted base, used for whichever of the two is not set explicitly |

Non-loopback origins must be HTTPS; plain HTTP is accepted only for
`localhost`, `127.0.0.1`, and `[::1]`.

## Connecting an account (OAuth 2.0 + PKCE, S256)

```bash
researchclaw orcarouter login             # browser callback (loopback), when available
researchclaw orcarouter login --flow oob  # print a URL, paste the code back
researchclaw orcarouter models            # what this account can actually call
researchclaw orcarouter status
```

The **loopback** flow is the default on a workstation: the CLI binds
`127.0.0.1:0` and receives the code directly. **Out-of-band** is for SSH
sessions, containers, and the hosted web UI, where no browser callback can
reach the user's machine — there is no redirect URI to pre-register either
way.

Properties enforced by the implementation:

- the verifier is 32 bytes of `secrets.token_bytes` per attempt and never
  leaves the process — not in a URL, a log line, an exception, or a
  screenshot;
- only `base64url(sha256(verifier))` travels on the authorize URL, and the
  method is always `S256` (never `plain`: the consent screen can hand a code
  to a human even in the callback flow);
- `state` is compared with `hmac.compare_digest` before the code is used;
- denial, state mismatch, timeout, cancel, expired/reused code (403),
  rejected request (400), rate limit (429), and transport errors all end the
  attempt with an actionable message and release the listener.

The grant is a **durable API key, not a refresh token**. It is stored at
`~/.researchclaw/orcarouter/credentials.json` (mode `0600`) — the project's
existing user-level state directory — and reused on every start until
OrcaRouter revokes it. There is no refresh grant to call, and none is
invented: `researchclaw orcarouter login` refuses to re-authorize when a
usable key already exists, because the consent endpoint allows 10 PKCE-issued
keys per user per 24 hours.

### Revocation

Revoke at <https://www.orcarouter.ai/console/authorized-apps>. A `401` from
the relay is terminal: exactly the credential *generation* that made the
rejected request is marked `needs_reauth`, the stored secret is kept (so a
transient misclassification is not irreversible account loss), and a late
failure from an old request cannot mark a newer credential broken.

## Models

The catalogue is `GET https://api.orcarouter.ai/v1/models`, requested with
your own bearer key so the answer is what *your* workspace can call. When
live discovery succeeds it is authoritative; a small verified seed exists
only for a cold start or an outage, is labelled degraded in the UI, and is
never mixed into a successful live result.

Each entry point gets its own filtered list:

| Entry point | Filter |
| --- | --- |
| Text chat / agent | `?capability=chat`, text wire endpoint, excludes image/video/rerank-only models |
| Multimodal | the above, plus a **declared** `architecture.input_modalities` entry for the uploaded modality |
| Embedding | `?capability=embedding` / `embeddings` endpoint |
| Image generation | `?capability=image` / `image-generation` endpoint |
| Video generation | `openai-video` endpoint |
| Rerank | `jina-rerank` endpoint |

Capability is never inferred from a model's name, and a model that does not
declare a required input modality is excluded rather than assumed
compatible. Changing the provider, the entry point, or the attachment type
recomputes the options and clears a selection that is no longer compatible.

## Server / settings UI

With `researchclaw serve`, the provider settings page is at `/providers`
and the JSON API at `/api/providers`. The API key stays on the server: the
model endpoint returns minimal model metadata and a masked key (`sk-orca…4f2a`)
only — a page never receives a credential it could leak.

### Screenshots / evidence bundle

The screenshots of that page are a **build product of a run**, regenerated by
the generator rather than drawn by hand:

```bash
python scripts/orcarouter_ui_evidence.py            # writes ./orca-evidence/
python scripts/orcarouter_ui_evidence.py --out DIR  # or ORCA_EVIDENCE_OUT
pytest tests/test_orcarouter_ui.py -k evidence      # generates and checks it
```

The generator boots the real app, drives the real page in Chromium, and writes
`manifest.json` plus `auth-methods.png` and `text-model-dropdown.png`
(declared-modality chat models are only added by the multimodal entry point
when one exists). Output goes to `orca-evidence/` at the repository root, which
is git-ignored: evidence has to be the product of a run of the code under test,
so a bundle carried in a patch is stale by construction and is refused. Nothing
in the bundle is hand-written — the manifest records each PNG's `sha256`, and
the test suite re-checks those digests, so an edited or stale screenshot fails
rather than being believed.

`manifest.json` uses the delivery gate's schema rather than a free-form report:
`automation` is an object carrying `framework`, `passed`, `catalog_source`
(the live `?capability=chat` catalogue URL) and the model counts, and every
entry in `artifacts` carries `kind`, `path`, `sha256` and the `ui` assertions
that screenshot has to prove. `validate_bundle` re-checks all of it — Playwright
provenance, the catalogue URL, count agreement, PNG size and digest, and the
per-screenshot UI assertions. It runs twice: once inside the generator (so a
bundle the gate would reject never leaves the machine) and once in
`test_gui_evidence_bundle_is_generated_into_the_repository`, which runs the
generator end to end and then holds the bundle it produced to the same
checklist.

## Evidence

| Item | Source |
| --- | --- |
| OpenAI-compatible inference | `https://api.orcarouter.ai/v1/chat/completions` (verified with a real request) |
| Model list | `https://api.orcarouter.ai/v1/models` — requires `Authorization: Bearer`; 16 models returned on 2026-09-16 |
| Consent screen | `https://www.orcarouter.ai/auth` |
| Code exchange | `POST https://www.orcarouter.ai/api/v1/auth/keys` |
| Discovery document | `https://www.orcarouter.ai/.well-known/openid-configuration` |
| Key management | `https://www.orcarouter.ai/console/token` |
| Revocation | `https://www.orcarouter.ai/console/authorized-apps` |
| Terms, privacy, subprocessors | OrcaRouter Trust Center — `https://www.orcarouter.ai/trust` (verified HTTP 200 on 2026-09-16; the public site exposes no `/terms` path, so the trust center is the authoritative legal document location) |
| Product / company | `https://www.orcarouter.ai` · pricing `https://www.orcarouter.ai/pricing` |
| Relay status / incidents | `https://status.orcarouter.ai` (linked from the site) |
| Community | Discord `discord.gg/YEubt8enRA` · X `https://x.com/OrcaRouter` |
| Maintenance owner | OrcaRouter team (engineering contact via the OrcaRouter Discord) |
| Verification date | 2026-09-16 |
| Affiliation | This integration is contributed by an engineer on the OrcaRouter team. |
