# Security Considerations

This document describes security-relevant properties of Primus as a **YAML-driven training framework** for AMD GPUs (ROCm, RCCL, containers). It is intended for operators, platform engineers, and security reviewers. It does not replace organizational policies, threat models, or vendor hardening guides.

**Related documentation:** [Environment variables](../03-configuration-reference/environment-variables.md), [Installation](../01-getting-started/installation.md), [CLI reference](../02-user-guide/cli-reference.md).

---

## 1. Overview

| Aspect | Description |
|--------|-------------|
| Role | Primus orchestrates distributed **training** jobs; it is **not** a general user-facing network service. |
| Authentication / authorization | **No built-in** authentication, authorization, or multi-tenant isolation in Primus itself. |
| Responsibility | **Security posture is determined by** the scheduler, container runtime, network, storage, identity systems, and operational practices of the deployment environment. |

Treat Primus like privileged infrastructure software: run it on appropriately isolated hosts and networks, and govern secrets and data the same way you would for large-scale ML training elsewhere.

---

## 2. Secrets management

Secrets are commonly passed as **environment variables** consumed by Primus, launchers, or third-party libraries.

| Variable (examples) | Typical use |
|---------------------|-------------|
| `HF_TOKEN` | Hugging Face token for **gated models** and authenticated downloads. |
| `WANDB_API_KEY` | Weights & Biases API key for experiment logging. |
| `DATABRICKS_HOST` / `DATABRICKS_TOKEN` | Databricks or MLflow-related credentials when those integrations are used. |

**Practices**

| Practice | Detail |
|----------|--------|
| Do not hardcode secrets | Avoid putting tokens or passwords directly in YAML, shell history, or committed scripts. |
| Prefer indirection | Use **`${VAR}`** substitution in configs to reference environment-injected values rather than literals. |
| Slurm | Use **`--export`** deliberately; prefer site-specific **secret injection** or **credential helpers** where available. |
| Containers | Pass secrets with **`--env`** or via **`runner/.primus.yaml`** env forwarding—never bake them into images. |
| Rotation | Rotate API keys and tokens on a schedule and after personnel or scope changes. |

A broader catalog of variables appears in [Environment variables](../03-configuration-reference/environment-variables.md).

---

## 3. Container security

Primus-oriented container runs often require **elevated access** so ROCm, profilers, and high-performance networking behave correctly.

**Common high-privilege options**

| Option | Typical purpose |
|--------|-----------------|
| `--privileged true` | Broad device access (often required for ROCm workflows on some setups). |
| `--cap-add SYS_PTRACE` | Debugging and profiling tooling. |
| `--cap-add CAP_SYS_ADMIN` | Administrative operations expected by parts of the ROCm/tooling stack. |
| `--security-opt seccomp=unconfined` | Relaxes seccomp constraints for compatibility with drivers and tools. |
| `--ipc host` | Shared memory semantics for large tensors and collectives. |
| `--network host` | Host networking—frequently used for **multi-node RCCL** performance and simplicity. |

**Device access (examples)**

| Device | Role |
|--------|------|
| `/dev/kfd` | ROCm kernel interface. |
| `/dev/dri` | GPU render nodes. |
| `/dev/infiniband` | InfiniBand character devices when using IB. |

**Risks**

| Risk | Why it matters |
|------|----------------|
| Privileged containers | Substantial **host** access; container escape or compromise has high impact. |
| Host networking | Exposes the container to the **host’s network namespace**; services may bind broadly. |
| Shared IPC | Potential for **cross-process interference** or information leakage if workloads share hosts improperly. |

**Mitigations**

| Mitigation | Detail |
|------------|--------|
| Dedicated training nodes | Run training on **isolated** machines rather than mixed with user-facing services. |
| Network controls | Apply **firewall rules** and **segmentation** so only required ports and peers are reachable. |
| Trusted images | Pull from **trusted registries**, pin digests, and verify image provenance. |
| Monitoring | Track **CPU, memory, GPU, and network** usage; alert on anomalous processes or egress. |

---

## 4. Third-party dependencies

Primus integrates **third-party submodules** and Python packages; each carries its own license and maintenance cadence.

**Representative submodules**

| Component | Notes (non-exhaustive) |
|-----------|-------------------------|
| Megatron-LM | MIT License; NVIDIA upstream. |
| TorchTitan | Apache 2.0; PyTorch / Meta ecosystem. |
| MaxText | Apache 2.0; Google upstream. |
| Megatron-Bridge | NVIDIA NeMo ecosystem. |
| Emerging-Optimizers | NVIDIA NeMo ecosystem. |
| HummingbirdXT | AMD AGI ecosystem. |

**Python dependencies**

Runtime tooling often includes packages such as **loguru**, **wandb**, **nltk**, **matplotlib**, **mlflow**, and others as declared in project requirements—verify the canonical list in the repository’s `requirements.txt` (or lockfile) for your revision.

**Recommendations**

| Recommendation | Rationale |
|----------------|-----------|
| Pin versions | Reproducible builds and controlled upgrade paths. |
| Update submodules | Security and correctness fixes flow from upstream projects. |
| Monitor advisories | Subscribe to upstream security notices for frameworks you enable. |

---

## 5. Network security

| Property | Detail |
|----------|--------|
| RCCL / NCCL traffic | **Not encrypted** at the application layer; assumes a **trusted network path**. |
| Coordination | **`MASTER_ADDR`** and **`MASTER_PORT`** should reside on a **private** or otherwise **trusted** segment. |
| InfiniBand | Often on a **dedicated fabric**; still treat adjacent compromised hosts as in-scope for lateral movement. |
| TLS / mTLS | Primus does **not** provide TLS or mTLS for inter-node training traffic by default. |

For physical and logical networking topics, see [Multi-node networking](../04-technical-guides/multi-node-networking.md).

---

## 6. Data security

| Asset | Consideration |
|-------|-----------------|
| Training data | May include **PII**, licensed corpora, or export-controlled material—classify and restrict accordingly. |
| Checkpoints | Contain **full model state**; treat as sensitive intellectual property. |
| Storage permissions | Use **least privilege** on shared filesystems and object stores. |
| `HF_TOKEN` | Grants access to **gated** Hugging Face assets—protect like any other long-lived credential. |

Checkpoint formats and operational practices are described in [Checkpoint management](../04-technical-guides/checkpoint-management.md).

---

## 7. What is not verified

The following items reflect **typical gaps** in public-facing evidence for many research and infrastructure codebases; confirm against your organization’s audits and CI for your fork and deployment.

| Topic | Status (evidence-based caveat) |
|-------|--------------------------------|
| Independent security audit | **No** comprehensive third-party audit of this codebase is asserted here. |
| CI secrets scanning | **No** guarantee of automated secret detection in CI unless your pipeline adds it. |
| Dependency vulnerability scanning | **No** guarantee of continuous SCA unless your pipeline adds it. |
| Container images | Images may contain **unpatched** OS or Python packages—scan and rebuild on a schedule. |
| RCCL / NCCL traffic | **Not** encrypted or mutually authenticated by default; rely on network trust boundaries. |

Use this section as a checklist for **your** production controls: add scanning, signing, policy-as-code, and periodic reviews appropriate to your threat model.
