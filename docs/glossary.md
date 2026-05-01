# Glossary

**Action** — an operation an agent executes against an external environment: calling a tool, querying an API, or signaling completion. Contrasts with *thought*, which is internal.

**Action space** — the set of operations available to an agent at any given step. With conventional tools, the action space is enumerated (the agent selects from a list of tool schemas). With code execution, the action space is generative (the agent constructs arbitrary operations within the constraints of the language and sandbox).

**Agent** — a system that uses an LLM to make autonomous decisions about how to accomplish a task. Distinguished from a single LLM call by the presence of a decision loop where the model's output determines what happens next.

**Agent loop** — the iterative cycle through which an agent operates. The structure of this loop (interleaved vs. separated reasoning and action, linear vs. branching exploration) defines the agent type.

**Chain-of-thought** — a prompting technique where the LLM generates intermediate reasoning steps before producing a final answer. Operates purely from internal knowledge with no external actions.

**Chunking** — the process of splitting source documents into smaller retrievable units for indexing. Chunk granularity — from sentences to paragraphs to full documents — determines the precision and context trade-off in retrieval.

**Constrained decoding** — a structured output enforcement mechanism where the model is restricted at the token level to only produce sequences valid against a schema. Guarantees structural validity by construction, unlike prompt-instructed or post-hoc validation approaches.

**Context window** — the finite token budget available for LLM input. All prompt content, conversation history, and trajectory must fit within this limit.

**Embedding** — a dense vector representation of text in a continuous vector space, where semantic similarity maps to spatial proximity. Used by retrieval tools and semantic memory to find relevant content via similarity search.

**Few-shot prompting** — a prompting technique that provides concrete examples of desired input-output behavior rather than describing the behavior with instructions. The model infers the pattern from the examples and applies it to the current input.

**Fine-tuning** — training a base model further on domain-specific data to adjust its weights for better performance in a target domain. Embeds knowledge into model parameters, contrasting with retrieval (external, updatable) and prompting (transient, per-request).

**Grounding** — anchoring LLM-generated output to verifiable external sources rather than relying solely on parametric knowledge. The primary mechanism for reducing hallucination. Retrieval-augmented generation is the most common grounding pattern.

**Hallucination** — generating plausible but factually incorrect information. Occurs when an LLM draws on pattern completion rather than grounded knowledge.

**Idempotency** — the property of an operation that produces the same result whether executed once or multiple times. Critical for agent reliability because the agent loop may invoke the same tool repeatedly through retries, replanning, or self-correction.

**Observation** — the result returned by an external environment after an agent executes an action. Provides the grounding that feeds into subsequent reasoning.

**Parametric knowledge** — knowledge encoded in a model's weights from training data. Contrasts with retrieved or grounded knowledge that is provided at inference time. The boundary between what the model "knows" and what it must look up determines when retrieval is necessary and when hallucination risk is high.

**Prompt injection** — an attack where untrusted input (user messages, retrieved documents, tool observations) contains instructions that override or subvert the system prompt. The primary security threat model for agents that accept external input.

**Reasoning trace** — a verbal, free-form output where the LLM articulates its thinking: interpreting state, decomposing problems, tracking progress, or deciding next steps. Called *thought* in the ReAct pattern.

**Retrieval-augmented generation (RAG)** — a pattern where an agent retrieves relevant information from an external knowledge base and incorporates it into the LLM's context before generating a response. Combines the LLM's reasoning ability with up-to-date, traceable external knowledge.

**Run** — a single end-to-end execution of an agent on a task. "Within-run" adaptation uses observations from the current execution; "across-run" adaptation carries lessons from prior executions.

**Sandbox** — an isolated execution environment that constrains what generated code can access — filesystem scope, network endpoints, process spawning, resource consumption. The mechanism that makes code execution a bounded tool rather than unrestricted arbitrary code execution.

**Self-correction** — an agent detecting and recovering from its own errors within a run. The agent receives error feedback (a failed tool call, a validation error, an unhelpful observation), reasons about what went wrong, and adjusts its next action. Distinct from external feedback or cross-run learning.

**Structured output** — a model response constrained to conform to a predefined schema (e.g., JSON matching a type definition). Eliminates the need for brittle text parsing by guaranteeing the response shape at the protocol level.

**System prompt** — the persistent instruction block sent at the start of every LLM request, defining the agent's role, behavioral constraints, and expectations. Remains constant across all turns of the agent loop. Occupies the highest priority position in the prompt hierarchy.

**Tool** — an executable function available to an agent, defined by a schema describing its parameters and purpose. The mechanism through which agents take actions.

**Trajectory** — the complete sequence of thoughts, actions, and observations produced during a run. Serves as both the agent's working context and a human-readable audit trail.
