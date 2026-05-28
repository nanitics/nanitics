# Migrating from the `WorkingMemory` workaround

This guide is for consumers who reached for `WorkingMemory` to make an
agent's second `Agent.run` remember its first. The thread-identity
substrate (`thread_key` + `ThreadStore`) is the supported answer; this
page walks through the symptom, why the workaround can't fully work,
and the replacement.

Related background: [Memory § Behavioral Continuity](memory.md#behavioral-continuity).

## Symptom

The pattern looks like this. An agent answers a first question, then a
second `Agent.run` arrives and the agent has no idea what it said
before. Each call starts from a fresh message list; nothing survives
between them.

The natural reach is `WorkingMemory`, because it appears in the docs as
"state that persists across steps." Two shapes show up in the wild:

1. **Journaling.** The system prompt tells the agent to write its
   action into a `<working_memory>## Latest turn ...</working_memory>`
   section on every turn, so the next call's `WorkingMemoryProvider`
   re-injects it.

   <!-- verify: skip — illustrative anti-pattern; not meant to compile against the public API -->
   ```python
   agent = ReActAgent(
       name="drafter",
       llm_client=client,
       emitter=emitter,
       system_prompt=(
           "You are a drafter. On every turn, journal your draft into "
           "<working_memory>## Latest draft\n...</working_memory> so you can "
           "see it on the next call."
       ),
       tools=[],
       working_memory=InMemoryWorkingMemory(),
       context_providers=[WorkingMemoryProvider(memory)],
   )

   await agent.run("Draft a pitch")
   await agent.run("Now make it shorter")  # second call expects to see the first draft
   ```

2. **Snapshot save/load.** Some consumers route around
   `ReActAgent`'s `working_memory.reset()`-on-run-start by snapshotting
   the store's content after each `run` and re-loading it before the
   next one. Mechanically it works for a single agent in a single
   process; the structural problems below still apply.

## Why it doesn't fully work

Three concrete failures, each independent of the others.

**1. The `<nanitics:context provider="working_memory">` wrapper tells
the model "this is injected context, not your prior turn."**
`WorkingMemoryProvider` content reaches the LLM wrapped in a
namespaced envelope on a `role="user"` message, by design — it's how
the SDK signals provider-injected content distinctly from the
conversation. The model reads the wrapped block as untrusted external
data, not as something it personally produced. It will reference the
content as a fact but it won't believe it authored it, and tool-use
chains break: the prior turn's `tool_call` IDs and `tool_result`
correlation are gone, so any logic that depends on the model
continuing a tool-use loop across runs fails. The runnable proof is
[`examples/memory/working_memory_vs_threads.py`](../../examples/memory/working_memory_vs_threads.py),
which asserts the wrapper's presence in the journaling scenario and
its absence in the threads scenario.

**2. The journaling burden lives in the prompt.** Every turn costs
tokens on instructions that tell the agent to write a particular
`<working_memory>` shape. Every prompt edit risks the agent dropping a
section (working memory is full-replacement per
[memory.md § Pitfalls](memory.md#pitfalls)). The substrate that should
be invisible to the model becomes the most fragile part of the prompt.

**3. `working_memory.reset()` runs at the start of every `_execute()`.**
`ReActAgent` calls `working_memory.reset()` before the first LLM call
in a run. This means the *single-run* multi-step path is the only
place `WorkingMemory` carries content between LLM calls — for
*cross-run* continuity, every consumer of this pattern has to monkey
around with save/load plumbing, and the substrate-comparison example
demonstrates this by running Scenario A as a *single multi-step
`run()`* rather than two `run()` calls. The two-run shape simply
doesn't work for `WorkingMemory` without overriding `reset()` or
re-priming the store from a snapshot. Concurrent invocations make it
worse: two runs on the same agent racing on the same `WorkingMemory`
will overwrite each other's snapshots.

These are not bugs in `WorkingMemory`. `WorkingMemory` is
*information continuity* — content the agent should know but not
believe it produced. Behavioral continuity — what the agent itself
said last turn — is a different substrate.

## The replacement

Drop the journaling instructions; drop any save/load plumbing. Add a
`ThreadStore` to the agent and pass `thread_key` to each call:

<!-- verify: skip — illustrative wiring; `client` and `emitter` are caller-supplied and the `await` runs inside an async context -->
```python
from nanitics.composition import InMemoryThreadStore
from nanitics.strategies import ReActAgent

store = InMemoryThreadStore()

agent = ReActAgent(
    name="drafter",
    llm_client=client,
    emitter=emitter,
    system_prompt="You are a drafter.",
    tools=[],
    thread_store=store,
)

await agent.run("Draft a pitch", thread_key="pitch-1")
await agent.run("Now make it shorter", thread_key="pitch-1")
```

Side-by-side diff against the journaling shape above:

```python
# remove: working_memory + provider + journaling instructions in the prompt.
# - working_memory=InMemoryWorkingMemory(),
# - context_providers=[WorkingMemoryProvider(memory)],
# - "On every turn, journal your draft into <working_memory>..."

# add: a thread store on the agent and a thread_key per run.
# + thread_store=InMemoryThreadStore(),
# + await agent.run("...", thread_key="pitch-1")
```

On the second `run`, the first run's assistant turn is replayed into
the message list as an unwrapped `assistant`-role message before the
new user input. The model sees its own prior work, not external
context. Tool calls and tool results are replayed in the same
structural shape they were produced in, so cross-run tool-use loops
hold together.

`ThreadStore` is a protocol; `InMemoryThreadStore` is the reference
implementation. For durable storage across process restarts, use
`PostgresThreadStore` (also under `nanitics.composition`).

A runnable end-to-end example using `AgentTool` to dispatch the same
specialist twice in one outer run lives at
[`examples/multi_agent/threads_repeated_agent_tool.py`](../../examples/multi_agent/threads_repeated_agent_tool.py).

## Coexistence

`WorkingMemory` and threads are not alternatives. They sit on
different axes and compose:

- `WorkingMemory` carries *information* the agent should know but not
  treat as its own prior turn — provider-injected, wrapped, clearly
  framed as external context.
- `thread_key` carries *behavior* the agent produced — its prior
  assistant turns, tool calls, and tool results, unwrapped, structurally
  framed as its own conversation history.

A single agent can use both at once. The substrate-comparison recipe
in [Memory § Recipe: information continuity vs behavioral continuity, side by side](memory.md#recipe-information-continuity-vs-behavioral-continuity-side-by-side)
shows the two surfaces in one block. The runnable counterpart is
[`examples/memory/working_memory_vs_threads.py`](../../examples/memory/working_memory_vs_threads.py).

If your need is "this fact must be present on every LLM call but the
agent shouldn't believe it authored it" — `WorkingMemory` is still
the right substrate. If your need is "the agent's next `run` should
pick up where the last one left off" — that's threads.
