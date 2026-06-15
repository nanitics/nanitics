"""Explicit run completion: finish() and the question-vs-finish fork.

Demonstrates ``require_explicit_finish`` on ``ReActAgent``. In this mode a run
ends only via a typed terminal action — ``finish()`` to deliver a result, or
``ask_human()`` to ask a person — so a clarifying question can never silently
end the run as an unanswered, undelivered message.

Section 1 contrasts the two modes on the same scripted "question" turn: in
default mode the bare-text question becomes the output and the run ends; in
explicit mode the same turn is non-terminal and the agent is nudged to act.
Section 2 shows the full fork: the agent asks a person via ``ask_human``,
receives the answer inline, then delivers its result via ``finish``.

Related guide: docs/guides/human-in-the-loop.md
"""

import asyncio

from examples.helpers import make_emitter, make_response
from nanitics.collaboration import (
    CallbackHumanInputProvider,
    HumanDecision,
    HumanInputResponse,
    create_ask_human_tool,
)
from nanitics.infrastructure import MockLLMClient
from nanitics.strategies import ReActAgent
from nanitics.tracing import ToolCall


def _finish(result: str, call_id: str = "f1") -> ToolCall:
    return ToolCall(id=call_id, name="finish", arguments={"result": result})


async def main() -> None:
    # --- Section 1: A bare-text question is terminal in default mode ---
    print("--- Section 1: default mode — the question silently ends the run ---")

    question = "Should I deploy to prod or staging?"
    default_agent = MockLLMClient([make_response(question)])
    agent = ReActAgent(
        name="default",
        llm_client=default_agent,
        emitter=make_emitter(),
        system_prompt="You are a deploy assistant.",
        tools=[],
    )
    result = await agent.run("Deploy the app.")
    assert result.termination_reason == "complete"
    assert result.output == question  # the question leaked out as the "answer"
    print(f"  termination_reason={result.termination_reason!r}; output is the unanswered question")

    # --- Section 1b: the same turn is non-terminal under explicit finish ---
    print("\n--- Section 1b: explicit mode — the question is nudged, then finished ---")

    client = MockLLMClient(
        [
            make_response(question),  # bare-text question: non-terminal, nudged
            make_response("Deploying to staging.", tool_calls=[_finish("Deployed to staging.")]),
        ]
    )
    agent = ReActAgent(
        name="explicit",
        llm_client=client,
        emitter=make_emitter(),
        system_prompt="You are a deploy assistant.",
        tools=[],
        require_explicit_finish=True,
    )
    result = await agent.run("Deploy the app.")
    assert result.termination_reason == "finished"
    assert result.output == "Deployed to staging."
    nudges = [m for m in result.messages if m.role == "user" and "finishing" in (m.content or "")]
    assert nudges, "a nudge was injected after the bare-text turn"
    print(f"  termination_reason={result.termination_reason!r}; output={result.output!r}")
    print(f"  nudge injected: {nudges[0].content!r}")

    # --- Section 2: the question-vs-finish fork with a real human channel ---
    print("\n--- Section 2: ask_human (question) then finish (answer) ---")

    provider = CallbackHumanInputProvider(
        lambda req: HumanInputResponse(request_id=req.request_id, decision=HumanDecision.ANSWER, content="staging")
    )
    ask_human = create_ask_human_tool(provider)

    client = MockLLMClient(
        [
            # The agent asks a person rather than assuming. The callback answers
            # inline ("staging"), so the run continues with that answer.
            make_response(
                "I need to know the target.",
                tool_calls=[ToolCall(id="q1", name="ask_human", arguments={"question": "prod or staging?"})],
            ),
            make_response("Deploying.", tool_calls=[_finish("Deployed to staging.", call_id="f2")]),
        ]
    )
    agent = ReActAgent(
        name="forked",
        llm_client=client,
        emitter=make_emitter(),
        system_prompt="You are a deploy assistant.",
        tools=[ask_human],
        require_explicit_finish=True,
        run_id="explicit-finish-example",
    )
    result = await agent.run("Deploy the app.")
    assert result.termination_reason == "finished"
    assert result.output == "Deployed to staging."
    answers = [m for m in result.messages if m.role == "tool_result" and "staging" in (m.content or "")]
    assert answers, "the human's answer was threaded back into the conversation"
    print(f"  asked the human, got {answers[0].content!r}, then finished: {result.output!r}")


if __name__ == "__main__":
    asyncio.run(main())
