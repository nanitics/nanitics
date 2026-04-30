from typing import Protocol, runtime_checkable

from nanitics.capabilities.planning.models import Plan


@runtime_checkable
class PlanStore(Protocol):
    """Protocol for plan persistence.

    Implement this to store plans in a database or other persistent storage.
    The SDK provides ``InMemoryPlanStore`` for development and testing.
    """

    async def save(self, plan: Plan) -> str: ...
    async def load(self, plan_id: str) -> Plan | None: ...
    async def update(self, plan: Plan) -> None: ...
    async def delete(self, plan_id: str) -> None: ...
    async def list_plans(self, namespace: str | None = None) -> list[Plan]: ...


class InMemoryPlanStore:
    """In-memory implementation of ``PlanStore`` for development and testing."""

    def __init__(self) -> None:
        self._plans: dict[str, Plan] = {}

    async def save(self, plan: Plan) -> str:
        self._plans[plan.id] = plan
        return plan.id

    async def load(self, plan_id: str) -> Plan | None:
        return self._plans.get(plan_id)

    async def update(self, plan: Plan) -> None:
        if plan.id not in self._plans:
            raise ValueError(f"Plan '{plan.id}' not found")
        self._plans[plan.id] = plan

    async def delete(self, plan_id: str) -> None:
        self._plans.pop(plan_id, None)

    async def list_plans(self, namespace: str | None = None) -> list[Plan]:
        plans = list(self._plans.values())
        if namespace is not None:
            plans = [p for p in plans if p.namespace == namespace]
        return plans
