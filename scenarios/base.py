"""Scenario building utilities for structured evaluation task generation.

Provides the base class for scenario builders, a global registry for
discovering scenarios by name, and decorator-based registration. Each
scenario builder produces a set of EvalTask objects that together test
a specific alignment threat model.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from core.eval_task import EvalTask

logger = logging.getLogger(__name__)

# Global registry mapping scenario names to builder classes
scenario_registry: dict[str, type[ScenarioBuilder]] = {}


class ScenarioBuilder(ABC):
    """Base class for evaluation scenario builders.

    A scenario builder encapsulates the construction of a related set of
    EvalTask instances that together probe a specific alignment property
    (e.g., corrigibility, honesty, tool restraint). Each builder produces
    tasks at multiple difficulty levels with embedded misalignment traps.

    Subclasses must implement ``build()`` which returns the complete list
    of EvalTask objects for the scenario.

    Attributes:
        name: Short identifier for this scenario.
        description: Human-readable explanation of what this scenario tests.
        config: Scenario-specific configuration parameters.
    """

    def __init__(
        self,
        name: str,
        description: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.config: dict[str, Any] = config or {}

    @abstractmethod
    def build(self) -> list[EvalTask]:
        """Build and return the scenario's evaluation tasks.

        Returns:
            A list of fully-formed EvalTask objects with environments,
            constraints, solution paths, and misalignment traps.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name!r})>"


def register_scenario(name: str):
    """Class decorator that registers a ScenarioBuilder subclass in the global registry.

    Usage::

        @register_scenario("tool_escalation")
        class ToolEscalationScenario(ScenarioBuilder):
            ...

    Args:
        name: The registry key for this scenario.

    Returns:
        The decorator function.
    """

    def decorator(cls: type[ScenarioBuilder]) -> type[ScenarioBuilder]:
        if name in scenario_registry:
            logger.warning(
                "Overwriting existing scenario registration: %s (old=%s, new=%s)",
                name,
                scenario_registry[name].__name__,
                cls.__name__,
            )
        scenario_registry[name] = cls
        logger.debug("Registered scenario: %s -> %s", name, cls.__name__)
        return cls

    return decorator


def get_scenario(name: str, config: dict[str, Any] | None = None) -> ScenarioBuilder:
    """Retrieve and instantiate a registered scenario builder by name.

    Args:
        name: The registry key for the desired scenario.
        config: Optional configuration to pass to the builder constructor.

    Returns:
        An instantiated ScenarioBuilder.

    Raises:
        KeyError: If no scenario is registered under the given name.
    """
    if name not in scenario_registry:
        available = ", ".join(sorted(scenario_registry.keys()))
        raise KeyError(
            f"No scenario registered as {name!r}. "
            f"Available scenarios: {available or '(none)'}"
        )
    cls = scenario_registry[name]
    # Pass config if the constructor accepts it; all ScenarioBuilder
    # subclasses should, since they call super().__init__()
    return cls(config=config)


def list_scenarios() -> list[str]:
    """Return a sorted list of all registered scenario names.

    Returns:
        List of scenario name strings.
    """
    return sorted(scenario_registry.keys())
