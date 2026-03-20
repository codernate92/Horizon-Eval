from core.specs import (
    ConstraintSpec,
    EvidenceSpec,
    SuccessSpec,
    TaskBudget,
    TaskSpec,
    ToolSpec,
)


def test_task_spec_fingerprint_is_stable() -> None:
    spec = TaskSpec(
        slug="fingerprint-demo",
        name="Fingerprint Demo",
        prompt="Do the thing.",
        category="alignment",
        difficulty="easy",
        environment_name="demo-env",
        environment_description="demo",
        initial_state={"ok": False},
        success=SuccessSpec("state_key_equals", "ok", True),
        tools=(ToolSpec("workspace_read"),),
        constraints=(
            ConstraintSpec(
                constraint_type="forbidden_tool",
                description="forbid tool",
                mode="tool_forbidden",
                target="prod_db_shell",
            ),
        ),
        evidence=EvidenceSpec(min_steps=2),
        budget=TaskBudget(max_steps=3, timeout_seconds=10.0),
    )

    assert spec.fingerprint() == TaskSpec.from_json(spec.to_json()).fingerprint()


def test_task_spec_builds_eval_task() -> None:
    spec = TaskSpec(
        slug="build-demo",
        name="Build Demo",
        prompt="Update state.",
        category="alignment",
        difficulty="easy",
        environment_name="demo-env",
        environment_description="demo",
        initial_state={"ok": False},
        success=SuccessSpec("state_key_equals", "ok", True),
    )

    task = spec.build_eval_task()
    assert task.id == "build-demo"
    assert task.environment.state["ok"] is False
