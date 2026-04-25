<!-- BEGIN robot-md — auto-generated; edit ROBOT.md then re-run `robot-md claude-md` to refresh -->
# CLAUDE.md — omx

> **Agent context file.** Drop this in the root of your robot's project (same directory as `ROBOT.md`) and Claude Code will read it at the start of every session.
>
> Template from [robot-md](https://github.com/RobotRegistryFoundation/robot-md) — customize everywhere you see `{{...}}` or `TODO`.

This project is a **robot workspace**. The file `ROBOT.md` in this directory is the authoritative declaration of what the robot is and what it can do (identity, physics, drivers, capabilities, safety gates). Consult it before answering any question about the robot.

## Recognizing robot-related intent

When the operator asks any of the following, **you should act** — not ask clarifying questions first:

| Operator intent (examples) | What to do |
|---|---|
| "What can this robot do?" / "What are its capabilities?" | Read `robot-md://omx/capabilities` (MCP) or run `robot-md render ROBOT.md` and extract `capabilities[]`. |
| "What are its safety gates?" / "What's dangerous?" | Read `robot-md://omx/safety` (MCP) or `robot-md render ROBOT.md` and extract `safety.hitl_gates`, `safety.estop`. |
| "Something's wrong" / "It's not responding" / "Why is X broken" | Run `robot-md doctor --path ROBOT.md`. Report each non-pass check. |
| "Is the manifest valid?" / "Did I break something?" | Run `robot-md validate ROBOT.md`. |
| "Pose the arm at zero" / "Calibrate" | `robot-md calibrate --zero ROBOT.md`. Relay the interactive prompts to the operator. |
| "Publish my robot" / "Give it a public URL" | `robot-md publish-discovery ROBOT.md --url <URL>` writes `.well-known/robot-md.json`. |
| "Pick up the X" / any physical motion | Call `mcp__robot-md-omx__execute_capability` (dry-run first). Check `safety.hitl_gates` for the cap's scope; if a gate with `require_auth: true` matches, request explicit operator approval before re-running without dry-run. |

## Tooling available in this workspace

```bash
robot-md --help              # full verb list
robot-md doctor              # diagnose install + manifest + drivers
robot-md validate ROBOT.md   # schema conformance
robot-md render ROBOT.md     # frontmatter → pure YAML (for parsing)
robot-md context ROBOT.md    # Claude-ready context block (if no MCP)
robot-md publish-discovery ROBOT.md --url <url>   # emit .well-known/robot-md.json
```

If `robot-md-mcp` is registered in this session (check with `/mcp`), prefer the MCP resources over shelling out — they stay in sync with the file on disk automatically:

- `robot-md://omx/frontmatter` — full parsed YAML
- `robot-md://omx/capabilities` — capabilities list
- `robot-md://omx/safety` — safety block
- `robot-md://omx/body` — prose

MCP tools (also available): `validate`, `render`, `estop`, `estop_clear`, `execute_capability`, `execute_task`.

## Safety posture

**Never actuate the robot without consulting `ROBOT.md:safety` first.** If the operator asks for a motion that matches any `hitl_gate.scope`, pause and request explicit authorization. If the manifest declares `safety.estop.software: true`, a software e-stop is available — learn the exact driver command for it before attempting any motion.

Declared gates for this robot:

- `destructive` — requires explicit authorization

## What this robot is running on

- Primary driver: dynamixel @ /dev/ttyUSB0
- Registered RRN: (unregistered)

## Conventions for this project

- **Do not** edit `ROBOT.md` fields under `metadata.*` without asking — those are bound to the registry entry and changing them creates drift.
- **Do not** commit `~/.robot-md/keys/*` — API keys live outside the project.
- When adding a new capability, also update the prose body's "What this robot can do" section so the description and the declaration stay aligned.

## Escalation

If the operator asks for something that *could* harm the robot, a human, or the workspace — and no matching HITL gate is declared — surface that gap explicitly. Don't silently proceed; don't silently decline. Tell the operator: "Your manifest doesn't have a gate for this scope; add one or authorize this specific action."

---

*Last updated: 2026-04-25. Keep this file short — Claude reads it every session.*
<!-- END robot-md -->
