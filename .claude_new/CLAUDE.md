# Multi-Agent Harness

This directory contains a **Manager → Coder → Tester** harness that drives implementation
tasks through a plan file. All task-specific knowledge (rules, build commands, retry
limits, correctness criteria, etc.) lives in `.claude_new/plans/`.

## How it works

| Agent | Role |
|-------|------|
| `manager` | Reads the plan file, parses parameters and retry limit, orchestrates Coder and Tester |
| `coder` | Reads the plan and writes the implementation file |
| `tester` | Reads the plan and runs build + correctness checks |

## How to invoke

Tell Claude:

> "Use the manager agent with plan `.claude_new/plans/<task>.md`"

For example:

> "Use the manager agent with plan `.claude_new/plans/cutlass_conv2d.md`"

Claude will invoke the `manager` agent, which reads the plan (including its retry limit),
spawns the `coder`, then the `tester`, and retries up to the plan-specified limit on failure.

## How to adapt to a new repo

1. Copy this `.claude_new/` directory into the target repository.
2. Write a new `plans/<your_task>.md` following the plan file format described below.
3. Tell Claude: "Use the manager agent with plan `.claude_new/plans/<your_task>.md`"

The agent `.md` files do **not** need to be edited.

## Plan file format

A plan file is a Markdown document with the following sections:

```
## Task description
What the coder should implement (1–3 sentences).

## Parameter schema
The input parameters the manager should parse from the user request.

## Max retries          ← controls the retry loop; omit to use the default of 3
Integer. How many Coder + Tester cycles the manager will run before giving up.

## Toolchain
Compiler, language standard, target architecture, environment setup commands.

## Tensor layout rules (optional)
Any memory layout conventions the coder must follow.

## Implementation rules
Numbered list of rules the coder must follow.

## Function signature
Exact function signature the coder must produce.

## Required headers
Exact `#include` lines required in the output file.

## Type configuration (optional)
Template parameters or type aliases.

## Output file
Exact path where the coder should write the solution.

## Build command
Exact shell command(s) the tester should run to compile.

## Run command
Exact shell command(s) the tester should run to execute the binary.

## Correctness criteria
How the tester determines PASS vs FAIL (exit code, output pattern, tolerance).
```

## Conventions

- Agents must read the plan file before taking any action.
- The coder must not run build or test commands.
- The tester must not modify source files.
- The manager owns the retry loop — neither coder nor tester retries independently.
- The retry limit is set per-plan via the **Max retries** field, not in the agent files.
