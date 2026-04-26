# Generic Multi-Agent Harness

This directory contains a **repo-agnostic** Manager → Coder → Tester multi-agent harness.
All repo-specific knowledge lives in `.claude_new/plans/`. The agent files in
`.claude_new/agents/` contain no repository-specific rules and can be copied verbatim
into any project.

## How it works

The harness uses three agents:

| Agent | Role |
|-------|------|
| `manager` | Parses the plan file, orchestrates Coder and Tester, drives the retry loop |
| `coder` | Reads the plan and writes the implementation file |
| `tester` | Reads the plan and runs build + correctness checks |

All repo-specific details (function signatures, build commands, correctness criteria,
tensor layout rules, etc.) are encoded in a **plan file** under `.claude_new/plans/`.

## How to invoke

Tell Claude:

> "Use the manager agent with plan `.claude_new/plans/<task>.md`"

For example, to run the CUTLASS conv2d pipeline in this repo:

> "Use the manager agent with plan `.claude_new/plans/cutlass_conv2d.md`"

Claude will invoke the `manager` agent, which will read the plan, spawn the `coder`,
then spawn the `tester`, and retry up to 3 times on failure.

## How to adapt to a new repo

1. Copy this `.claude_new/` directory into the target repository.
2. Write a new `plans/<your_task>.md` following the plan file format described below.
3. Tell Claude: "Use the manager agent with plan `.claude_new/plans/<your_task>.md`"

The agent `.md` files do **not** need to be edited.

## Plan file format

A plan file is a Markdown document with the following sections (all required):

```
## Task description
What the coder should implement (1–3 sentences).

## Parameter schema
The input parameters the manager should parse from the user request.

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

## Generic coding conventions

- Agents must read the plan file before taking any action.
- The coder must not run build or test commands.
- The tester must not modify source files.
- The manager drives all retries; neither coder nor tester retries on its own.
- Maximum 3 coder/tester cycles per manager invocation.
