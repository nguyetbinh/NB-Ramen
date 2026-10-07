# Codex Worker Operations — Launcher Only

Operational reference for the launcher. Do not include this file or  
`orchestrator.md` in worker task packets. Workers still follow applicable  
`AGENTS.md` instructions.

## When to delegate

- Follow the active session's delegation rules and existing owner authorization.  
This reference does not itself authorize spawning workers.
- Handle small, bounded tasks directly. Delegate when an independent subtask or  
review materially helps complete the user's request.
- The user specifies the coding/research and review models, plus reasoning effort,  
in the active session. Apply those assignments; this document sets no defaults.  
Do not switch models or effort based on perceived task difficulty. If an assigned  
configuration is unavailable, report it and ask for a replacement.

## Task packet

Give each worker only what it needs:

- one objective and concrete completion criteria;
- relevant paths/excerpts and concise verified state;
- permitted edits, execution scope and existing authorization;
- dependencies, shared-file ownership and any material limits;
- the expected result: changes, evidence, unresolved issues and claim limits.

Do not send broad plan history or old session transcripts by default. Workers may  
read additional relevant files when needed. Context minimization must not suppress  
repository instructions or evidence needed to answer the question.

## Launch and permissions

Prefer the session's worker tools when available. For a separate Codex CLI worker,  
check the installed CLI's supported options before constructing the invocation.

- Set the exact working directory and pass the bounded task packet.
- Inherit the active sandbox and approval constraints. Do not default to  
`danger-full-access`, disable approvals or use another launcher to bypass them.
- Keep final output and diagnostic logs in a task-specific workspace cache path.  
For headless execution, close stdin and use tool-managed background execution.
- Avoid overlapping edits. Assign distinct files or sequence dependent changes;  
inspect the resulting diff before integration.

## Completion and recovery

- Use completion notifications or bounded waits; do not continuously tail logs.  
Keep the user informed during long work.
- Read the concise result first, then inspect the changed files and evidence needed  
to assess it. Read detailed logs when diagnosing failures or missing evidence.
- Continue a worker when its context remains useful. Start a fresh worker when the  
session is unusable or its context is misleading; pass verified state and the  
remaining task, without replaying the entire history.
- Diagnose repeated failures before retrying. Report technical failures separately  
from quality results, including completed/attempted counts where relevant.

## Review and execution

- For risky code, scope independent review to changed files, acceptance criteria  
and concrete correctness or measurement issues, subject to session delegation  
rules. Review must not invent a new product scope or architecture task.
- Ordinary documentation edits do not need an automatic reviewer loop. A clean  
review closes that step unless subsequent changes introduce a relevant issue.
- Run checks only within the user's request and session instructions. Report what  
actually ran; do not silently launch downloads, inference or paid work via tests.
- The launcher coordinates long inference/training, device jobs and servers so  
resource use, logs, progress and recovery have a clear owner. Workers may prepare  
bounded implementations or diagnostics within their assigned scope.
- Complete authorized batch workloads and save outputs for the agreed final review.  
Do not insert per-model or per-variant listening/approval gates unless requested.  
Isolate independent failures and continue the remaining authorized items when  
feasible; never present missing outputs as completed results.
- Ask only for genuinely missing authorization required by the active task or  
applicable instructions. Honor permission already given in the session.

## Handoff

State what changed, what completed, where the evidence is, and what remains.  
Distinguish implementation, measured results and user review. Finish the assigned  
scope; do not add workers, benchmarks or optimization work after it is answered.