# Simplified Workflow Test

Testing the simplified Claude Code Review shared workflow (565 → 108 lines).

Key changes:
- Removed redundant gh pr diff/comment instructions (action handles natively)
- Removed retry mechanism, comment tracking, and diagnostics
- Default review focus changed to "correctness, bugs, and security"
- Minimal 3-line auto-review prompt for pull_request events
- issue_comment events handled entirely by the action

Delete after test.
