# CLAUDE.md - AI Assistant Context & Guidelines

This file contains project-specific context, conventions, and lessons learned for AI assistants working on this repository.

## Git Workflow & Commit Conventions

### Commit Message Format
- **Always use Conventional Commits format**: `type(scope): description`
  - Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, etc.
  - Example: `feat(vllm): add GCE deployment support`

### Branching Strategy
- **First commit**: Always start a fresh branch off `main`
  - Command: `git checkout main && git pull && git checkout -b feature/your-branch-name`
- **Subsequent commits**: Always amend the previous commit
  - Command: `git commit --amend -sS`
  - Update the commit message if something meaningful has changed

### Commit Signing
- **Always use `-sS` flags when committing**
  - `-s`: Add Signed-off-by line
  - `-S`: GPG sign the commit
  - Example: `git commit -sS -m "feat: add new feature"`
  - For amend: `git commit --amend -sS`

### Pull Request Guidelines
- **Always add the "made with bob" label** to PRs
  - This can be done via GitHub CLI: `gh pr create --label "made with bob"`
  - Or manually add the label after PR creation
- **Update PR description when pushing updates** if something meaningful has changed
  - Use: `gh pr edit <pr-number> --body "updated description"`
  - Keep the PR description in sync with the actual changes

## Project Structure
- Rust workspace with multiple crates
- Main crate: `spnl/`
- CLI tool: `cli/`
- Web interface: `web/`
- GitHub workflows in `.github/workflows/`

## Development Notes
- Pre-commit hooks configured via cargo-husky
- Multiple CI/CD workflows for different components (core, CLI, images)
- Homebrew formula maintained in `Formula/`