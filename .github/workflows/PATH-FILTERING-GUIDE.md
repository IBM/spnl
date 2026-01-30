# GitHub Workflow Path Filtering Guide

## Problem

GitHub's native `paths:` and `paths-ignore:` filters in workflow triggers are incompatible with branch protection rules. When these filters exclude a workflow from running, the required status check never completes, blocking PRs from being merged.

## Solution

Instead of using `paths:` or `paths-ignore:` in the workflow trigger, we implement the filtering logic as a job within the workflow. This ensures the workflow always runs and reports a status, but expensive jobs are conditionally skipped when irrelevant files change.

## Implementation Pattern

### For workflows with `paths-ignore:` (e.g., python.yml, core.yml, publish_pypi.yml)

```yaml
on:
  pull_request:
    branches: [ main ]
    # Remove paths-ignore from here

jobs:
  check_changes:
    name: Check if relevant files changed
    runs-on: ubuntu-latest
    outputs:
      should_run: ${{ steps.check.outputs.should_run }}
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0
      - name: Check for relevant changes
        id: check
        run: |
          if [ "${{ github.event_name }}" = "push" ]; then
            echo "should_run=true" >> $GITHUB_OUTPUT
          else
            git fetch origin ${{ github.base_ref }}
            CHANGED_FILES=$(git diff --name-only origin/${{ github.base_ref }}...HEAD)
            
            # Filter out ignored patterns
            RELEVANT_FILES=$(echo "$CHANGED_FILES" | grep -v '\.md$' | grep -v '^docker/' || true)
            
            if [ -z "$RELEVANT_FILES" ]; then
              echo "should_run=false" >> $GITHUB_OUTPUT
            else
              echo "should_run=true" >> $GITHUB_OUTPUT
            fi
          fi

  build_and_test:
    needs: check_changes
    if: needs.check_changes.outputs.should_run == 'true'
    runs-on: ubuntu-latest
    # ... rest of job
```

### For workflows with `paths:` (e.g., cli-image-build.yml, llm-d-image-build.yml)

```yaml
on:
  pull_request:
    branches: [ main ]
    # Remove paths from here

jobs:
  check_changes:
    name: Check if relevant files changed
    runs-on: ubuntu-latest
    outputs:
      should_run: ${{ steps.check.outputs.should_run }}
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 0
      - name: Check for relevant changes
        id: check
        run: |
          if [ "${{ github.event_name }}" = "release" ]; then
            echo "should_run=true" >> $GITHUB_OUTPUT
          else
            git fetch origin ${{ github.base_ref }}
            CHANGED_FILES=$(git diff --name-only origin/${{ github.base_ref }}...HEAD)
            
            # Check if files match include patterns
            RELEVANT_FILES=$(echo "$CHANGED_FILES" | grep -E '^docker/vllm/llmd/|^\.github/workflows/.*\.yml$' || true)
            
            if [ -z "$RELEVANT_FILES" ]; then
              echo "should_run=false" >> $GITHUB_OUTPUT
            else
              echo "should_run=true" >> $GITHUB_OUTPUT
            fi
          fi

  build_and_push:
    needs: check_changes
    if: needs.check_changes.outputs.should_run == 'true'
    runs-on: ubuntu-latest
    # ... rest of job
```

## Key Benefits

1. **Branch Protection Compatible**: The workflow always runs and reports a status, satisfying required status checks
2. **Cost Efficient**: Expensive jobs are skipped when irrelevant files change
3. **No External Dependencies**: Uses only bash and git, no third-party actions required
4. **Transparent**: Clear logging shows why jobs were skipped or run

## Modified Workflows

The following workflows have been updated to use this pattern:

- `python.yml` - Ignores `**/*.md` and `docker/**`
- `core.yml` - Ignores `**/*.md` and `docker/**`
- `publish_pypi.yml` - Ignores `**/*.md` and `docker/**`
- `cli-image-build.yml` - Only runs for `docker/vllm/llmd/**` and workflow files
- `llm-d-image-build.yml` - Only runs for `docker/vllm/llmd/**` and workflow file

## Testing

To test the implementation:

1. Create a PR that only changes markdown files
2. Verify the `check_changes` job runs and sets `should_run=false`
3. Verify dependent jobs are skipped
4. Verify the workflow reports success (green checkmark)
5. Verify branch protection allows the PR to merge

## Notes

- The `check_changes` job always succeeds, even when it determines jobs should be skipped
- For push events to main, workflows always run (no filtering)
- For release events, workflows always run (no filtering)
- The `fetch-depth: 0` ensures we have full git history for accurate diffs