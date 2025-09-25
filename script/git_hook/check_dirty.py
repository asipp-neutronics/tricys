import sys
import subprocess

# Check for tracked files that are modified but NOT staged
result = subprocess.run(['git', 'diff', '--name-only'], capture_output=True, text=True)
if result.stdout:
    print("ERROR: Found tracked files with modifications that have not been staged (`git add`).", file=sys.stderr)
    print("Please stage your changes before committing:", file=sys.stderr)
    print(result.stdout.strip(), file=sys.stderr)
    sys.exit(1)

# Check for untracked files
result = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard', '--directory'], capture_output=True, text=True)
if result.stdout:
    print("ERROR: Untracked files detected. Please add them to .gitignore or commit them.", file=sys.stderr)
    print("Untracked files:", file=sys.stderr)
    print(result.stdout.strip(), file=sys.stderr)
    sys.exit(1)

# If no dirty state is found, allow the commit to proceed
sys.exit(0)
