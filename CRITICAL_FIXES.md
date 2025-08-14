# 🚨 CRITICAL FIXES REQUIRED - PR #14

## ⚠️ Blocking Issues (Must Fix Before Merge)

### 1. Thread Safety Bug in `tricys/simulation.py`

**Problem**: Global timestamp variable causes race conditions in multi-threaded execution.

**Current Code** (Lines 21, 67, 178):
```python
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Global variable - UNSAFE

def setup_logging(config: Dict[str, Any]):
    global timestamp  # Modifying global state in threaded context
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def _run_single_job(...):
    result_filename = f"{timestamp}_simulation_results_{job_id}.csv"  # Race condition
```

**Required Fix**:
```python
def _run_single_job(...):
    # Generate timestamp locally for each job
    job_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"{job_timestamp}_simulation_results_{job_id}.csv"
```

### 2. Resource Leak in `tricys/simulation.py`

**Problem**: Variables may not be initialized when exception occurs, causing UnboundLocalError.

**Current Code** (Lines 213-220):
```python
def _run_single_job(...):
    try:
        omc = get_om_session()  # May fail here
        mod = ModelicaSystem(...)  # May fail here
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        return ""
    finally:
        if mod is not None:  # UnboundLocalError if mod wasn't assigned
            del mod
        if omc is not None:  # UnboundLocalError if omc wasn't assigned
            omc.sendExpression("quit()")
```

**Required Fix**:
```python
def _run_single_job(...):
    omc = None
    mod = None
    try:
        omc = get_om_session()
        mod = ModelicaSystem(...)
        # ... rest of code
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        return ""
    finally:
        if mod is not None:
            del mod
        if omc is not None:
            omc.sendExpression("quit()")
```

### 3. Missing Error Handling in `tricys/utils/db_utils.py`

**Problem**: Database connection errors are not handled.

**Current Code** (Line 99):
```python
def get_parameters_from_db(db_path: str) -> List[Dict[str, Any]]:
    with sqlite3.connect(db_path) as conn:  # May fail - no handling
        cursor = conn.cursor()
```

**Required Fix**:
```python
def get_parameters_from_db(db_path: str) -> List[Dict[str, Any]]:
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # ... rest of code
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Database file not found: {db_path}")
        raise
```

## 📝 Documentation Fixes

### 4. Fix README.md Duplicate Content

**File**: `README.md`, lines 5-6
**Problem**: Project description is repeated word-for-word.

**Required Action**: Remove the duplicate sentence.

### 5. Create Missing config.json

**Problem**: Code references `config.json` as default but file doesn't exist.

**Required Action**: Copy `example/example_config.json` to root as `config.json`.

### 6. Fix pyproject.toml Package Discovery

**Current Code**:
```toml
[tool.setuptools]
packages = ["tricys","tricys.utils"]  # Missing subpackages
```

**Required Fix**:
```toml
[tool.setuptools]
packages = ["tricys", "tricys.utils", "tricys.analysis", "tricys.manager", "tricys.simulation"]
```

## 🧪 Missing Test Files

**Problem**: Git diff shows these test files changed but they don't exist:
- `test/test_single_simulation.py`
- `test/test_sweep_simulation.py`

**Required Action**: Create these test files or remove them from the commit.

## ⚡ Quick Fix Commands

Run these in the repository root:

```bash
# 1. Create missing config file
cp example/example_config.json config.json

# 2. Fix README duplicate content (manual edit required)
# Remove duplicate sentence in lines 5-6

# 3. Create missing test files or clean up git history
touch test/test_single_simulation.py test/test_sweep_simulation.py
# OR git rm test/test_single_simulation.py test/test_sweep_simulation.py

# 4. Test the fixes
make test
```

## 🔄 Next Steps

1. **Address these blocking issues first**
2. **Test thoroughly with both single and parallel simulations**
3. **Verify GUI functionality**
4. **Run full test suite**
5. **Update PR description with changes made**

**Status**: 🔴 **BLOCKING** - These issues must be resolved before merge.
