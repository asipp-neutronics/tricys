# Concurrent Operation & Enhanced Mode

For parameter sweeps involving a large number of simulation tasks, executing them sequentially one by one can be very time-consuming. `tricys` supports concurrent operation (parallel computing) to fully utilize your computer's multiple CPU cores, significantly reducing the total simulation time.

Additionally, TRICYS introduces **Enhanced Mode**, which uses a "Compile Once, Run Many" strategy and efficient data streaming to drastically improve simulation efficiency and stability.

## 1. What is Enhanced Mode?

Enhanced Mode is an optimized execution mode in `tricys` designed for large-scale simulations. It addresses performance bottlenecks and Out-Of-Memory (OOM) issues often encountered when using traditional OpenModelica calls for thousands of simulations.

### Core Features:

1.  **Compile Once, Run Many**:
    *   **Traditional Mode**: Every simulation (even if the model is the same and only parameters differ) re-invokes OMC to compile the model. This is acceptable for small-scale tasks, but in large sweeps, compilation time takes up a huge proportion.
    *   **Enhanced Mode**: The program first compiles the model into a standalone executable (Exe/XML) and then directly calls this executable for all subsequent tasks. This eliminates repetitive compilation overhead, significantly speeding up startup.

2.  **HDF5 Streaming**:
    *   **Traditional Mode**: Each task generates an independent CSV file, which are merged at the end. When tasks number in the tens of thousands, this creates huge disk I/O pressure, and the final merge process can easily lead to memory overflow (OOM).
    *   **Enhanced Mode**: Results from all concurrent tasks are streamed in real-time into a single HDF5 file (`sweep_results.h5`). HDF5 is an efficient binary format capable of storing massive amounts of data without consuming large amounts of memory.

3.  **Process Isolation**:
    *   Each simulation task still runs in an independent process, ensuring that a calculation crash caused by specific parameters does not affect the stability of the main program.

## 2. How to Enable

### 2.1 Quick Enable via CLI (Recommended)

The most **convenient and fast** way to enable Enhanced Mode is to append the `--enhanced` argument directly to the command line. You can enjoy the performance boost of single compilation and HDF5 without modifying the configuration file.

```bash
tricys -c config.json --enhanced
```

This command automatically overrides settings in the configuration file, enforcing Enhanced Mode (including concurrent execution).

### 2.2 Enable via Configuration

You can also explicitly enable `concurrent` in your configuration file:

```json
{
    "simulation": {
        "model_name": "example_model.Cycle",
        ...
        "concurrent": true,
        "max_workers": 4
    }
}
```

When `concurrent` is set to `true`, TRICYS automatically activates Enhanced Mode.

## 3. Configuration Details

### `simulation.concurrent`

- **Description**: Whether to enable concurrent operation (and automatically activate Enhanced Mode).
- **Type**: Boolean (`true` or `false`).
- **Default**: `false`.

### `simulation.max_workers`

- **Description**: Controls the maximum number of processes used for concurrent execution.
- **Type**: Integer (optional).
- **Default**: Smart calculation (see strategy below).

## 4. Smart Concurrency Strategy

`tricys` uses a smart strategy to determine the final number of processes to use, adapting to multi-user shared server scenarios to ensure full performance without inadvertently crowding out other users.

The priority order is as follows:

1.  **Turbo Mode (`--turbo`)**:
    - If you use the `--turbo` flag on the command line, the program will ignore all limits and force the use of **100% of available CPU cores**.
    - **Best for**: Dedicated servers, or when results are needed urgently.

2.  **Configuration Override (`config.json`)**:
    - If you explicitly set `max_workers` in the config file (e.g., `"max_workers": 20`), the program will strictly adhere to that value.
    - **Best for**: Precise control over resource usage.

3.  **Default Safety Limit (Default Safety)**:
    - If Turbo is not enabled and `max_workers` is not configured, the program defaults to using only **50% of available CPU cores**.
    - **Best for**: Daily use on shared servers, leaving resources for other users.

4.  **Task Clamping**:
    - **This is the final hard optimization**. If the calculated worker count exceeds the actual number of simulation tasks, the program will automatically reduce the worker count to match the number of tasks.
    - *Example*: Even if you have a 128-core server but run only 5 simulation tasks, the program will launch only 5 processes, not 128. This significantly reduces system overhead.

## 5. Result File Differences

The result file structure differs slightly when Concurrent/Enhanced Mode is enabled:

- **Traditional Mode**: Generates `sweep_results.csv`.
- **Enhanced Mode**: Generates `sweep_results.h5`.

`tricys` post-processing modules and plotting tools automatically detect and support reading HDF5 formatted results.

!!! success "From CSV to HDF5"
    HDF5 offers orders of magnitude improvement in read/write performance compared to CSV. For results containing millions of rows, CSV parsing might take minutes, while HDF5 takes milliseconds.