In the simulation and analysis of the tritium fuel cycle, evaluating the system's economic efficiency and sustainability is crucial. `tricys` has a built-in series of Key Performance Indicators (KPIs) used to quantitatively assess the key characteristics of the system from dynamic simulation results.


## 1. Startup Inventory

- **Calculation Function**: `calculate_startup_inventory`
- **Physical Meaning**: Startup inventory refers to the amount of tritium that must be pre-loaded at the beginning of a reactor's operation to sustain continuous operation until the system achieves tritium self-sufficiency (i.e., the tritium breeding rate equals or exceeds its consumption rate). On a time-varying curve of tritium inventory, this is represented by the vertical distance from the initial inventory down to its lowest point (the turning point).
- **Calculation Method**:
    1.  In the simulation time series, find the lowest point of the tritium inventory (the Turning Point), which typically represents the moment the system transitions from net consumption to net breeding.
    2.  The startup inventory is calculated as the difference between the **Initial Inventory** and the **Minimum Inventory**.
    
    `Startup Inventory = Initial Inventory - Minimum Inventory`
- **Interpretation**: Startup inventory is a key economic indicator. A **lower startup inventory** means the system can achieve self-sufficiency faster, reducing reliance on external tritium sources and upfront investment costs, making it a primary optimization target in tritium fuel cycle system design.

## 2. Self-Sufficiency Time

- **Calculation Function**: `time_of_turning_point`
- **Physical Meaning**: Self-sufficiency time is the time from the start of the system's operation until its total tritium inventory reaches its lowest point and begins to increase. This point in time marks when the system's internal tritium breeding rate starts to exceed the rates of consumption and loss, and the system theoretically becomes "self-sufficient."
- **Calculation Method**:
    1.  The function first smooths the inventory data to eliminate noise and determine if a true "turning point" exists.
    2.  If the inventory curve is monotonically decreasing (meaning self-sufficiency is not achieved within the simulation time), it returns `NaN` (Not a Number).
    3.  Otherwise, it returns the exact time point from the original (unsmoothed) data where the inventory reaches its minimum value.
- **Interpretation**: The self-sufficiency time directly reflects the speed at which the system reaches tritium balance. A **shorter self-sufficiency time** is ideal as it indicates the system can more quickly move past consuming the startup inventory and begin accumulating tritium, which is crucial for the rapid commissioning of commercial fusion power plants.

## 3. Doubling Time

- **Calculation Function**: `calculate_doubling_time`
- **Physical Meaning**: Doubling time is the time required for the total tritium inventory in the system to double (i.e., reach twice the initial inventory) after achieving tritium self-sufficiency.
- **Calculation Method**:
    1.  First, determine the system's "turning point" (the point of minimum inventory).
    2.  Then, in the inventory data after the turning point, find the first time point where the inventory is greater than or equal to **twice the initial inventory**.
    3.  The doubling time is the difference between this time point and the start of the simulation. If the inventory never reaches twice the initial value, it returns `NaN`.
- **Interpretation**: Doubling time is a core metric for measuring the "profitability" of a tritium breeding system. A finite and reasonable doubling time means the fusion power plant can not only sustain itself but also provide additional tritium fuel to start new plants. This is key to achieving the large-scale development of fusion energy.

## 4. Constraint-Solving Metrics (e.g., Required_TBR)

- **Calculation Method**: `bisection_search` (implemented in `tricys.simulation.simulation_analysis`)
- **Physical Meaning**: In many design studies, the concern is not how the system performs at a fixed Tritium Breeding Ratio (TBR), but the reverse: **to achieve a specific engineering goal (e.g., a doubling time of less than 10 years), what is the minimum required TBR?**
- **Implementation**:
    -   `Required_TBR` is not a directly calculated metric but a **constraint-solving task**.
    -   When you include it in `dependent_variables`, `tricys` enables an optimization algorithm (like [`bisection_search`](../../guides/tricys_analysis/multi_parameter_sensitivity_analysis.md#2-advanced-usage-goal-seeking-analysis)).
    -   This algorithm iteratively runs simulations within a given `search_range`, using a `parameter_to_optimize` (usually the `TBR` parameter in the model) as the variable.
    -   In each iteration, it checks if a key performance indicator (e.g., `Doubling_Time`) meets a predefined constraint (e.g., is less than a certain `metric_max_value`).
    -   Ultimately, the algorithm converges and outputs the **minimum TBR value** that satisfies the constraint.
- **Interpretation**: This "inverse" solving capability is extremely powerful. It transforms the design problem from "forward validation" to "inverse optimization," helping engineers quickly determine the minimum design requirements to achieve key performance goals, thereby greatly accelerating the design iteration process.
