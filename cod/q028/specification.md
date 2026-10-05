# EV Battery Swap Consumption Insights

**Domain:** Smart Mobility / EV Infrastructure  
**Difficulty:** Easy  
**Total Marks:** 10  
**Expected Duration:** 10–15 Minutes  
**Objective:** Implement five independent PySpark functions.

## Problem Statement
An EV battery-swap network records battery swap events across multiple stations. The operations team needs monthly energy-consumption insights by battery chemistry and a simple view of the highest-consumption battery type.

This assessment uses proven function patterns from Q211 Public Library and Q214 PySpark Farm: CSV loading, date conversion, month derivation, validity filtering, grouped aggregation, deterministic sorting, and tuple extraction.

**Independence Rule:** Every function is evaluated independently. Do not call one trainee function from another trainee function.

## Dataset
**Path:** `data/battery_swaps.csv`

| Column | Description |
|---|---|
| swap_id | Unique battery-swap identifier |
| station_id | Swap station identifier |
| battery_type | Battery chemistry/type |
| energy_kwh | Energy handled during the swap |
| swap_date | Swap date |
| status | Swap processing status |

---

## Task 1 — Load Battery Swap Data — 2 Marks
### Function Name
`load_battery_swap_data`

### Function Signature
```python
def load_battery_swap_data(spark: SparkSession, path: str) -> DataFrame:
```

### Parameters
- `spark: SparkSession`
- `path: str`

### Returns
- `DataFrame`

### Description
Load the battery-swap CSV and convert `swap_date` to DateType.

### Implementation Flow
1. Read the CSV using `header=True`.
2. Enable schema inference.
3. Convert `swap_date` using `to_date()`.
4. Return the DataFrame.

### Expected Columns
`swap_id | station_id | battery_type | energy_kwh | swap_date | status`

---

## Task 2 — Add Swap Month Bucket — 2 Marks
### Function Name
`add_swap_month`

### Function Signature
```python
def add_swap_month(df: DataFrame) -> DataFrame:
```

### Description
Add `swap_month`, representing the beginning of the month containing `swap_date`.

### Implementation Flow
Use `date_trunc("month", col("swap_date"))` and store the result in `swap_month`.

---

## Task 3 — Filter Valid Swap Records — 2 Marks
### Function Name
`filter_valid_swaps`

### Function Signature
```python
def filter_valid_swaps(df: DataFrame) -> DataFrame:
```

### Business Rules
Keep only rows where:
- `energy_kwh >= 0`
- `battery_type` is not null

Return the filtered DataFrame.

---

## Task 4 — Monthly Energy by Battery Type — 2 Marks
### Function Name
`monthly_battery_energy`

### Function Signature
```python
def monthly_battery_energy(df: DataFrame) -> DataFrame:
```

### Implementation Flow
1. Group by `battery_type` and `swap_month`.
2. Sum `energy_kwh`.
3. Alias the result as `total_energy_kwh`.
4. Return exactly these columns:
   - `battery_type`
   - `swap_month`
   - `total_energy_kwh`

---

## Task 5 — Top Battery Type — 2 Marks
### Function Name
`top_battery_type`

### Function Signature
```python
def top_battery_type(df: DataFrame) -> Tuple[str, float]:
```

### Expected Input
The evaluator supplies a DataFrame containing at least `battery_type` and `total_energy_kwh` directly. **Do not call `monthly_battery_energy()` inside this function.**

### Implementation Flow
1. Group by `battery_type`.
2. Sum `total_energy_kwh` across all input rows.
3. Sort total energy descending.
4. Use `battery_type` ascending as the deterministic tie-breaker.
5. Return `(battery_type, total_energy)`.
6. If no rows exist, return `("", 0.0)`.

## Evaluation Rules
- Exactly five trainee functions and six tests.
- Each test creates/evaluates its own inputs; tests do not chain trainee functions.
- Correct alternative PySpark solutions are accepted if behavior matches the specification.
- No specific import alias such as `functions as F` is required in trainee code.
- `print()` and `DataFrame.show()` are allowed for debugging.
- Do not create or stop SparkSession inside assessment functions.
- Do not use Python file I/O to bypass Spark CSV loading.
- Do not hardcode expected results.
