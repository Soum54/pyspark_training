# Autonomous Last-Mile Delivery Trip Analytics

**Domain:** Autonomous Logistics / Quick Commerce  
**Difficulty:** Medium  
**Total Marks:** 20  
**Expected Duration:** 20–25 Minutes  
**Objective:** Implement five independent PySpark functions.

## Problem Statement
A quick-commerce platform uses partner-assisted and autonomous vehicles for last-mile delivery trips. Each trip records dispatch, promised and actual timestamps, distance, fare, delivery partner, delay reason, and final status. Operations teams need delay, trip-duration, speed, partner-performance, and delay-reason insights.

This assessment uses proven function patterns from Q216 Food Delivery Delay and the cab/trip-analysis family: timestamp parsing, `unix_timestamp()`, non-negative time calculations, threshold filtering, grouped sum/average/count metrics, deterministic sorting, and tuple extraction.

**Independence Rule:** Every function is evaluated independently. Do not call one trainee function from another trainee function.

## Dataset
**Path:** `data/delivery_trips.csv`

| Column | Description |
|---|---|
| trip_id | Unique trip identifier |
| partner | Delivery partner |
| dispatch_ts | Dispatch timestamp |
| promised_ts | Promised delivery timestamp |
| actual_ts | Actual delivery timestamp |
| distance_km | Trip distance |
| fare_amount | Fare amount |
| delay_reason | Delay reason |
| status | Final trip status |

Timestamp format: `yyyy-MM-dd HH:mm:ss`

---

## Task 1 — Load Delivery Trip Data — 4 Marks
### Function Name
`load_delivery_trips`

### Function Signature
```python
def load_delivery_trips(spark: SparkSession, path: str) -> DataFrame:
```

### Implementation Flow
1. Read CSV with `header=True` and schema inference.
2. Convert `dispatch_ts`, `promised_ts`, and `actual_ts` using `to_timestamp()`.
3. Cast `distance_km` to double.
4. Cast `fare_amount` to double.
5. Return the DataFrame.

---

## Task 2 — Compute Trip Metrics — 4 Marks
### Function Name
`compute_trip_metrics`

### Function Signature
```python
def compute_trip_metrics(df: DataFrame) -> DataFrame:
```

### Required Calculations
Create `delay_hours`:
```text
(unix_timestamp(actual_ts) - unix_timestamp(promised_ts)) / 3600
```
If negative, use `0.0`.

Create `trip_hours`:
```text
(unix_timestamp(actual_ts) - unix_timestamp(dispatch_ts)) / 3600
```
If negative, use `0.0`.

Create `avg_speed_kmph`:
- if `trip_hours > 0`, calculate `distance_km / trip_hours`
- otherwise use `0.0`

Return the updated DataFrame.

---

## Task 3 — Filter Significant Delay Trips — 4 Marks
### Function Name
`filter_significant_delays`

### Function Signature
```python
def filter_significant_delays(
    df: DataFrame,
    delay_threshold: float,
    min_distance: float
) -> DataFrame:
```

### Business Rules
Keep rows where:
- `delay_hours > delay_threshold`
- `distance_km >= min_distance`

Return the filtered DataFrame.

---

## Task 4 — Partner Delay Summary — 4 Marks
### Function Name
`partner_delay_summary`

### Function Signature
```python
def partner_delay_summary(df: DataFrame) -> DataFrame:
```

### Implementation Flow
1. Keep only `status == "Completed"`.
2. Group by `partner`.
3. Calculate:
   - `sum(delay_hours)` as `total_delay_hours`
   - `avg(fare_amount)` as `avg_fare`
   - `count(trip_id)` as `trip_count`
4. Return exactly:
   - `partner`
   - `total_delay_hours`
   - `avg_fare`
   - `trip_count`

---

## Task 5 — Highest Average Delay Reason — 4 Marks
### Function Name
`highest_average_delay_reason`

### Function Signature
```python
def highest_average_delay_reason(df: DataFrame) -> Tuple[str, float]:
```

### Expected Input
The evaluator supplies a DataFrame containing `delay_reason` and `delay_hours` directly. **Do not call `compute_trip_metrics()` inside this function.**

### Implementation Flow
1. Ignore rows where `delay_reason` is null.
2. Ignore rows where `delay_hours` is null.
3. Group by `delay_reason`.
4. Calculate average `delay_hours` as `avg_delay_hours`.
5. Sort `avg_delay_hours` descending and `delay_reason` ascending.
6. Return `(delay_reason, avg_delay_hours)`.
7. If no valid rows exist, return `("", 0.0)`.

## Evaluation Rules
- Exactly five trainee functions and six tests.
- Every test creates its own input; no evaluator test chains trainee functions.
- Correct alternative PySpark solutions are accepted if behavior matches the specification.
- No specific import alias is required in trainee code.
- `print()` and `DataFrame.show()` are allowed for debugging.
- Do not create or stop SparkSession inside assessment functions.
- Do not hardcode expected results.
