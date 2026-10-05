import builtins
import os
from datetime import datetime
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, DateType, TimestampType
)
from pyspark.sql.window import Window

import solution

ASSESSMENT_NAME = 'EV Battery Swap Consumption Insights'
ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "test_report.log"
STRICT_QA = os.environ.get("LEARNLYTICA_STRICT_QA", "0") == "1"

def _write_header_once():
    LOG_FILE.write_text(
        f"=== {ASSESSMENT_NAME} Test Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n",
        encoding="utf-8"
    )

def _log(no, name, status, expected=None, actual=None, reason=None, fix=None):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        if status == "PASS":
            f.write(f"[PASS] Visible Test Case {no} Passed: {name}\n")
        else:
            f.write(f"[FAIL] Visible Test Case {no} Failed: {name}\n")
            f.write(f"Expected : {expected or 'Required behavior'}\n")
            f.write(f"Actual   : {actual or 'Function did not satisfy the required behavior'}\n")
            f.write(f"Reason   : {reason or 'Validation failed'}\n")
            f.write(f"Fix      : {fix or 'Review the function against the task specification and rerun.'}\n")

def _run(no, name, expected, fn):
    try:
        fn()
        _log(no, name, "PASS")
    except Exception as e:
        actual = str(e).replace("\n", " ").strip() or e.__class__.__name__
        _log(
            no, name, "FAIL", expected=expected, actual=actual,
            reason=f"{e.__class__.__name__}: {actual}",
            fix="Review the function logic, return datatype, Spark column expressions, and task requirements."
        )
        if STRICT_QA:
            raise

_write_header_once()

def test_01_load_battery_swap_data(spark):
    def body():
        df = solution.load_battery_swap_data(spark, str(ROOT / "data" / "battery_swaps.csv"))
        assert df.count() == 6, "Expected 6 battery-swap rows"
        assert dict(df.dtypes)["swap_date"] == "date", "swap_date must be DateType"
    _run(1, "load_battery_swap_data", "6 rows and swap_date as DateType", body)

def test_02_add_swap_month(spark):
    def body():
        df = spark.createDataFrame([("S1", "2026-03-17")], ["swap_id", "d"]).withColumn("swap_date", F.to_date("d")).drop("d")
        row = solution.add_swap_month(df).select("swap_month").first()
        assert row is not None, "Expected a result row"
        assert str(row["swap_month"]).startswith("2026-03-01"), f"Unexpected swap_month {row['swap_month']}"
    _run(2, "add_swap_month", "swap_month representing the first day of March 2026", body)

def test_03_filter_valid_swaps(spark):
    def body():
        schema = StructType([
            StructField("battery_type", StringType(), True),
            StructField("energy_kwh", DoubleType(), True),
        ])
        df = spark.createDataFrame([("LFP", 10.0), ("NMC", -1.0), (None, 8.0)], schema)
        rows = solution.filter_valid_swaps(df).collect()
        assert len(rows) == 1 and rows[0]["battery_type"] == "LFP", f"Unexpected valid rows {rows}"
    _run(3, "filter_valid_swaps", "Only non-negative energy rows with non-null battery_type", body)

def test_04_monthly_battery_energy(spark):
    def body():
        df = spark.createDataFrame([
            ("LFP", "2026-01", 10.0),
            ("LFP", "2026-01", 15.0),
            ("NMC", "2026-01", 8.0),
        ], ["battery_type", "swap_month", "energy_kwh"])
        got = solution.monthly_battery_energy(df)
        assert got.columns == ["battery_type", "swap_month", "total_energy_kwh"], f"Unexpected columns {got.columns}"
        vals = {(r["battery_type"], r["swap_month"]): float(r["total_energy_kwh"]) for r in got.collect()}
        assert vals[("LFP", "2026-01")] == 25.0 and vals[("NMC", "2026-01")] == 8.0, f"Wrong totals {vals}"
    _run(4, "monthly_battery_energy", "Grouped monthly totals with exact output columns", body)

def test_05_top_battery_type(spark):
    def body():
        df = spark.createDataFrame([
            ("LFP", 30.0), ("LFP", 20.0), ("NMC", 45.0)
        ], ["battery_type", "total_energy_kwh"])
        got = solution.top_battery_type(df)
        assert got[0] == "LFP" and builtins.abs(float(got[1]) - 50.0) < 1e-9, f"Unexpected top battery {got}"
    _run(5, "top_battery_type", "('LFP', 50.0) after regrouping total energy", body)

def test_06_top_battery_type_tie_empty(spark):
    def body():
        df = spark.createDataFrame([("NMC", 20.0), ("LFP", 20.0)], ["battery_type", "total_energy_kwh"])
        got = solution.top_battery_type(df)
        assert got[0] == "LFP" and builtins.abs(float(got[1]) - 20.0) < 1e-9, f"Tie rule failed {got}"
        empty = spark.createDataFrame([], "battery_type string, total_energy_kwh double")
        assert solution.top_battery_type(empty) == ("", 0.0), "Empty default must be ('', 0.0)"
    _run(6, "top_battery_type_tie_empty", "Alphabetical tie-break and empty default", body)
