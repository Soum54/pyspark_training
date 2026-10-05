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

ASSESSMENT_NAME = 'Autonomous Last-Mile Delivery Trip Analytics'
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

def test_01_load_delivery_trips(spark):
    def body():
        df = solution.load_delivery_trips(spark, str(ROOT / "data" / "delivery_trips.csv"))
        dtypes = dict(df.dtypes)
        assert df.count() == 5, "Expected 5 delivery trip rows"
        assert dtypes["dispatch_ts"] == "timestamp", f"dispatch_ts type is {dtypes['dispatch_ts']}"
        assert dtypes["promised_ts"] == "timestamp", f"promised_ts type is {dtypes['promised_ts']}"
        assert dtypes["actual_ts"] == "timestamp", f"actual_ts type is {dtypes['actual_ts']}"
        assert dtypes["distance_km"] == "double" and dtypes["fare_amount"] == "double", f"Wrong numeric types {dtypes}"
    _run(1, "load_delivery_trips", "5 rows, three timestamp columns, distance/fare as double", body)

def test_02_compute_trip_metrics(spark):
    def body():
        schema = StructType([
            StructField("trip_id", StringType(), False),
            StructField("dispatch_ts", TimestampType(), True),
            StructField("promised_ts", TimestampType(), True),
            StructField("actual_ts", TimestampType(), True),
            StructField("distance_km", DoubleType(), True),
        ])
        df = spark.createDataFrame([
            ("T1", datetime(2026,1,1,9,0), datetime(2026,1,1,10,0), datetime(2026,1,1,10,30), 15.0),
            ("T2", datetime(2026,1,1,9,0), datetime(2026,1,1,10,0), datetime(2026,1,1,9,45), 6.0),
        ], schema)
        rows = {r["trip_id"]: r for r in solution.compute_trip_metrics(df).collect()}
        assert builtins.abs(float(rows["T1"]["delay_hours"]) - 0.5) < 1e-9, f"Wrong delay {rows['T1']}"
        assert builtins.abs(float(rows["T1"]["trip_hours"]) - 1.5) < 1e-9, f"Wrong trip hours {rows['T1']}"
        assert builtins.abs(float(rows["T1"]["avg_speed_kmph"]) - 10.0) < 1e-9, f"Wrong speed {rows['T1']}"
        assert float(rows["T2"]["delay_hours"]) == 0.0, f"Negative delay must become zero {rows['T2']}"
    _run(2, "compute_trip_metrics", "Correct non-negative delay/trip hours and average speed", body)

def test_03_filter_significant_delays(spark):
    def body():
        df = spark.createDataFrame([
            ("T1", 1.5, 20.0),
            ("T2", 0.5, 30.0),
            ("T3", 2.0, 5.0),
        ], ["trip_id", "delay_hours", "distance_km"])
        ids = [r["trip_id"] for r in solution.filter_significant_delays(df, 1.0, 10.0).collect()]
        assert ids == ["T1"], f"Unexpected filtered trips {ids}"
    _run(3, "filter_significant_delays", "Only T1 satisfies delay > 1.0 and distance >= 10.0", body)

def test_04_partner_delay_summary(spark):
    def body():
        df = spark.createDataFrame([
            ("T1", "P1", 1.0, 100.0, "Completed"),
            ("T2", "P1", 2.0, 200.0, "Completed"),
            ("T3", "P2", 5.0, 300.0, "Cancelled"),
        ], ["trip_id", "partner", "delay_hours", "fare_amount", "status"])
        got = solution.partner_delay_summary(df)
        assert got.columns == ["partner", "total_delay_hours", "avg_fare", "trip_count"], f"Unexpected columns {got.columns}"
        rows = got.collect()
        assert len(rows) == 1, f"Expected only P1 summary, got {rows}"
        r = rows[0]
        assert r["partner"] == "P1", f"Wrong partner {r}"
        assert builtins.abs(float(r["total_delay_hours"]) - 3.0) < 1e-9, f"Wrong total delay {r}"
        assert builtins.abs(float(r["avg_fare"]) - 150.0) < 1e-9, f"Wrong avg fare {r}"
        assert int(r["trip_count"]) == 2, f"Wrong trip count {r}"
    _run(4, "partner_delay_summary", "Completed-only partner sum/avg/count with exact columns", body)

def test_05_highest_average_delay_reason(spark):
    def body():
        df = spark.createDataFrame([
            ("Traffic", 1.0), ("Traffic", 3.0), ("Weather", 2.5)
        ], ["delay_reason", "delay_hours"])
        got = solution.highest_average_delay_reason(df)
        assert got[0] == "Weather" and builtins.abs(float(got[1]) - 2.5) < 1e-9, f"Unexpected reason {got}"
    _run(5, "highest_average_delay_reason", "('Weather', 2.5)", body)

def test_06_highest_average_delay_reason_tie_empty(spark):
    def body():
        df = spark.createDataFrame([("Weather", 2.0), ("Traffic", 2.0)], ["delay_reason", "delay_hours"])
        got = solution.highest_average_delay_reason(df)
        assert got[0] == "Traffic" and builtins.abs(float(got[1]) - 2.0) < 1e-9, f"Tie rule failed {got}"
        empty = spark.createDataFrame([], "delay_reason string, delay_hours double")
        assert solution.highest_average_delay_reason(empty) == ("", 0.0), "Empty default must be ('', 0.0)"
    _run(6, "highest_average_delay_reason_tie_empty", "Alphabetical tie-break and empty default", body)
