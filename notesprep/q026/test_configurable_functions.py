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

ASSESSMENT_NAME = 'Bioreactor Run Performance Analytics'
ROOT=Path(__file__).resolve().parent
LOG_FILE=ROOT/"test_report.log"
STRICT_QA=os.environ.get("LEARNLYTICA_STRICT_QA", "0") == "1"

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

def test_01_define_run_schema(spark):
    def body():
        s=solution.define_run_schema(); got=[(f.name,type(f.dataType).__name__) for f in s.fields]
        exp=[('run_id','StringType'),('reactor_id','StringType'),('process_type','StringType'),('start_ts','StringType'),('end_ts','StringType'),('input_mass','DoubleType'),('output_mass','DoubleType'),('run_status','StringType')]
        assert got==exp, f'Expected {exp}, got {got}'
    _run(1,'define_run_schema','Exact 8-field StructType',body)

def test_02_load_bioreactor_data(spark):
    def body():
        schema=StructType([StructField('run_id',StringType(),True),StructField('reactor_id',StringType(),True),StructField('process_type',StringType(),True),StructField('start_ts',StringType(),True),StructField('end_ts',StringType(),True),StructField('input_mass',DoubleType(),True),StructField('output_mass',DoubleType(),True),StructField('run_status',StringType(),True)])
        r,m=solution.load_bioreactor_data(spark,str(ROOT/'data'/'bioreactor_runs.csv'),str(ROOT/'data'/'reactors.csv'),schema)
        d=dict(r.dtypes); assert r.count()==4 and m.count()==4, 'Wrong row counts'; assert d['start_ts']=='timestamp' and d['end_ts']=='timestamp', f'Wrong timestamp types {d}'
    _run(2,'load_bioreactor_data','Correct counts and timestamp conversions',body)

def test_03_compute_run_metrics(spark):
    def body():
        df=(spark.createDataFrame([('R1','2026-09-01 08:00:00','2026-09-01 10:00:00',100.0,80.0),('R2','2026-09-01 10:00:00','2026-09-01 09:00:00',100.0,50.0)],['run_id','start_raw','end_raw','input_mass','output_mass']).withColumn('start_ts',F.to_timestamp('start_raw')).withColumn('end_ts',F.to_timestamp('end_raw')).drop('start_raw','end_raw'))
        got={r['run_id']:(float(r['duration_minutes']),float(r['yield_pct'])) for r in solution.compute_run_metrics(df).collect()}
        assert got=={'R1':(120.0,80.0),'R2':(0.0,50.0)}, f'Wrong duration/yield metrics {got}'
    _run(3,'compute_run_metrics','unix_timestamp duration, negative cap, yield percentage',body)

def test_04_join_reactor_metadata(spark):
    def body():
        runs=spark.createDataFrame([('R1','BR1')],['run_id','reactor_id']); refs=spark.createDataFrame([('BR1','Core',None)], StructType([StructField('reactor_id',StringType(),False),StructField('reactor_name',StringType(),False),StructField('facility',StringType(),True)]))
        r=solution.join_reactor_metadata(runs,refs).first(); assert r['facility']=='Unknown' and r['reactor_label']=='Core - Unknown', 'coalesce/concat_ws enrichment incorrect'
    _run(4,'join_reactor_metadata','Join plus Unknown facility and reactor label',body)

def test_05_dense_rank_runs_by_yield(spark):
    def body():
        df=spark.createDataFrame([('R1','P','COMPLETED',90.0),('R2','P','COMPLETED',90.0),('R3','P','COMPLETED',80.0),('R4','P','FAILED',100.0)],['run_id','process_type','run_status','yield_pct'])
        got={r['run_id']:r['yield_rank'] for r in solution.dense_rank_runs_by_yield(df).collect()}
        assert got=={'R1':1,'R2':1,'R3':2}, f'dense_rank must be 1,1,2; got {got}'
    _run(5,'dense_rank_runs_by_yield','COMPLETED dense_rank behavior 1,1,2',body)

def test_06_dense_rank_partition(spark):
    def body():
        df=spark.createDataFrame([('R1','P1','COMPLETED',80.0),('R2','P2','COMPLETED',70.0)],['run_id','process_type','run_status','yield_pct'])
        got={r['run_id']:r['yield_rank'] for r in solution.dense_rank_runs_by_yield(df).collect()}
        assert got=={'R1':1,'R2':1}, f'Ranking must restart per process_type, got {got}'
    _run(6,'dense_rank_runs_partition','Window partitioning by process_type',body)
