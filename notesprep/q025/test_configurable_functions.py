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

ASSESSMENT_NAME = 'Remote Patient Monitoring Alert Analytics'
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

def test_01_define_observation_schema(spark):
    def body():
        s=solution.define_observation_schema(); got=[(f.name,type(f.dataType).__name__) for f in s.fields]
        exp=[('observation_id','StringType'),('patient_id','StringType'),('reading_ts','StringType'),('systolic','StringType'),('heart_rate','StringType'),('device_status','StringType')]
        assert got==exp, f'Expected {exp}, got {got}'
    _run(1,'define_observation_schema','Exact observation StructType',body)

def test_02_load_monitoring_data(spark):
    def body():
        schema=StructType([StructField(x,StringType(),True) for x in ['observation_id','patient_id','reading_ts','systolic','heart_rate','device_status']])
        o,p=solution.load_monitoring_data(spark,str(ROOT/'data'/'observations.csv'),str(ROOT/'data'/'patients.csv'),schema)
        d=dict(o.dtypes); assert o.count()==4 and p.count()==3, 'Wrong row counts'
        assert d['reading_ts']=='timestamp' and d['systolic']=='int' and d['heart_rate']=='int', f'Wrong converted types {d}'
    _run(2,'load_monitoring_data','Timestamp and integer casts plus correct counts',body)

def test_03_classify_readings(spark):
    def body():
        df=spark.createDataFrame([('O1',185,80),('O2',150,90),('O3',120,70),('O4',130,110)],['observation_id','systolic','heart_rate'])
        got={r['observation_id']:r['alert_level'] for r in solution.classify_readings(df).collect()}
        assert got=={'O1':'Critical','O2':'Warning','O3':'Normal','O4':'Warning'}, f'Wrong alert levels {got}'
    _run(3,'classify_readings','Critical/Warning/Normal when logic',body)

def test_04_latest_reading_per_patient(spark):
    def body():
        df=spark.createDataFrame([('O1','P1','2026-09-01 08:00:00'),('O2','P1','2026-09-01 09:00:00'),('O3','P2','2026-09-01 07:00:00')],['observation_id','patient_id','ts']).withColumn('reading_ts',F.to_timestamp('ts')).drop('ts')
        got={r['patient_id']:r['observation_id'] for r in solution.latest_reading_per_patient(df).collect()}
        assert got=={'P1':'O2','P2':'O3'}, f'row_number latest-row result incorrect {got}'
    _run(4,'latest_reading_per_patient','One latest reading per patient using row_number',body)

def test_05_care_team_alert_summary(spark):
    def body():
        o=spark.createDataFrame([('O1','P1','Warning',100),('O2','P2','Critical',130),('O3','P3','Normal',70)],['observation_id','patient_id','alert_level','heart_rate'])
        p=spark.createDataFrame([('P1','TeamA'),('P2','TeamA'),('P3','TeamB')],['patient_id','care_team'])
        rows=solution.care_team_alert_summary(o,p).collect(); assert len(rows)==1, f'Expected one team row, got {rows}'
        r=rows[0]; assert r['care_team']=='TeamA' and r['alert_count']==2 and builtins.abs(float(r['avg_heart_rate'])-115.0)<1e-9, f'Wrong summary {r}'
    _run(5,'care_team_alert_summary','Joined Warning/Critical count and avg heart rate',body)

def test_06_latest_modified(spark):
    def body():
        df=spark.createDataFrame([('A','P1','2026-09-01 09:00:00'),('B','P1','2026-09-01 10:00:00')],['observation_id','patient_id','ts']).withColumn('reading_ts',F.to_timestamp('ts')).drop('ts')
        rows=solution.latest_reading_per_patient(df).collect(); assert len(rows)==1 and rows[0]['observation_id']=='B', 'Latest timestamp must win on modified input'
    _run(6,'latest_reading_per_patient_modified','Dynamic row_number latest-row behavior',body)
