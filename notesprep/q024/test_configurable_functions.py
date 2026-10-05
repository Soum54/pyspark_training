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

ASSESSMENT_NAME = 'Merchant Settlement Risk Analytics'
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

def test_01_define_settlement_schema(spark):
    def body():
        s=solution.define_settlement_schema(); got=[(f.name,type(f.dataType).__name__) for f in s.fields]
        exp=[('settlement_id','StringType'),('merchant_id','StringType'),('settlement_ts','StringType'),('gross_amount','DoubleType'),('fee_amount','DoubleType'),('settlement_status','StringType')]
        assert got==exp, f'Expected {exp}, got {got}'
    _run(1,'define_settlement_schema','Exact 6-field StructType',body)

def test_02_load_settlement_data(spark):
    def body():
        schema=StructType([StructField('settlement_id',StringType(),True),StructField('merchant_id',StringType(),True),StructField('settlement_ts',StringType(),True),StructField('gross_amount',DoubleType(),True),StructField('fee_amount',DoubleType(),True),StructField('settlement_status',StringType(),True)])
        s,m=solution.load_settlement_data(spark,str(ROOT/'data'/'settlements.csv'),str(ROOT/'data'/'merchants.csv'),schema)
        assert s.count()==4 and m.count()==4, 'Expected 4 rows in each dataset'
        assert dict(s.dtypes)['settlement_ts']=='timestamp' and dict(m.dtypes)['risk_score'] in ('int','bigint'), 'Timestamp/cast requirements not met'
    _run(2,'load_settlement_data','Correct counts, timestamp conversion and risk_score cast',body)

def test_03_enrich_settlements(spark):
    def body():
        s=spark.createDataFrame([('S1','M1',100.0,5.0)],['settlement_id','merchant_id','gross_amount','fee_amount'])
        m=spark.createDataFrame([('M1','Shop',None)], StructType([StructField('merchant_id',StringType(),False),StructField('merchant_name',StringType(),False),StructField('city',StringType(),True)]))
        r=solution.enrich_settlements(s,m).first()
        assert r['city']=='Unknown' and r['merchant_label']=='Shop - Unknown', 'coalesce/concat_ws result incorrect'
        assert builtins.abs(float(r['net_amount'])-95.0)<1e-9, 'net_amount must equal gross-fee'
    _run(3,'enrich_settlements','Inner join, coalesce, concat_ws and net amount',body)

def test_04_merchants_without_successful_settlement(spark):
    def body():
        m=spark.createDataFrame([('M1',),('M2',),('M3',)],['merchant_id'])
        s=spark.createDataFrame([('M1','SUCCESS'),('M2','FAILED')],['merchant_id','settlement_status'])
        got={r['merchant_id'] for r in solution.merchants_without_successful_settlement(m,s).collect()}
        assert got=={'M2','M3'}, f'Expected M2/M3, got {got}'
    _run(4,'merchants_without_successful_settlement','left_anti against distinct SUCCESS merchants',body)

def test_05_rank_merchants_by_net_amount(spark):
    def body():
        df=spark.createDataFrame([('M1','A','S1','SUCCESS',100.0),('M1','A','S2','SUCCESS',50.0),('M2','B','S3','SUCCESS',120.0),('M3','C','S4','FAILED',999.0)],['merchant_id','merchant_label','settlement_id','settlement_status','net_amount'])
        got={r['merchant_id']:(float(r['total_net_amount']),r['settlement_count'],r['settlement_rank']) for r in solution.rank_merchants_by_net_amount(df).collect()}
        assert got=={'M1':(150.0,2,1),'M2':(120.0,1,2)}, f'Unexpected ranking {got}'
    _run(5,'rank_merchants_by_net_amount','SUCCESS aggregation with sum/count/rank',body)

def test_06_rank_tie_behavior(spark):
    def body():
        df=spark.createDataFrame([('M1','A','S1','SUCCESS',100.0),('M2','B','S2','SUCCESS',100.0),('M3','C','S3','SUCCESS',50.0)],['merchant_id','merchant_label','settlement_id','settlement_status','net_amount'])
        ranks={r['merchant_id']:r['settlement_rank'] for r in solution.rank_merchants_by_net_amount(df).collect()}
        assert ranks=={'M1':1,'M2':1,'M3':3}, f'rank() tie behavior must be 1,1,3; got {ranks}'
    _run(6,'rank_merchants_by_net_amount_tie','rank() tie behavior 1,1,3',body)
