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

ASSESSMENT_NAME = 'Digital Banking KYC Risk Insights'
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

def test_01_load_kyc_data(spark):
    def body():
        df=solution.load_kyc_data(spark,str(ROOT/'data'/'kyc_customers.csv'))
        assert df.count()==5, 'Expected 5 KYC rows'
        assert dict(df.dtypes)['onboarding_date']=='date', 'onboarding_date must be DateType'
    _run(1,'load_kyc_data','5 rows and onboarding_date as date',body)

def test_02_remove_invalid_emails(spark):
    def body():
        df=spark.createDataFrame([('C1',' a@b.com '),('C2','bad'),('C3',None),('C4','   ')],['customer_id','email'])
        out=solution.remove_invalid_emails(df)
        rows=out.collect(); assert len(rows)==1 and rows[0]['customer_id']=='C1', 'Only valid trimmed email should remain'
        assert rows[0]['email']=='a@b.com', 'Email should be trimmed'
    _run(2,'remove_invalid_emails','Trimmed valid-email rows only',body)

def test_03_filter_review_customers(spark):
    def body():
        df=spark.createDataFrame([('C1',60,'PENDING'),('C2',80,'REVIEW'),('C3',90,'APPROVED'),('C4',40,'REVIEW')],['customer_id','risk_score','kyc_status'])
        got={r['customer_id'] for r in solution.filter_review_customers(df,50,85).collect()}
        assert got=={'C1','C2'}, f'Unexpected review set {got}'
    _run(3,'filter_review_customers','Inclusive score range plus PENDING/REVIEW status',body)

def test_04_risk_score_statistics(spark):
    def body():
        df=spark.createDataFrame([(10,),(20,),(None,),(30,)],['risk_score'])
        got=solution.risk_score_statistics(df)
        assert got['min_score']==10 and got['max_score']==30, f'Wrong min/max {got}'
        assert builtins.abs(float(got['avg_score'])-20.0)<1e-9 and got['total_customers']==3, f'Wrong avg/count {got}'
    _run(4,'risk_score_statistics','Dictionary with min/max/avg/count over non-null scores',body)

def test_05_city_highest_average_risk(spark):
    def body():
        df=spark.createDataFrame([('Mumbai',80),('Mumbai',60),('Delhi',75)],['city','risk_score'])
        got=solution.city_highest_average_risk(df)
        assert got[0]=='Delhi' and builtins.abs(float(got[1])-75.0)<1e-9, f'Unexpected top city {got}'
    _run(5,'city_highest_average_risk','Tuple of city with highest average risk',body)

def test_06_city_tie_empty(spark):
    def body():
        df=spark.createDataFrame([('B',80),('A',80)],['city','risk_score'])
        assert solution.city_highest_average_risk(df)==('A',80.0), 'Tie must resolve alphabetically'
        assert solution.city_highest_average_risk(spark.createDataFrame([],df.schema))==('',0.0), 'Empty default must be ("",0.0)'
    _run(6,'city_highest_average_risk_tie_empty','Alphabetical tie and empty default',body)
