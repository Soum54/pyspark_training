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

ASSESSMENT_NAME = 'Music Streaming Release Insights'
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

def test_01_load_music_tracks(spark):
    def body():
        df=solution.load_music_tracks(spark,str(ROOT/'data'/'music_tracks.csv'))
        assert df.count()==5 and dict(df.dtypes)['release_date']=='date', 'Expected 5 rows and release_date DateType'
    _run(1,'load_music_tracks','5 rows and release_date as date',body)

def test_02_fill_missing_genre(spark):
    def body():
        df=spark.createDataFrame([('T1','Rock'),('T2',None)],['track_id','genre'])
        got={r['track_id']:r['genre'] for r in solution.fill_missing_genre(df).collect()}
        assert got=={'T1':'Rock','T2':'Uncategorized'}, f'Wrong fillna result {got}'
    _run(2,'fill_missing_genre','Null genre filled with Uncategorized',body)

def test_03_add_release_calendar(spark):
    def body():
        df=spark.createDataFrame([('T1','2026-09-15')],['track_id','d']).withColumn('release_date',F.to_date('d')).drop('d')
        r=solution.add_release_calendar(df).first()
        assert r['release_year']==2026 and r['release_month']==9, 'Wrong year/month'
        assert str(r['release_month_start']).startswith('2026-09-01'), f'Wrong month bucket {r["release_month_start"]}'
    _run(3,'add_release_calendar','year/month/date_trunc columns',body)

def test_04_unique_published_artists(spark):
    def body():
        df=spark.createDataFrame([('Zed','Published'),('Amy','Published'),('Zed','Published'),('Bob','Draft')],['artist_name','status'])
        assert solution.unique_published_artists(df)==['Amy','Zed'], 'Expected sorted distinct Published artists'
    _run(4,'unique_published_artists','Sorted distinct Python list',body)

def test_05_top_n_genres_by_streams(spark):
    def body():
        df=spark.createDataFrame([('Rock','Published',100),('Rock','Published',50),('Jazz','Published',130),('Pop','Draft',1000)],['genre','status','stream_count'])
        out=solution.top_n_genres_by_streams(df,2).collect()
        assert [(r['genre'],r['total_streams']) for r in out]==[('Rock',150),('Jazz',130)], f'Unexpected top-N {out}'
    _run(5,'top_n_genres_by_streams','Published-only sums ordered and limited to n',body)

def test_06_top_n_modified(spark):
    def body():
        df=spark.createDataFrame([('B','Published',100),('A','Published',100),('C','Published',90)],['genre','status','stream_count'])
        out=solution.top_n_genres_by_streams(df,1).collect()
        assert len(out)==1 and out[0]['genre']=='A', 'Tie must use genre ascending before limit'
    _run(6,'top_n_genres_by_streams_modified','Dynamic n and deterministic tie ordering',body)
