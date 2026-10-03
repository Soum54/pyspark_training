from typing import Tuple, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import to_date,col,trim
from pyspark.sql.window import Window

def load_kyc_data(spark: SparkSession, path: str) -> DataFrame:
    df=spark.read.csv(path,header=True,inferSchema=True)
    return df.withColumn("onboarding_date",to_date("onboarding_date"))

def remove_invalid_emails(df: DataFrame) -> DataFrame:
    df=df.filter((col("email").isNotNull()) & (trim(col("email")))!="")

def filter_review_customers(df: DataFrame, min_score: int, max_score: int) -> DataFrame:
    return (
        (df.filter(col("risk_score").between(min_score,max_score))) &
        (df.filter(col("kyc_status").isin("PENDING","REVIEW")))
    )

def risk_score_statistics(df: DataFrame) -> dict:
    df=df.filter(col("risk_score").isNotNull())
    return df.agg(min("risk_score").alias("min_score"),max("risk_score").alias("max_score"),avg("risk_score").alias("avg_score"),count(""))

def city_highest_average_risk(df: DataFrame) -> tuple:
    pass

