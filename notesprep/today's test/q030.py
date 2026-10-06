from typing import Tuple, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import to_date,trim,col,avg,count,min,max
from pyspark.sql.window import Window

def load_kyc_data(spark: SparkSession, path: str) -> DataFrame:
    df=spark.read.csv(path,header=True,inferSchema=True)
    return df.withColumn("onboarding_date",to_date("onboarding_date"))

def remove_invalid_emails(df: DataFrame) -> DataFrame:
    df=df.withColumn("email",trim(col("email")))
    df=df.filter(
        (col("email").isNotNull()) &
        (col("email")!="") &
        (col("email").rlike(r"^[^@]+@[^@]+$"))
    )
    return df

def filter_review_customers(df: DataFrame, min_score: int, max_score: int) -> DataFrame:
    return df.filter(
        (col("risk_score").between(min_score,max_score)) &
        (col("kyc_status").isin("PENDING","REVIEW"))
    )

def risk_score_statistics(df: DataFrame) -> dict:
    df=df.filter(col("risk_score").isNotNull())
    df=df.agg(min("risk_score").alias("min_score"),max("risk_score").alias("max_score"),avg("risk_score").alias("avg_score"),count("risk_score").alias("total_customers")).first()
    return {
        "min_score":df["min_score"],
        "max_score":df["max_score"],
        "avg_score":df["avg_score"],
        "total_customers":df["total_customers"]
    }

def city_highest_average_risk(df: DataFrame) -> tuple:
    df=df.filter((col("risk_score").isNotNull()) |(col("city").isNotNull()))
    df=df.groupBy("city").agg(avg("risk_score").alias("avg_score")).orderBy(col("avg_score").desc(),col("city").asc()).first()
    if df is None:
        return ("",0.0)
    return (df["city"],float(df["avg_score"]))
