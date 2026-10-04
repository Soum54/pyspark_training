from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import *
from pyspark.sql.window import Window


def define_payment_schema() -> StructType:
    return StructType([
        StructField("payment_id",StringType(),True),
        StructField("crew_id",StringType(),True),
        StructField("payment_date",StringType(),True),
        StructField("payment_amount",DoubleType(),True),
        StructField("payment_status",StringType(),True)
    ])

def load_and_clean_payments(spark: SparkSession, path: str, schema: StructType) -> DataFrame:
    df=spark.read.csv(path,header=True,schema=schema)
    df=df.withColumn("payment_date",to_date("payment_date","yyyy-MM-dd"))
    df=df.dropna(subset=["payment_id","crew_id","payment_amount"])
    df=df.fillna({"payment_status":"Pending"})
    df=df.withColumn("payment_year",year("payment_date"))
    return df.withColumn("payment_month",month("payment_date"))

def join_crew_with_payments(crew_df: DataFrame, payments_df: DataFrame) -> DataFrame:
    return crew_df.join(payments_df,"crew_id","inner")

def crew_without_payments(crew_df: DataFrame, payments_df: DataFrame) -> DataFrame:
    return crew_df.join(payments_df,"crew_id","left_anti")

def rank_crew_by_total_payment(df: DataFrame) -> DataFrame:
    df=df.groupBy("department","crew_id","crew_name").agg(sum("payment_amount").alias("total_payment"))
    window_spec=Window.partitionBy("department").orderBy(col("total_payment").desc())
    df=df.withColumn("payment_rank",rank().over(window_spec))
    return df.select("department","crew_id","crew_name","total_payment","payment_rank")
