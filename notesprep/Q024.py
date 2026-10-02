from typing import Tuple, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType,StructField,StringType,DoubleType,IntegerType
from pyspark.sql.functions import col,to_timestamp,coalesce,lit,concat_ws
from pyspark.sql.window import Window

def define_settlement_schema() -> StructType:
    return StructType([
        StructField("settlement_id",StringType(),True),
        StructField("merchant_id",StringType(),True),
        StructField("settlement_ts",StringType(),True),
        StructField("gross_amount",DoubleType(),True),
        StructField("fee_amount",DoubleType(),True),
        StructField("settlement_status",StringType(),True)
    ])

def load_settlement_data(spark: SparkSession, settlements_path: str, merchants_path: str, schema: StructType) -> tuple:
    settlements_df=spark.read.csv(settlements_path,header=True,schema=schema)
    merchants_df=spark.read.csv(merchants_path,header=True,inferSchema=True)
    settlements_df=settlements_df.withColumn("settlement_ts",to_timestamp(col("settlement_ts")))
    merchants_df=merchants_df.withColumn("risk_score",col("risk_score").cast(IntegerType()))
    return settlements_df,merchants_df

def enrich_settlements(settlements_df: DataFrame, merchants_df: DataFrame) -> DataFrame:
    df=settlements_df.join(merchants_df,"merchant_id","inner")
    df=df.withColumn("city",coalesce(col("city"),lit("Unknown")))
    df=df.withColumn("merchant_label",concat_ws("-",col("merchant_name"),col("city")))
    df=df.withColumn("net_amount",col("gross_amount")-col("fee_amount"))
    return df

def merchants_without_successful_settlement(merchants_df: DataFrame, settlements_df: DataFrame) -> DataFrame:
    pass

def rank_merchants_by_net_amount(df: DataFrame) -> DataFrame:
    pass

