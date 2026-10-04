from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.sql.functions import *
from pyspark.sql.window import Window


def define_maintenance_schema() -> StructType:
    return StructType([
        StructField("maintenance_id",StringType(),True),
        StructField("aircraft_id",StringType(),True),
        StructField("maintenance_date",StringType(),True),
        StructField("maintenance_cost",DoubleType(),True),
        StructField("maintenance_status",StringType(),True),
    ])

def load_maintenance_data(spark: SparkSession, path: str, schema: StructType) -> DataFrame:
    df=spark.read.csv(path,header=True,schema=schema)
    return df.withColumn("maintenance_date",to_date("maintenance_date"))

def join_aircraft_with_maintenance(aircraft_df: DataFrame, maintenance_df: DataFrame) -> DataFrame:
    return aircraft_df.join(maintenance_df,"aircraft_id","inner")

def aircraft_without_maintenance(aircraft_df: DataFrame, maintenance_df: DataFrame) -> DataFrame:
    return aircraft_df.join(maintenance_df,"aircraft_id","left_anti")

def rank_aircraft_by_maintenance_cost(df: DataFrame) -> DataFrame:
    df=df.groupby(["aircraft_model","aircraft_id"]).agg(sum("maintenance_cost").alias("total_maintenance_cost"))
    window_spec=Window.partitionBy("aircraft_model").orderBy(col("total_maintenance_cost").desc())
    return df.withColumn("maintenance_rank",rank().over(window_spec))
