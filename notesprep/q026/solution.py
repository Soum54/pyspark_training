from typing import Tuple, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType,StructField,IntegerType,StringType,DoubleType
from pyspark.sql.functions import col,dense_rank,lit,to_timestamp,unix_timestamp,when,coalesce,concat_ws
from pyspark.sql.window import Window

def define_run_schema() -> StructType:
    return StructType([
        StructField("run_id",StringType(),True),
        StructField("reactor_id",StringType(),True),
        StructField("process_type",StringType(),True),
        StructField("start_ts",StringType(),True),
        StructField("end_ts",StringType(),True),
        StructField("input_mass",DoubleType(),True),
        StructField("output_mass",DoubleType(),True),
        StructField("run_status",StringType(),True),
    ])

def load_bioreactor_data(spark: SparkSession, runs_path: str, reactors_path: str, schema: StructType) -> tuple:
    df=spark.read.csv(runs_path,header=True,schema=schema)
    df1=spark.read.csv(reactors_path,header=True,inferSchema=True)
    df=(df.withColumn("start_ts",to_timestamp("start_ts"))
            .withColumn("end_ts",to_timestamp("end_ts"))
    )
    return df,df1

def compute_run_metrics(df: DataFrame) -> DataFrame:
    duration_minutes=(unix_timestamp(col("end_ts"))-unix_timestamp(col("start_ts")))/60
    df=df.withColumn("duration_minutes",when(duration_minutes<0,0).otherwise(duration_minutes))
    yield_pct=(col("output_mass")/col("input_mass"))*100
    return df.withColumn("yield_pct",when(yield_pct<0,0).otherwise(yield_pct))

def join_reactor_metadata(runs_df: DataFrame, reactors_df: DataFrame) -> DataFrame:
    df=runs_df.join(reactors_df,"reactor_id","inner")
    df=df.withColumn("facility",coalesce(col("facility"),lit("Unknown")))
    df=df.withColumn("reactor_label",concat_ws(" - ",col("reactor_name"),col("facility")))
    return df

def dense_rank_runs_by_yield(df: DataFrame) -> DataFrame:
    df=df.filter(
        (col("run_status")=="COMPLETED") &
        (col("yield_pct").isNotNull())
    )
    window_spec=Window.partitionBy("process_type").orderBy(col("yield_pct").desc())
    df=df.withColumn("yield_rank",dense_rank().over(window_spec))
    return df.select("run_id","process_type","yield_pct","yield_rank")
