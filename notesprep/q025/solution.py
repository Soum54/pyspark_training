from typing import Tuple, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType,StructField,StringType,IntegerType,DoubleType
from pyspark.sql.functions import col,rank,to_timestamp,when,row_number,count,avg
from pyspark.sql.window import Window

def define_observation_schema() -> StructType:
    return StructType([
        StructField("observation_id",StringType(),True),
        StructField("patient_id",StringType(),True),
        StructField("reading_ts",StringType(),True),
        StructField("systolic",StringType(),True),
        StructField("heart_rate",StringType(),True),
        StructField("device_status",StringType(),True)
    ])

def load_monitoring_data(spark: SparkSession, observations_path: str, patients_path: str, schema: StructType) -> tuple:
    df1=spark.read.csv(observations_path,header=True,schema=schema)
    df2=spark.read.csv(patients_path,header=True,inferSchema=True)
    df1=df1.withColumn("reading_ts",to_timestamp("reading_ts"))
    df1=df1.withColumn("systolic",col("systolic").cast(IntegerType()))
    df1=df1.withColumn("heart_rate",col("heart_rate").cast(IntegerType()))
    return df1,df2

def classify_readings(df: DataFrame) -> DataFrame:
    df=(df.withColumn(
        "alert_level",
        when((col("systolic")>=180) | (col("heart_rate")>=130),"Critical")
        .when((col("systolic")>=140) | (col("heart_rate")>=100),"Warning")
        .otherwise("Normal")
        )
    )
    return df

def latest_reading_per_patient(df: DataFrame) -> DataFrame:
    window_spec=Window.partitionBy("patient_id").orderBy(col("reading_ts").desc())
    df=df.withColumn("latest_reading",row_number().over(window_spec))
    return df.filter(col("latest_reading")==1)

def care_team_alert_summary(observations_df: DataFrame, patients_df: DataFrame) -> DataFrame:
    df=observations_df.join(patients_df,"patient_id","inner")
    df=df.filter(col("alert_level").isin("Critical","Warning"))
    df=df.groupBy("care_team").agg(count("observation_id").alias("alert_count"),avg("heart_rate").alias("avg_heart_rate"))
    return df
