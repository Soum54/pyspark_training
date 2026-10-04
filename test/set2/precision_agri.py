from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import to_date,year,month,avg


def load_field_inspections(spark: SparkSession, path: str) -> DataFrame:
    df=spark.read.csv(path,header=True,inferSchema=True)
    return df.withColumn("inspection_date",to_date("inspection_date"))

def drop_incomplete_inspections(df: DataFrame) -> DataFrame:
    df=df.dropna(subset=["inspection_id","field_id","moisture_level"])
    return df

def fill_missing_inspector(df: DataFrame) -> DataFrame:
    return df.fillna({"inspector_name":"Unknown"})

def add_inspection_year_month(df: DataFrame) -> DataFrame:
    df=(df.withColumn("inspection_year",year("inspection_date"))
       .withColumn("inspection_month",month("inspection_date"))
    )
    return df

def average_moisture_by_crop(df: DataFrame) -> DataFrame:
    df=df.groupby("crop_type").agg(avg("moisture_level").alias("avg_moisture"))
    return df.select("crop_type","avg_moisture")
