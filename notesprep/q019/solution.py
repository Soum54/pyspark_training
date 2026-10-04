from typing import Tuple, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import to_date,col,year,month,date_trunc,sum
from pyspark.sql.window import Window

def load_music_tracks(spark: SparkSession, path: str) -> DataFrame:
    df=spark.read.csv(path,header=True,inferSchema=True)
    return df.withColumn("release_date",to_date("release_date"))

def fill_missing_genre(df: DataFrame) -> DataFrame:
    return df.fillna({"genre":"Uncategorized"})

def add_release_calendar(df: DataFrame) -> DataFrame:
    return (df.withColumn("release_year",year("release_date"))
        .withColumn("release_month",month("release_date"))
        .withColumn("release_month_start",date_trunc("month",col("release_date")))
    )

def unique_published_artists(df: DataFrame) -> list:
    df=df.filter(col("status")=="Published")
    df=df.select("artist_name").distinct().orderBy(col("artist_name").asc()).collect()
    result=[]
    for i in df:
        result.append(i["artist_name"])
    return result

def top_n_genres_by_streams(df: DataFrame, n: int) -> DataFrame:
    df=df.filter(col("status")=="Published")
    df=df.groupBy("genre").agg(sum("stream_count").alias("total_streams")).orderBy(col("total_streams").desc(),col("genre").asc()).limit(n)
    return df

