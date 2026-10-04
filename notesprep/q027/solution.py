from typing import Tuple, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import to_timestamp,col,when,count,sum
from pyspark.sql.window import Window

def load_ad_impressions(spark: SparkSession, path: str) -> DataFrame:
    df=spark.read.csv(path,header=True,inferSchema=True)
    df=df.withColumn("impression_ts",to_timestamp("impression_ts"))
    df=(df.withColumn("watched_seconds",col("watched_seconds").cast(DoubleType()))
        .withColumn("ad_length_seconds",col("ad_length_seconds").cast(DoubleType()))
        .withColumn("spend_amount",col("spend_amount").cast(DoubleType()))
    )
    return df

def join_campaign_metadata(impressions_df: DataFrame, campaigns_df: DataFrame) -> DataFrame:
    return impressions_df.join(campaigns_df,"campaign_id","inner")

def add_engagement_metrics(df: DataFrame) -> DataFrame:
    df=df.withColumn("watch_pct",when(
        (col("ad_length_seconds").isNull()) | (col("ad_length_seconds")<=0),"0.0")
        .otherwise((col("watched_seconds"))/(col("ad_length_seconds"))*100)
        )
    df=df.withColumn("engagement_band",when(
        col("watch_pct")>=80,"High")
        .when(col("watch_pct")>=40,"Medium")
        .otherwise("Low")
        )
    return df

def campaign_performance_summary(df: DataFrame) -> DataFrame:
    df=df.filter(col("delivery_status")=="DELIVERED")
    df=(df.groupBy("campaign_id","campaign_name").
        agg(count("impression_id").alias("impression_count"),
            sum("spend_amount").alias("total_spend"),
            sum(when(col("clicked")=="Y",1)
                .otherwise(0)).alias("click_count"))
    )
    return df

def top_n_campaigns_by_spend(df: DataFrame, n: int) -> DataFrame:
    df=df.orderBy(col("total_spend").desc(),col("campaign_id").asc()).limit(n)
    df=df.select("campaign_id","campaign_name","total_spend","impression_count","click_count")
    return df

