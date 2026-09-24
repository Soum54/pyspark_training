from typing import List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import col, to_date, datediff, when, sum, avg, count


def define_schema() -> StructType:
    return StructType([
        StructField("txn_id",StringType(),True),
        StructField("order_date",StringType(),True),
        StructField("customer_id",StringType(),True),
        StructField("region",StringType(),True),
        StructField("channel",StringType(),True),
        StructField("category",StringType(),True),
        StructField("product_id",StringType(),True),
        StructField("quantity",IntegerType(),True),
        StructField("unit_price",DoubleType(),True),
        StructField("discount_rate",DoubleType(),True),
        StructField("returned",StringType(),True),
        StructField("ship_date",StringType(),True),
        StructField("delivery_date",StringType(),True)
    ])

def load_data(spark: SparkSession, path: str, schema: StructType) -> DataFrame:
    return spark.read.csv(path,header=True,schema=schema)

def parse_dates(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("order_date",to_date("order_date"))
        .withColumn("ship_date",to_date("ship_date"))
        .withColumn("delivery_date",to_date("delivery_date"))
    )

def add_gross_amount(df: DataFrame) -> DataFrame:
    return df.withColumn("gross_amount",col("quantity")*col("unit_price"))


def add_net_amount(df: DataFrame) -> DataFrame:
    df=add_gross_amount(df)
    return df.withColumn("net_amount",col("gross_amount")*(1-col("discount_rate")))

def add_delivery_days(df: DataFrame) -> DataFrame:
    return df.withColumn("delivery_days",datediff(col("delivery_date"),col("ship_date")))


def flag_on_time_delivery(df: DataFrame, max_days: int) -> DataFrame:
    df=add_delivery_days(df)
    return df.withColumn("is_on_time",when(col("delivery_days") <= max_days,True).otherwise(False))

def filter_returned_orders(df):
    return df.filter(col("returned")=='Y')

def filter_by_region(df, region) -> DataFrame:
    return df.filter(col("region")==region)

def top_n_customers_by_spend(df, n) -> DataFrame:
    df=add_net_amount(df)
    return (df.groupBy(col("customer_id"))
            .agg(sum(col("net_amount")).alias("total_spend"))
            .orderBy(col("total_spend").desc(),col("customer_id").asc())).limit(n)

def revenue_by_category(df):
    df=add_net_amount(df)
    return df.groupBy("category").agg(sum("net_amount").alias("total_revenue"))

def top_category_by_revenue(df) -> str:
    df=add_net_amount(df)
    row=df.groupBy("category").agg(sum("net_amount").alias("total_revenue")).orderBy(["total_revenue","category"],ascending=[0,1]).limit(1).collect()
    return row[0]["category"] if row else None


