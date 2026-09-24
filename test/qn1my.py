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

def avg_discount_by_channel(df):
  return df.groupby("channel").agg(avg("discount_rate").alias("avg_discount"))

def daily_revenue_trend(df):
  df=add_net_amount(df)
  return df.groupby("order_date").agg(sum("net_amount").alias("daily_revenue")).orderBy("order_date")
def high_value_orders(df, threshold):
  df=add_net_amount(df)
  return df.filter(col("net_amount")>threshold)
def count_late_deliveries(df, max_days):
  df=add_delivery_days(df)
  return df.filter(col("delivery_days")>max_days).count()
def return_rate_by_category(df):
  df=df.groupby("category").agg(count("*").alias("total_cnt"),sum(when(col("returned")=="Y",1).otherwise(0)).alias("returned_cnt"))
  return df.withColumn("return_rate",col("returned_cnt")/col("total_cnt"))
  
def best_selling_product(df):
  pass
def list_channels(df)->List[str]:
  df=df.select("channel").distinct().orderBy("channel")
  return df.collect()
def orders_in_date_range(df, start_date, end_date):
  return df.filter((col("order_date")>=start_date) &(col("order_date")<=end_date))
from pyspark.sql import SparkSession

# Create Spark session
spark = SparkSession.builder \
    .appName("Retail Commerce Operations") \
    .getOrCreate()


# 1. Define schema
schema = define_schema()


# 2. Load data
df = load_data(
    spark,
    "/content/commerce.csv",
    schema
)

print("Original Data")
df.show()


# 3. Parse dates
df = parse_dates(df)

print("After Parsing Dates")
df.show()


# 4. Add gross amount
result = add_gross_amount(df)

print("Gross Amount")
result.show()


# 5. Add net amount
result = add_net_amount(df)

print("Net Amount")
result.show()


# 6. Add delivery days
result = add_delivery_days(df)

print("Delivery Days")
result.show()


# 7. Flag on-time delivery
result = flag_on_time_delivery(df, 5)

print("On Time Delivery")
result.show()


# 8. Filter returned orders
result = filter_returned_orders(df)

print("Returned Orders")
result.show()


# 9. Filter by region
result = filter_by_region(df, "North")

print("Orders by Region")
result.show()


# 10. Top N customers
result = top_n_customers_by_spend(df, 5)

print("Top 5 Customers")
result.show()


# 11. Revenue by category
result = revenue_by_category(df)

print("Revenue by Category")
result.show()


# 12. Top category by revenue
result = top_category_by_revenue(df)

print("Top Category")
print(result)


# 13. Average discount by channel
result = avg_discount_by_channel(df)

print("Average Discount by Channel")
result.show()


# 14. Daily revenue trend
result = daily_revenue_trend(df)

print("Daily Revenue")
result.show()


# 15. High value orders
result = high_value_orders(df, 1000)

print("High Value Orders")
result.show()


# 16. Count late deliveries
result = count_late_deliveries(df, 5)

print("Number of Late Deliveries")
print(result)


# 17. Return rate by category
result = return_rate_by_category(df)

print("Return Rate by Category")
result.show()


# 18. Best selling product
result = best_selling_product(df)

print("Best Selling Product")
print(result)


# 19. List channels
result = list_channels(df)

print("Available Channels")
print(result)


# 20. Orders in date range
result = orders_in_date_range(
    df,
    "2025-01-01",
    "2025-12-31"
)

print("Orders in Date Range")
result.show()


# Stop Spark
spark.stop()
