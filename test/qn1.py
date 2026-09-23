from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *


# 1. Define Schema
def define_schema():
    schema = StructType([
        StructField("txn_id", StringType(), True),
        StructField("order_date", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("region", StringType(), True),
        StructField("channel", StringType(), True),
        StructField("category", StringType(), True),
        StructField("product_id", StringType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("unit_price", DoubleType(), True),
        StructField("discount_rate", DoubleType(), True),
        StructField("returned", StringType(), True),
        StructField("ship_date", StringType(), True),
        StructField("delivery_date", StringType(), True)
    ])

    return schema


# 2. Load Data
def load_data(spark, path, schema):
    df = spark.read.csv(
        path,
        header=True,
        schema=schema
    )

    return df


# 3. Parse Dates
def parse_dates(df):
    df = df.withColumn("order_date", to_date("order_date"))
    df = df.withColumn("ship_date", to_date("ship_date"))
    df = df.withColumn("delivery_date", to_date("delivery_date"))

    return df


# 4. Add Gross Amount
def add_gross_amount(df):
    df = df.withColumn(
        "gross_amount",
        col("quantity") * col("unit_price")
    )

    return df


# 5. Add Net Amount
def add_net_amount(df):
    if "gross_amount" not in df.columns:
        df = df.withColumn(
            "gross_amount",
            col("quantity") * col("unit_price")
        )

    df = df.withColumn(
        "net_amount",
        col("gross_amount") * (1 - col("discount_rate"))
    )

    return df


# 6. Add Delivery Days
def add_delivery_days(df):
    df = df.withColumn(
        "delivery_days",
        datediff(
            col("delivery_date"),
            col("ship_date")
        )
    )

    return df


# 7. Flag On-Time Delivery
def flag_on_time_delivery(df, max_days):
    df = df.withColumn(
        "is_on_time",
        col("delivery_days") <= max_days
    )

    return df


# 8. Filter Returned Orders
def filter_returned_orders(df):
    return df.filter(
        col("returned") == "Y"
    )


# 9. Filter By Region
def filter_by_region(df, region):
    return df.filter(
        col("region") == region
    )


# 10. Top N Customers By Spend
def top_n_customers_by_spend(df, n):
    df = add_net_amount(df)

    result = df.groupBy("customer_id") \
        .agg(
            sum("net_amount").alias("total_spend")
        ) \
        .orderBy(
            col("total_spend").desc(),
            col("customer_id").asc()
        ) \
        .limit(n)

    return result


# 11. Revenue By Category
def revenue_by_category(df):
    df = add_net_amount(df)

    result = df.groupBy("category") \
        .agg(
            sum("net_amount").alias("total_revenue")
        )

    return result


# 12. Top Category By Revenue
def top_category_by_revenue(df):
    df = add_net_amount(df)

    result = df.groupBy("category") \
        .agg(
            sum("net_amount").alias("total_revenue")
        ) \
        .orderBy(
            col("total_revenue").desc(),
            col("category").asc()
        ) \
        .limit(1) \
        .collect()

    return result[0]["category"]


# 13. Average Discount By Channel
def avg_discount_by_channel(df):
    result = df.groupBy("channel") \
        .agg(
            avg("discount_rate").alias("avg_discount")
        )

    return result


# 14. Daily Revenue Trend
def daily_revenue_trend(df):
    df = add_net_amount(df)

    result = df.groupBy("order_date") \
        .agg(
            sum("net_amount").alias("daily_revenue")
        ) \
        .orderBy(
            col("order_date").asc()
        )

    return result


# 15. High Value Orders
def high_value_orders(df, threshold):
    df = add_net_amount(df)

    return df.filter(
        col("net_amount") > threshold
    )


# 16. Count Late Deliveries
def count_late_deliveries(df, max_days):
    df = add_delivery_days(df)

    result = df.filter(
        col("delivery_days") > max_days
    ).count()

    return result


# 17. Return Rate By Category
def return_rate_by_category(df):
    result = df.groupBy("category") \
        .agg(
            count("*").alias("total_cnt"),
            sum(
                when(col("returned") == "Y", 1)
                .otherwise(0)
            ).alias("returned_cnt")
        ) \
        .withColumn(
            "return_rate",
            col("returned_cnt") / col("total_cnt")
        )

    return result


# 18. Best Selling Product
def best_selling_product(df):
    result = df.groupBy("product_id") \
        .agg(
            sum("quantity").alias("total_qty")
        ) \
        .orderBy(
            col("total_qty").desc(),
            col("product_id").asc()
        ) \
        .limit(1) \
        .collect()

    return (
        result[0]["product_id"],
        result[0]["total_qty"]
    )


# 19. List Channels
def list_channels(df):
    channels = df.select("channel") \
        .distinct() \
        .collect()

    result = []

    for row in channels:
        result.append(row["channel"])

    return sorted(result)


# 20. Orders In Date Range
def orders_in_date_range(df, start, end):
    return df.filter(
        col("order_date").between(start, end)
    )
# ---------------- DRIVER CODE ----------------

# 1. Create Spark Session
spark = SparkSession.builder \
    .appName("Retail Commerce Operations") \
    .getOrCreate()


# 2. Define schema
schema = define_schema()


# 3. Load CSV
df = load_data(
    spark,
    "data/commerce.csv",
    schema
)


# 4. Parse dates
df = parse_dates(df)


# 5. Display original data
print("Original Data:")
df.show()


# 6. Add gross amount
print("Gross Amount:")
df1 = add_gross_amount(df)
df1.show()


# 7. Add net amount
print("Net Amount:")
df2 = add_net_amount(df)
df2.show()


# 8. Add delivery days
print("Delivery Days:")
df3 = add_delivery_days(df)
df3.show()


# 9. Flag on-time delivery
print("On-Time Delivery:")
df4 = flag_on_time_delivery(df3, 5)
df4.show()


# 10. Returned orders
print("Returned Orders:")
df5 = filter_returned_orders(df)
df5.show()


# 11. Filter by region
print("Orders By Region:")
df6 = filter_by_region(df, "South")
df6.show()


# 12. Top N customers
print("Top 5 Customers By Spend:")
df7 = top_n_customers_by_spend(df, 5)
df7.show()


# 13. Revenue by category
print("Revenue By Category:")
df8 = revenue_by_category(df)
df8.show()


# 14. Top category by revenue
print("Top Category By Revenue:")
result1 = top_category_by_revenue(df)
print(result1)


# 15. Average discount by channel
print("Average Discount By Channel:")
df9 = avg_discount_by_channel(df)
df9.show()


# 16. Daily revenue trend
print("Daily Revenue Trend:")
df10 = daily_revenue_trend(df)
df10.show()


# 17. High value orders
print("High Value Orders:")
df11 = high_value_orders(df, 5000)
df11.show()


# 18. Count late deliveries
print("Number of Late Deliveries:")
result2 = count_late_deliveries(df, 5)
print(result2)


# 19. Return rate by category
print("Return Rate By Category:")
df12 = return_rate_by_category(df)
df12.show()


# 20. Best selling product
print("Best Selling Product:")
result3 = best_selling_product(df)
print(result3)


# 21. List channels
print("Available Channels:")
result4 = list_channels(df)
print(result4)


# 22. Orders in date range
print("Orders In Date Range:")
df13 = orders_in_date_range(
    df,
    "2025-01-01",
    "2025-01-31"
)
df13.show()


# 23. Stop Spark
spark.stop()
