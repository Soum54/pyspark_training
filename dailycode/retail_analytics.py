from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import (StructType,StructField,IntegerType,StringType,DoubleType)

spark=(
    SparkSession.builder.appName("retail_analytics").master("local[*]").getOrCreate()
)
schema=StructType([
    StructField("order_id",IntegerType(),True),
    StructField("order_date",StringType(),True),
    StructField("customer_id",StringType(),True),
    StructField("customer_name",StringType(),True),
    StructField("product", StringType(), True),
    StructField("category", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("unit_price", DoubleType(), True),
    StructField("payment_method", StringType(), True),
    StructField("store_city", StringType(), True),
    StructField("status", StringType(), True),
    StructField("discount", IntegerType(), True)
])
ret_path="day1/data/orders.csv"
orders_df=spark.read.schema(schema).csv(ret_path,header=True)
#orders_df.show()
orders_df=orders_df.withColumn("order_date",to_date("order_date","yyyy-MM-dd"))
# orders_df.printSchema()
# orders_df.show()

# orders_df=orders_df.withColumn("order_year",year("order_date"))\
# .withColumn("order_month",month("order_date"))\
# .withColumn("current_year",lit(2026))
# orders_df.show()

orders_df=orders_df.withColumn("formatted_date",date_format("order_date","dd-MMM-yyyy"))
orders_df.show()

orders_df.withColumn("days_since_order",datediff(current_date(),col("order_date"))).show()

