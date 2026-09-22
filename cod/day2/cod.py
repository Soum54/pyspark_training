from pyspark.sql import SparkSession

spark=SparkSession.builder.appName("CustomerOrdersLab").getOrCreate()

from pyspark.sql.functions import (col,to_date,sum,count,year,month,dayofmonth)
from pyspark.sql.types import (StructType,StructField,StringType,IntegerType)

dataschema=StructType(
    [
        StructField("order_id",IntegerType(),True),
        StructField("order_date",StringType(),True),
        StructField("customer_id",StringType(),True),
        StructField("customer_name",StringType(),True),
        StructField("product_category",StringType(),True),
        StructField("product_name",StringType(),True),
        StructField("brand",StringType(),True),
        StructField("unit_price",IntegerType(),True),
        StructField("quantity",IntegerType(),True),
        StructField("payment_mode",StringType(),True),
        StructField("city",StringType(),True)
    ]
)
df=spark.read.option("header","true").schema(dataschema).csv("/home/kasm-user/pyspark_training/day1/data/coddataset.csv")
df.printSchema()
df.show()

df1=df.withColumn("order_date",to_date(col("order_date"),"MM/dd/yyyy"))
df2=df1.withColumn("total_amount",col("unit_price")*col("quantity"))
df3=df2.withColumnRenamed("unit_price","price")
df4=df3.withColumn("order_year",year(col("order_date")))
df5=df4.withColumn("order_month",month(col("order_date")))
df6=df5.withColumn("order_day",dayofmonth(col("order_date")))

catgeory_sales=df6.groupBy("product_category").agg(sum("total_amount").alias("total_sales"))
catgeory_sales.show()

catgeory_sales.orderBy(col("total_sales").desc()).show()

brand_orders=df6.groupBy("brand").agg(count("order_id").alias("order_count"))
brand_orders.show()

city_sales=df6.groupBy("city").agg(count("total_amount").alias("city_sales"))
city_sales.show()

city_sales.orderBy(col("city_sales").desc()).show()

payment_sales=df6.groupBy("payment_mode").agg(count("total_amount").alias("payment_sales"))
payment_sales.show()

monthly_orders=df6.groupBy("order_month").agg(count("order_id").alias("total_orders"))
monthly_orders.show()

final_df=df6.select(
    "order_id","order_date","customer_name","product_category","product_name",
    "brand","price","quantity","total_amount","payment_mode","city"
)
final_df.show(truncate=False)
