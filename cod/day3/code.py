from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark=SparkSession.builder.appName("EmployeeData").getOrCreate()
emp_path="day1/data/employee.csv"
per_path="day1/data/performance.csv"
employees_df=spark.read.csv(emp_path,header=True,inferSchema=True)
performance_df=spark.read.csv(per_path,header=True,inferSchema=True)
employees_df=employees_df.withColumn("employee_id",col("employee_id").cast(IntegerType()))
performance_df=performance_df.withColumn("employee_id",col("employee_id").cast(IntegerType()))
#employees_df.show()
#performance_df.show()

joined_df=employees_df.join(performance_df,on="employee_id",how="inner")
#joined_df.show()

enrich_df=joined_df.withColumn("full_name",when(col("full_name").isNull(),
            concat_ws(" ",col("first_name"),col("last_name"))).otherwise(col("full_name")))
#enrich_df.show()

fil_df=enrich_df.filter(col("rating")=="Poor").select("full_name","department","rating")
fil_df.show()
