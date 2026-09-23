from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark=(
    SparkSession.builder.appName("employee_analytics").master("local[*]")
        .getOrCreate()
)
emp_path="day1/data/employees.csv"
dept_path="day1/data/departments.csv"
empdf=spark.read.option("inferSchema",True).option("header",True).csv(emp_path)
deptdf=spark.read.option("inferSchema",True).option("header",True).csv(dept_path)
#empdf.show()
#deptdf.show()
#print(empdf.count())
#print(deptdf.count())
#emp_dept_df=empdf.join(deptdf,"department_id",how="inner")
#emp_dept_df=empdf.join(deptdf,"department_id",how="leftsemi")
emp_dept_df=empdf.join(deptdf,"department_id",how="leftanti")
emp_dept_df.show()
