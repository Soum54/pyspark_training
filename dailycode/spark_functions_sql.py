from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark=(
    SparkSession.builder.appName("employee_analytics").master("local[*]")
        .getOrCreate()
)
emp_path="day1/data/employees.csv"
#dept_path="day1/data/departments.csv"
#read as dataframe
empdf=spark.read.csv(emp_path,header=True,inferSchema=True)
#deptdf=spark.read.csv(dept_path,header=True,inferSchema=True)
#deptdf.show()
#deptdf.show()
#read as rdd
# emprdd=spark.sparkContext.textFile(emp_path)
# print(emprdd.collect())

# deptrdd=spark.sparkContext.textFile(dept_path)
# print(deptrdd.collect())

# empdf.createTempView("employees")
# spark.sql("select * from employees").show()

# empdf.createOrReplaceTempView("employees")
# spark.sql("select * from employees").show()

#empdf.createOrReplaceTempView("employee")
#spark.sql("select first_name,last_name,salary from employee where salary>4000").show()
#empdf.select("first_name","last_name","salary").filter("salary>4000").show()
#empdf.createOrReplaceTempView("employee")
# spark.sql("select department_id,sum(salary) as totalsal from employee where salary>7000 group by department_id order by totalsal desc").show()
df=empdf.select("department_id","salary").filter("salary>7000").groupby("department_id").agg(sum("salary").alias("Totalsal")).orderBy("Totalsal",ascending=False)
print(type(df))
df.show()
