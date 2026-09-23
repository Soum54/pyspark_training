from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count


spark = (
    SparkSession.builder
    .appName("PySparkInstallationTest")
    .master("local[*]")
    .getOrCreate()
)

print("=" * 50)
print("PYSPARK INSTALLATION TEST")
print("=" * 50)

print("Spark Version :", spark.version)


data = [
    (101, "Laptop", "Electronics", 75000),
    (102, "Mobile", "Electronics", 30000),
    (103, "Chair", "Furniture", 10000),
    (104, "Table", "Furniture", 20000),
    (105, "TV", "Electronics", 50000)
]

columns = [
    "product_id",
    "product_name",
    "category",
    "price"
]


df = spark.createDataFrame(data, columns)


print("\n1. DataFrame")
df.show()
