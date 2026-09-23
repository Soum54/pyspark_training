from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

spark=(
    SparkSession.builder.appName("PySparkDataframeCreation")
    .master("local[*]")
    .getOrCreate()
)

#cricket_df=spark.read.csv("/home/kasm-user/pyspark_training/day1/data/cricket_players.csv")
cricket_df=spark.read.option("inferSchema",True).option("header",True).csv("/home/kasm-user/pyspark_training/day1/data/cricket_players.csv")
#cricket_df.printSchema()
#cricket_df.show()

#print("total count:",cricket_df.count())

#cricket_df.select("player_name","country").orderBy("player_name").show(2)
#cricket_df.select("player_name","country").sort("player_name").show()
#cricket_df.select("country","player_name").sort(["country","player_name"],ascending=[0,1]).show()
#cricket_df.select("country","player_name").sort(col("country").desc()).show()
#cricket_df.select("country","player_name").sort(col("country").desc(),col("player_name").desc()).show()
#cric_india_df=cricket_df.filter("country='India'" and "age>40")
#cric_india_df=cricket_df.where("country='India'" and "age>40")
#cric_india_df=cricket_df.select("player_name","COUNTRY","AGE").filter("country='India'" and "age>40")
#cric_india_df.show()

#bats_india_df=cricket_df.filter("role='Batsman'")
#bats_india_df.show()

#cric_country_total=cricket_df.groupby("country").count()
#can give 2 groupby col but not good output similar to pandas see synatx there
#cric_country_avg=cricket_df.groupby("country").round(avg("runs"),1)
#cric_country_avg=cricket_df.groupby("country").sum("runs")

cric_multiple_agg=cricket_df.groupBy("role").agg(round(avg("age"),2).alias("Avg_age"),
                                                 sum("runs").alias("Total_runs"),
                                                 max("wickets").alias("MaxWickets"),
                                                 count("*").alias("totalplayers"))
cric_multiple_agg.show()
