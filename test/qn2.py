from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, DoubleType
)
from pyspark.sql import functions as F


# =========================================================
# 1. DEFINE TELECOM SCHEMA
# =========================================================

def define_schema():
    schema = StructType([
        StructField("event_id", StringType(), True),
        StructField("user_id", IntegerType(), True),
        StructField("plan", StringType(), True),
        StructField("city", StringType(), True),
        StructField("event_date", StringType(), True),
        StructField("data_mb", DoubleType(), True),
        StructField("voice_minutes", DoubleType(), True),
        StructField("sms_count", IntegerType(), True),
        StructField("roaming", StringType(), True),
        StructField("device_type", StringType(), True),
        StructField("dropped_calls", IntegerType(), True),
        StructField("latency_ms", IntegerType(), True)
    ])

    return schema


# =========================================================
# 2. LOAD TELECOM CSV
# =========================================================

def load_data(spark, path, schema):
    df = spark.read.csv(
        path,
        header=True,
        schema=schema
    )

    return df


# =========================================================
# 3. PARSE EVENT DATE
# =========================================================

def parse_event_date(df):
    df = df.withColumn(
        "event_date",
        F.to_date("event_date")
    )

    return df


# =========================================================
# 4. ADD TOTAL ACTIVITY
# =========================================================

def add_total_activity(df):
    df = df.withColumn(
        "total_activity",
        F.col("voice_minutes") + F.col("sms_count")
    )

    return df


# =========================================================
# 5. ADD USAGE SCORE
# =========================================================

def add_usage_score(df):
    df = df.withColumn(
        "usage_score",
        F.col("data_mb")
        + (F.col("voice_minutes") * 0.5)
        + (F.col("sms_count") * 0.1)
    )

    return df


# =========================================================
# 6. FILTER ROAMING EVENTS
# =========================================================

def filter_roaming_events(df):
    return df.filter(
        F.col("roaming") == "Y"
    )


# =========================================================
# 7. TOP N USERS BY DATA
# =========================================================

def top_n_users_by_data(df, n):
    result = df.groupBy("user_id") \
        .agg(
            F.sum("data_mb").alias("total_data_mb")
        ) \
        .orderBy(
            F.col("total_data_mb").desc(),
            F.col("user_id").asc()
        ) \
        .limit(n)

    return result


# =========================================================
# 8. AVERAGE LATENCY BY CITY
# =========================================================

def avg_latency_by_city(df):
    result = df.groupBy("city") \
        .agg(
            F.avg("latency_ms").alias("avg_latency_ms")
        )

    return result


# =========================================================
# 9. MOST USED PLAN
# =========================================================

def most_used_plan(df):
    result = df.groupBy("plan") \
        .agg(
            F.count("*").alias("plan_count")
        ) \
        .orderBy(
            F.col("plan_count").desc(),
            F.col("plan").asc()
        ) \
        .limit(1) \
        .collect()

    return result[0]["plan"]


# =========================================================
# 10. DROPPED CALL RATE BY PLAN
# =========================================================

def dropped_call_rate_by_plan(df):
    result = df.groupBy("plan") \
        .agg(
            F.count("*").alias("total_calls"),
            F.sum("dropped_calls").alias("dropped_total")
        ) \
        .withColumn(
            "dropped_call_rate",
            F.col("dropped_total") / F.col("total_calls")
        ) \
        .select(
            "plan",
            "dropped_call_rate"
        )

    return result


# =========================================================
# 11. HIGH LATENCY EVENTS
# =========================================================

def high_latency_events(df, threshold):
    return df.filter(
        F.col("latency_ms") > threshold
    )


# =========================================================
# 12. COUNT USERS PER CITY
# =========================================================

def count_users_per_city(df):
    result = df.groupBy("city") \
        .agg(
            F.countDistinct("user_id").alias("user_count")
        )

    return result


# =========================================================
# 13. DAILY DATA TREND
# =========================================================

def daily_data_trend(df):
    result = df.groupBy("event_date") \
        .agg(
            F.sum("data_mb").alias("daily_data_mb")
        ) \
        .orderBy(
            F.col("event_date").asc()
        )

    return result


# =========================================================
# 14. TOP DEVICE BY USAGE SCORE
# =========================================================

def top_device_by_usage_score(df):
    df = add_usage_score(df)

    result = df.groupBy("device_type") \
        .agg(
            F.avg("usage_score").alias("avg_usage_score")
        ) \
        .orderBy(
            F.col("avg_usage_score").desc(),
            F.col("device_type").asc()
        ) \
        .limit(1) \
        .collect()

    return result[0]["device_type"]


# =========================================================
# 15. LIST CITIES
# =========================================================

def list_cities(df):
    cities = df.select("city") \
        .distinct() \
        .collect()

    result = []

    for row in cities:
        result.append(row["city"])

    return sorted(result)


# =========================================================
# 16. EVENTS IN DATE RANGE
# =========================================================

def events_in_date_range(df, start, end):
    return df.filter(
        F.col("event_date").between(start, end)
    )


# =========================================================
# 17. FLAG NETWORK RISK
# =========================================================

def flag_network_risk(df, latency_threshold, drop_threshold):
    df = df.withColumn(
        "is_risky",
        (F.col("latency_ms") > latency_threshold)
        |
        (F.col("dropped_calls") > drop_threshold)
    )

    return df


# =========================================================
# 18. TOP N RISKY USERS
# =========================================================

def top_n_risky_users(df, n):
    result = df.filter(
        F.col("is_risky") == True
    ) \
        .groupBy("user_id") \
        .agg(
            F.count("*").alias("risky_event_count")
        ) \
        .orderBy(
            F.col("risky_event_count").desc(),
            F.col("user_id").asc()
        ) \
        .limit(n)

    return result


# =========================================================
# 19. AVERAGE DATA PER PLAN
# =========================================================

def avg_data_per_plan(df):
    result = df.groupBy("plan") \
        .agg(
            F.avg("data_mb").alias("avg_data_mb")
        )

    return result


# =========================================================
# 20. GET HEAVIEST USER
# =========================================================

def get_heaviest_user(df):
    result = df.groupBy("user_id") \
        .agg(
            F.sum("data_mb").alias("total_data_mb")
        ) \
        .orderBy(
            F.col("total_data_mb").desc(),
            F.col("user_id").asc()
        ) \
        .limit(1) \
        .collect()

    return (
        result[0]["user_id"],
        result[0]["total_data_mb"]
    )


# =========================================================
# 21. PROFILE SCHEMA
# =========================================================

def define_profile_schema():
    schema = StructType([
        StructField("user_id", IntegerType(), True),
        StructField("first_name", StringType(), True),
        StructField("last_name", StringType(), True),
        StructField("full_name", StringType(), True),
        StructField("plan_id", StringType(), True)
    ])

    return schema


# =========================================================
# 22. PLAN SCHEMA
# =========================================================

def define_plan_schema():
    schema = StructType([
        StructField("plan_id", StringType(), True),
        StructField("plan_name", StringType(), True)
    ])

    return schema


# =========================================================
# 23. LOAD INLINE PROFILE + PLAN DATA
# =========================================================

def load_inline_profile_data(spark, profile_schema, plan_schema):

    profiles_rows = [
        (201, "Aarav", "Iyer", None, "P1"),
        (202, "Diya", "Sharma", "Diya Sharma", "P2"),
        (203, "Kabir", "Mehta", None, "P2"),
        (204, "Meera", "Nair", None, "P3"),
        (205, "Rohan", "Singh", "Rohan S.", "P3"),
        (206, "Sara", "Khan", None, "P99")
    ]

    plans_rows = [
        ("P1", "Basic"),
        ("P2", "Plus"),
        ("P3", "Premium")
    ]

    profiles_df = spark.createDataFrame(
        profiles_rows,
        schema=profile_schema
    )

    plans_df = spark.createDataFrame(
        plans_rows,
        schema=plan_schema
    )

    return profiles_df, plans_df


# =========================================================
# 24. JOIN PROFILE + PLAN
# =========================================================

def join_profile_plan(profiles_df, plans_df):

    result = profiles_df.join(
        plans_df,
        on="plan_id",
        how="left"
    )

    return result.select(
        "user_id",
        "first_name",
        "last_name",
        "full_name",
        "plan_id",
        "plan_name"
    )


# =========================================================
# 25. ENRICH FULL NAME
# =========================================================

def enrich_full_name(profile_joined_df):

    result = profile_joined_df.withColumn(
        "full_name",
        F.when(
            F.col("full_name").isNull(),
            F.concat_ws(
                " ",
                F.col("first_name"),
                F.col("last_name")
            )
        ).otherwise(
            F.col("full_name")
        )
    )

    return result


# =========================================================
# 26. ADD EVENT EPOCH SECONDS
# =========================================================

def add_event_epoch_seconds(df):

    result = df.withColumn(
        "event_epoch_seconds",
        F.unix_timestamp(
            F.col("event_date")
        )
    )

    return result


# =========================================================
# 27. ADD EVENT TIMESTAMP FROM EPOCH
# =========================================================

def add_event_ts_from_epoch(epoch_df):

    result = epoch_df.withColumn(
        "event_ts",
        F.from_unixtime(
            F.col("event_epoch_seconds")
        ).cast("timestamp")
    )

    return result
# =========================================================
# DRIVER CODE
# =========================================================

spark = SparkSession.builder \
    .appName("Telecom Usage Intelligence") \
    .getOrCreate()


# ---------------------------------------------------------
# MAIN TELECOM DATA
# ---------------------------------------------------------

print("========== LOAD DATA ==========")

schema = define_schema()

df = load_data(
    spark,
    "data/telecom_usage.csv",
    schema
)

df = parse_event_date(df)

df.show()


# ---------------------------------------------------------
# 1. TOTAL ACTIVITY
# ---------------------------------------------------------

print("========== TOTAL ACTIVITY ==========")

df1 = add_total_activity(df)
df1.show()


# ---------------------------------------------------------
# 2. USAGE SCORE
# ---------------------------------------------------------

print("========== USAGE SCORE ==========")

df2 = add_usage_score(df)
df2.show()


# ---------------------------------------------------------
# 3. ROAMING EVENTS
# ---------------------------------------------------------

print("========== ROAMING EVENTS ==========")

df3 = filter_roaming_events(df)
df3.show()


# ---------------------------------------------------------
# 4. TOP 5 USERS BY DATA
# ---------------------------------------------------------

print("========== TOP 5 USERS BY DATA ==========")

df4 = top_n_users_by_data(df, 5)
df4.show()


# ---------------------------------------------------------
# 5. AVERAGE LATENCY BY CITY
# ---------------------------------------------------------

print("========== AVERAGE LATENCY BY CITY ==========")

df5 = avg_latency_by_city(df)
df5.show()


# ---------------------------------------------------------
# 6. MOST USED PLAN
# ---------------------------------------------------------

print("========== MOST USED PLAN ==========")

result1 = most_used_plan(df)
print(result1)


# ---------------------------------------------------------
# 7. DROPPED CALL RATE BY PLAN
# ---------------------------------------------------------

print("========== DROPPED CALL RATE BY PLAN ==========")

df6 = dropped_call_rate_by_plan(df)
df6.show()


# ---------------------------------------------------------
# 8. HIGH LATENCY EVENTS
# ---------------------------------------------------------

print("========== HIGH LATENCY EVENTS ==========")

df7 = high_latency_events(df, 100)
df7.show()


# ---------------------------------------------------------
# 9. USERS PER CITY
# ---------------------------------------------------------

print("========== USERS PER CITY ==========")

df8 = count_users_per_city(df)
df8.show()


# ---------------------------------------------------------
# 10. DAILY DATA TREND
# ---------------------------------------------------------

print("========== DAILY DATA TREND ==========")

df9 = daily_data_trend(df)
df9.show()


# ---------------------------------------------------------
# 11. TOP DEVICE BY USAGE SCORE
# ---------------------------------------------------------

print("========== TOP DEVICE BY USAGE SCORE ==========")

result2 = top_device_by_usage_score(df)
print(result2)


# ---------------------------------------------------------
# 12. LIST CITIES
# ---------------------------------------------------------

print("========== LIST OF CITIES ==========")

result3 = list_cities(df)
print(result3)


# ---------------------------------------------------------
# 13. EVENTS IN DATE RANGE
# ---------------------------------------------------------

print("========== EVENTS IN DATE RANGE ==========")

df10 = events_in_date_range(
    df,
    "2025-01-01",
    "2025-01-31"
)

df10.show()


# ---------------------------------------------------------
# 14. NETWORK RISK
# ---------------------------------------------------------

print("========== NETWORK RISK ==========")

df11 = flag_network_risk(
    df,
    100,
    2
)

df11.show()


# ---------------------------------------------------------
# 15. TOP 5 RISKY USERS
# ---------------------------------------------------------

print("========== TOP 5 RISKY USERS ==========")

df12 = top_n_risky_users(
    df11,
    5
)

df12.show()


# ---------------------------------------------------------
# 16. AVERAGE DATA PER PLAN
# ---------------------------------------------------------

print("========== AVERAGE DATA PER PLAN ==========")

df13 = avg_data_per_plan(df)
df13.show()


# ---------------------------------------------------------
# 17. HEAVIEST USER
# ---------------------------------------------------------

print("========== HEAVIEST USER ==========")

result4 = get_heaviest_user(df)
print(result4)


# =========================================================
# PROFILE + PLAN DATA
# =========================================================

print("========== PROFILE + PLAN DATA ==========")

profile_schema = define_profile_schema()
plan_schema = define_plan_schema()


profiles_df, plans_df = load_inline_profile_data(
    spark,
    profile_schema,
    plan_schema
)


print("Profiles:")
profiles_df.show()


print("Plans:")
plans_df.show()


# ---------------------------------------------------------
# 18. JOIN PROFILE + PLAN
# ---------------------------------------------------------

print("========== PROFILE + PLAN JOIN ==========")

joined_df = join_profile_plan(
    profiles_df,
    plans_df
)

joined_df.show()


# ---------------------------------------------------------
# 19. ENRICH FULL NAME
# ---------------------------------------------------------

print("========== ENRICHED FULL NAME ==========")

enriched_df = enrich_full_name(
    joined_df
)

enriched_df.show()


# ---------------------------------------------------------
# 20. EVENT EPOCH SECONDS
# ---------------------------------------------------------

print("========== EVENT EPOCH SECONDS ==========")

epoch_df = add_event_epoch_seconds(df)

epoch_df.show()


# ---------------------------------------------------------
# 21. EVENT TIMESTAMP FROM EPOCH
# ---------------------------------------------------------

print("========== EVENT TIMESTAMP ==========")

timestamp_df = add_event_ts_from_epoch(
    epoch_df
)

timestamp_df.show()


# =========================================================
# STOP SPARK
# =========================================================

spark.stop()
