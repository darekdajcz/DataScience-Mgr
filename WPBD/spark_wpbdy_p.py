from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum

spark = SparkSession.builder.appName("TestSpark").getOrCreate()

print("Sesja Spark działa!")
spark.stop()

spark = SparkSession.builder.appName("CustomerOrders").getOrCreate()

df = spark.read.csv("customer-orders.csv", header=False, inferSchema=True)

df = df.withColumnRenamed("_c0", "customer_id") \
       .withColumnRenamed("_c1", "product_id") \
       .withColumnRenamed("_c2", "order_amount")

df.show(5)

df = df.withColumn("order_amount", col("order_amount").cast("float"))

total_order_value = df.groupBy("customer_id").agg(_sum("order_amount").alias("total_order_value"))

total_order_value.show()

sorted_order_value = total_order_value.orderBy(col("total_order_value").desc())

sorted_order_value.show()
