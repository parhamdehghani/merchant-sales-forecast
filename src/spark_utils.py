from pyspark.sql import SparkSession
from pyspark.conf import SparkConf

def get_or_create_spark_session(app_name="MerchantSalesForecast") -> SparkSession:
    """
    Gets an existing SparkSession or creates a new one if it doesn't exist.
    Configures Spark for optimal performance and local execution.

    Args:
        app_name (str): The name of the Spark application.

    Returns:
        SparkSession: An instance of SparkSession.
    """
    conf = SparkConf() \
        .setAppName(app_name) \
        .set("spark.executor.memory", "1g") \
        .set("spark.driver.memory", "2g") \
        .set("spark.executor.cores", "1") \
        .set("spark.sql.shuffle.partitions", "4") \
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .set("spark.sql.adaptive.enabled", "true") \
        .set("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .set("spark.local.dir", "/tmp/spark") \
        .set("spark.ui.enabled", "false") \
        .set("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
        .set("spark.driver.host", "127.0.0.1") \
        .set("spark.driver.bindAddress", "127.0.0.1") \
        .set("spark.executor.host", "127.0.0.1") \
        .set("spark.network.timeout", "800s") \
        .set("spark.executor.heartbeatInterval", "60s") \
        .set("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .set("spark.driver.port", "0") \
        .set("spark.executor.port", "0") \
        .set("spark.blockManager.port", "0") \
        .set("spark.scheduler.maxRegisteredResourcesWaitingTime", "30s") \
        .set("spark.scheduler.minRegisteredResourcesRatio", "1.0")

    # For local development, use 'local[*]' master
    # In a production Docker environment, this would be overridden by a Kubernetes/YARN master
    conf = conf.setMaster("local[*]")

    spark = SparkSession.builder \
        .config(conf=conf) \
        .getOrCreate()

    return spark
