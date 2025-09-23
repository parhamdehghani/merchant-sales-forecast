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
        .set("spark.executor.memory", "2g") \
        .set("spark.driver.memory", "4g") \
        .set("spark.sql.shuffle.partitions", "200")

    # For local development, use 'local[*]' master
    # In a production Docker environment, this would be overridden by a Kubernetes/YARN master
    conf = conf.setMaster("local[*]")

    spark = SparkSession.builder \
        .config(conf=conf) \
        .getOrCreate()

    return spark
