import sparknlp
import logging
import os

from pyspark.sql import SparkSession

from brand_sentiment import AWSInterface, BrandIdentification, \
    SentimentIdentification

logging.basicConfig(level=logging.INFO)

extraction_bucket = os.environ.get("EXTRACTION_BUCKET_NAME")
sentiment_bucket = os.environ.get("SENTIMENT_BUCKET_NAME")
extraction_date = os.environ.get("EXTRACTION_DATE")

sentiment_model = os.environ.get("SENTIMENT_MODEL")
ner_model = os.environ.get("NER_MODEL")

partitions = int(os.environ.get("DATAFRAME_PARTITIONS"))

limit = os.environ.get("DATAFRAME_LIMIT")
limit = int(limit) if limit is not None else None

spark = SparkSession.builder \
    .appName("ArticleSentimentiser") \
    .config("spark.sql.broadcastTimeout", "36000") \
    .config("fs.s3.maxConnections", 100) \
    .getOrCreate()

logging.info(f"Running Apache Spark v{spark.version}"
             f" and Spark NLP v{sparknlp.version()}")

aws_interface = AWSInterface(spark, extraction_bucket, sentiment_bucket,
                             partitions, extraction_date)

brand_identifier = BrandIdentification(spark, ner_model, partitions)
sentimentiser = SentimentIdentification(spark, sentiment_model, partitions)

df = aws_interface.download(limit)
logging.info(f"Successfully downloaded datafram with {df.count()}"
             f" rows and {df.rdd.getNumPartitions()} partitions.")

brand_df = brand_identifier.predict_brand(df, True)
logging.info("NER Analysis complete.")

brand_sentiment_df = sentimentiser.predict_sentiment(brand_df)
logging.info("Sentiment Analysis complete.")

aws_interface.upload(brand_sentiment_df)
logging.info("AWS Upload complete.")
