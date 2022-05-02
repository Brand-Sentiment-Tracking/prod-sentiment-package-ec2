import unittest

from pyspark.sql import SparkSession
from datetime import datetime, timedelta

from .. import AWSInterface


class TestAWSInterface(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.spark = SparkSession.builder \
            .appName("TestAWSInterface") \
            .config("spark.sql.broadcastTimeout", "36000") \
            .config("fs.s3.maxConnections", 100) \
            .getOrCreate()

        self.resources = "./brand_sentiment/tests/resources"

        self.unit_test_bucket = "brand-sentiment-unit-testing"

        self.extraction_bucket = f"{self.unit_test_bucket}/sent/downloads"
        self.sentiment_bucket = f"{self.unit_test_bucket}/sent/uploads"

        self.partitions = 32
        self.extraction_date = datetime(2022, 4, 26)

        super().__init__(*args, **kwargs)

    def setUp(self):
        self.aws = AWSInterface(self.spark, self.extraction_bucket,
                                self.sentiment_bucket, self.partitions,
                                self.extraction_date)

        return super().setUp()

    def test_valid_extraction_bucket(self):
        valid_bucket = "extracted-news-articles"
        self.aws.extraction_bucket = valid_bucket

        self.assertEqual(self.aws.extraction_bucket, valid_bucket)

    def test_invalid_extraction_bucket(self):
        with self.assertRaises(TypeError) as a:
            self.aws.extraction_bucket = 123

        self.assertEqual(str(a.exception), "Bucket name is not a string.")

    def test_valid_sentiment_bucket(self):
        valid_bucket = "processed-news-articles"
        self.aws.sentiment_bucket = valid_bucket

        self.assertEqual(self.aws.sentiment_bucket, valid_bucket)

    def test_invalid_sentiment_bucket(self):
        with self.assertRaises(TypeError) as a:
            self.aws.sentiment_bucket = list()

        self.assertEqual(str(a.exception), "Bucket name is not a string.")

    def test_valid_partition_size(self):
        self.aws.partitions = 50
        self.assertEqual(self.aws.partitions, 50)

    def test_invalid_partition_size(self):
        with self.assertRaises(TypeError) as a1:
            self.aws.partitions = "Not too many records pls"

        with self.assertRaises(ValueError) as a2:
            self.aws.partitions = -1

        e1 = str(a1.exception)
        e2 = str(a2.exception)

        self.assertEqual(e1, "Partitions is not an integer.")
        self.assertEqual(e2, "Partitions is not greater than 0.")

    def test_valid_extraction_date_string(self):
        self.aws.extraction_date = "2021-12-01"
        self.assertEqual(self.aws.extraction_date, "2021-12-01")

    def test_valid_extraction_date_datetime(self):
        self.aws.extraction_date = datetime(2021, 12, 1)
        self.assertEqual(self.aws.extraction_date, "2021-12-01")

    def test_invalid_extraction_date_bad_type(self):
        with self.assertRaises(TypeError) as a:
            self.aws.extraction_date = {"hello": "word!"}

        e = str(a.exception)

        self.assertEqual(e, "Extraction date is not a string or datetime.")

    def test_invalid_extraction_date_malformed_string(self):
        with self.assertRaises(ValueError) as a:
            self.aws.extraction_date = "2021-123-01"

        e = str(a.exception)

        self.assertEqual(e, "Extraction date isn't ISO format.")

    def test_invalid_extraction_date_bad_day(self):
        with self.assertRaises(ValueError) as a:
            self.aws.extraction_date = datetime.now() + timedelta(days=1)

        e = str(a.exception)

        self.assertEqual(e, "Extraction date is in the future.")

    def test_extraction_bucket_partition_url(self):
        self.aws.extraction_bucket = "the-extraction-bucket"
        self.aws.extraction_date = datetime(2021, 12, 1)
        self.aws.language = "xyz"

        expected_url = "s3a://the-extraction-bucket/" \
                       "date_crawled=2021-12-01/" \
                       "language=xyz/"

        self.assertEqual(self.aws.extraction_url, expected_url)

    def test_sentiment_bucket_url(self):
        self.aws.sentiment_bucket = "the-sentiment-bucket"
        expected_url = "s3a://the-sentiment-bucket/"

        self.assertEqual(self.aws.sentiment_url, expected_url)

    def test_download_parquet_partition_to_spark(self):
        df = self.aws.download()

        self.assertEqual(df.count(), 9789)
        self.assertEqual(len(df.columns), 6)

    def test_upload_spark_dataframe_to_parquet(self):
        df = self.spark.read.parquet(f"{self.resources}/articles.parquet")
        self.aws.upload(df)


if __name__ == "__main__":
    unittest.main()
