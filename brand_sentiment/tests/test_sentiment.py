import unittest

from pyspark.sql import SparkSession

from .. import BrandIdentification, SentimentIdentification


class TestSentimentIdentification(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.spark = SparkSession.builder \
            .appName("TestSentimentIdentification") \
            .config("spark.sql.broadcastTimeout", "36000") \
            .config("fs.s3.maxConnections", 100) \
            .getOrCreate()

        self.resources = "./brand_sentiment/tests/resources"

        self.brand_model_name = "xlnet_base"
        self.sent_model_name = "classifierdl_bertwiki_finance" \
                               "_sentiment_pipeline"

        self.partitions = 32

        self.brand = BrandIdentification(self.spark, self.brand_model_name,
                                         self.partitions)

        self.sentiment_columns = set(["text", "source_domain", "language",
                                      "date_publish", "positive", "neutral",
                                      "negative", "score",
                                      "Predicted_Entity_and_Sentiment"])

        super().__init__(*args, **kwargs)

    def setUp(self):
        self.sent = SentimentIdentification(self.spark, self.sent_model_name,
                                            self.partitions)
        return super().setUp()

    def test_valid_model_name(self):
        # Find another model to check this against.
        # This doesn't actually check anything.
        new_model = "classifierdl_bertwiki_finance_sentiment_pipeline"
        self.sent.model_name = new_model

        self.assertEqual(self.sent.model_name, new_model)

    def test_invalid_model_name(self):
        with self.assertRaises(TypeError) as a:
            self.sent.model_name = {"hello": "world!"}

        e = str(a.exception)
        self.assertEqual(e, "Model name is not a string.")

    def test_valid_partition_size(self):
        self.brand.partitions = 50
        self.assertEqual(self.brand.partitions, 50)

    def test_invalid_partition_size(self):
        with self.assertRaises(ValueError) as a1:
            self.brand.partitions = "Not too many records pls"

        with self.assertRaises(ValueError) as a2:
            self.brand.partitions = -1

        e1 = str(a1.exception)
        e2 = str(a2.exception)

        self.assertEqual(e1, "Partitions is not an integer.")
        self.assertEqual(e2, "Partitions is not greater than 0.")

    def test_predict_sentiment_valid_df(self):
        df = self.spark.read.parquet(f"{self.resources}/articles.parquet")
        brand_df = self.brand.predict_brand(df, False)
        sentiment_df = self.sent.predict_sentiment(brand_df)

        self.assertEqual(brand_df.count(), sentiment_df.count())
        self.assertSetEqual(set(sentiment_df.columns), self.sentiment_columns)


if __name__ == "__main__":
    unittest.main()
