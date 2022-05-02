import unittest

from pyspark.sql import SparkSession, DataFrame

from .. import BrandIdentification


class TestBrandIdentification(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        self.spark = SparkSession.builder \
            .appName("TestBrandIdentification") \
            .config("spark.sql.broadcastTimeout", "36000") \
            .config("fs.s3.maxConnections", 100) \
            .getOrCreate()

        self.resources = "./brand_sentiment/tests/resources"

        self.model_name = "xlnet_base"
        self.partitions = 32

        self.brand_columns = set(["title", "main_text", "url",
                                  "source_domain", "date_publish",
                                  "language"])

        super().__init__(*args, **kwargs)

    def setUp(self):
        self.brand = BrandIdentification(self.spark, self.model_name,
                                         self.partitions)
        return super().setUp()

    def test_valid_model_name(self):
        model = "ner_conll_bert_base_cased"
        self.brand.model_name = model
        self.assertEqual(self.brand.model_name, model)

        model = "xlnet_base"
        self.brand.model_name = model
        self.assertEqual(self.brand.model_name, model)

    def test_invalid_model_name(self):
        with self.assertRaises(ValueError) as a:
            self.brand.model_name = "not-a-valid-model"

        e = str(a.exception)
        expected_message = "Model must be either 'xlnet_base'" \
                           " or 'ner_conll_bert_base_cased'."

        self.assertEqual(e, expected_message)

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

    def test_build_xlnet_pipeline(self):
        self.brand.model_name = "xlnet_base"
        # Need to find a better way to check this is the correct pipeline
        self.assertEqual(len(self.brand.pipeline.stages), 4)

    def test_build_conll_pipeline(self):
        self.brand.model_name = "ner_conll_bert_base_cased"
        # Need to find a better way to check this is the correct pipeline
        self.assertEqual(len(self.brand.pipeline.stages), 5)

    def test_predict_brand_xlnet_base(self):
        df = self.spark.read.parquet(f"{self.resources}/articles.parquet")
        self.brand.model_name = "xlnet_base"
        brand_df = self.brand.predict_brand(df)

        self.assertIsInstance(brand_df, DataFrame)
        self.assertEqual(brand_df.count(), df.count())
        self.assertSetEqual(set(brand_df.columns), self.brand_columns)

    def test_predict_brand_conll_bert(self):
        df = self.spark.read.parquet(f"{self.resources}/articles.parquet")
        self.brand.model_name = "ner_conll_bert_base_cased"
        brand_df = self.brand.predict_brand(df)

        self.assertIsInstance(brand_df, DataFrame)
        self.assertEqual(brand_df.count(), df.count())
        self.assertSetEqual(set(brand_df.columns), self.brand_columns)

    def test_predict_brand_with_filtering(self):
        df = self.spark.read.parquet(f"{self.resources}/articles.parquet")
        brand_df = self.brand.predict_brand(df, True)

        self.assertIsInstance(brand_df, DataFrame)
        self.assertLess(brand_df.count(), df.count())
        self.assertSetEqual(set(brand_df.columns), self.brand_columns)


if __name__ == "__main__":
    unittest.main()
