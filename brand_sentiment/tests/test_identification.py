import unittest

from datetime import datetime

from pyspark.sql import SparkSession, DataFrame # noqa

from pyspark.ml import Pipeline

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import XlnetForTokenClassification, \
    Tokenizer, BertEmbeddings, NerDLModel, NerConverter

from .. import AWSInterface, BrandIdentification


class TestBrandIdentification(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        jslnlp_package = "com.johnsnowlabs.nlp:spark-nlp_2.12:3.4.2"

        self.spark = SparkSession.builder \
            .appName("TestBrandIdentification") \
            .config('spark.jars.packages', jslnlp_package) \
            .config("spark.driver.memory", "10g") \
            .config("spark.sql.broadcastTimeout", "36000") \
            .config("fs.s3.maxConnections", 100) \
            .getOrCreate()

        self.resources = "./brand_sentiment/tests/resources"

        self.unit_test_bucket = "brand-sentiment-unit-testing"

        self.extraction_bucket = f"{self.unit_test_bucket}/sent/downloads"
        self.sentiment_bucket = f"{self.unit_test_bucket}/sent/uploads"

        self.extraction_date = datetime(2022, 4, 3)

        self.model_name = "xlnet_base"
        self.partitions = 32

        self.brand_columns = set(["text", "source_domain", "date_publish",
                                  "language", "entities"])

        aws = AWSInterface(self.spark, self.extraction_bucket,
                           self.sentiment_bucket, self.partitions,
                           self.extraction_date)

        self.df = aws.download(limit=100)

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

    """
    Cannot run these unit tests in GitHub because the models are too large.
    Ideally we use a self-hosted runner for this job.
    These tests have been run and passed on AWS.
    """

    """
    def test_build_xlnet_pipeline(self):
        self.brand.model_name = "xlnet_base"
        self.brand.build_pipeline()
        stages = self.brand.pipeline.getStages()

        instances = [
            DocumentAssembler,
            Tokenizer,
            XlnetForTokenClassification,
            NerConverter
        ]

        for stage, instance in zip(stages, instances):
            self.assertIsInstance(stage, instance)
        
        self.assertEqual(len(stages), len(instances))

    def test_build_conll_pipeline(self):
        self.brand.model_name = "ner_conll_bert_base_cased"
        self.brand.build_pipeline()
        
        stages = self.brand.pipeline.getStages()

        instances = [
            DocumentAssembler, 
            Tokenizer, 
            BertEmbeddings,
            NerDLModel, 
            NerConverter
        ]

        for stage, instance in zip(stages, instances):
            self.assertIsInstance(stage, instance)
        
        self.assertEqual(len(stages), len(instances))

    def test_predict_brand_xlnet_base(self):
        self.brand.model_name = "xlnet_base"
        brand_df = self.brand.predict_brand(self.df)

        self.assertIsInstance(brand_df, DataFrame)
        self.assertEqual(brand_df.count(), self.df.count())
        self.assertSetEqual(set(brand_df.columns), self.brand_columns)

    def test_predict_brand_conll_bert(self):
        self.brand.model_name = "ner_conll_bert_base_cased"
        brand_df = self.brand.predict_brand(self.df)

        self.assertIsInstance(brand_df, DataFrame)
        self.assertEqual(brand_df.count(), self.df.count())
        self.assertSetEqual(set(brand_df.columns), self.brand_columns)

    def test_predict_brand_with_filtering(self):
        brand_df = self.brand.predict_brand(self.df, True)

        self.assertIsInstance(brand_df, DataFrame)
        self.assertLess(brand_df.count(), self.df.count())
        self.assertSetEqual(set(brand_df.columns), self.brand_columns)
    """

if __name__ == "__main__":
    unittest.main()
