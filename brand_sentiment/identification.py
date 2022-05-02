import logging

from typing import List

from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql import functions as F

from pyspark.ml import Pipeline, PipelineModel

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import XlnetForTokenClassification, \
    Tokenizer, BertEmbeddings, NerDLModel, NerConverter


class BrandIdentification:
    """Run NER Analysis on articles using spark dataframes.

    Currently, only two NER models are supported:
    - `xlnet_base` (default)
    - `ner_conll_bert_base_cased`

    Args:
        spark (SparkSession): The spark session to run the app through.
        model_name (str): The NER model to use. Must be one of the above.
        partitions (int): The number of partitions to store the dataframe in
            during execution. The best value is dependent on how the spark
            session has been created and the hardware of your machine.
        log_level (_Level): The severity level of logs to be reported.
    """
    NER_FIELDS = ("text", "source_domain", "date_publish",
                  "language", "entities")

    def __init__(self, spark: SparkSession, model_name: str,
                 partitions: int = 32, log_level: int = logging.INFO):

        self.logger = logging.getLogger("BrandIdentification")
        self.logger.setLevel(log_level)

        self.spark = spark

        self.model_name = model_name
        self.partitions = partitions

    @property
    def model(self) -> PipelineModel:
        """`PipelineModel`: The model to transform the dataframe by."""
        return self.__model

    @property
    def pipeline(self) -> Pipeline:
        """`Pipeline`: The pipeline to build the model from."""
        return self.__pipeline

    @property
    def model_name(self) -> str:
        """`str`: The name of the pretrained model to build.

        Once this has been set, the pipeline and model will be built.
        """
        return self.__model_name

    @model_name.setter
    def model_name(self, name: str):
        if name not in ("xlnet_base", "ner_conll_bert_base_cased"):
            raise ValueError("Model must be either 'xlnet_base' or "
                             "'ner_conll_bert_base_cased'.")

        self.__model_name = name
        self.__build_pipeline()

    @property
    def partitions(self) -> int:
        """`int`: The number of partitions to store a dataframe in during
        analysis.

        The best value is dependent on how the spark session has been created
        and the hardware of your machine. By default, this is 32. Setting
        this value to less than 1 will throw a ValueError.
        """
        return self.__partitions

    @partitions.setter
    def partitions(self, n: int):
        if type(n) != int:
            raise ValueError("Partitions is not an integer")
        elif n <= 0:
            raise ValueError("Partitions is not greater than 0.")

        self.__partitions = n

    def __build_document_stages(self) -> List:
        """Build the preprocessing stages of the NER model.

        This includes:
        - A `DocumentAssembler` stage.
        - A `Tokenizer` stage

        Returns:
            List: The stage instances wrapped in a list.
        """
        document_assembler = DocumentAssembler() \
            .setInputCol('text') \
            .setOutputCol('document')

        tokenizer = Tokenizer() \
            .setInputCols(['document']) \
            .setOutputCol('token')

        return [document_assembler, tokenizer]

    def __build_converter_stages(self) -> List:
        """Build the postprocessing stage for NER evaluation.

        Returns:
            List: The `NerConverter` stage wrapped in a list.
                (incase this is extended to beyond a single stage)
        """
        ner_converter = NerConverter() \
            .setInputCols(['document', 'token', 'ner']) \
            .setOutputCol('ner_chunk')

        return [ner_converter]

    def __build_xlnet_model_stages(self) -> List:
        """Build the XLNet Classification stage.

        Note:
            Parameters for this stage have not been made configurable.

        Returns:
            List: The `XlnetForTokenClassification` stage wrapped in a
                list (incase this is extended to beyond a single stage).
        """
        token_classifier = XlnetForTokenClassification \
            .pretrained('xlnet_base_token_classifier_conll03', 'en') \
            .setInputCols(['token', 'document']) \
            .setOutputCol('ner') \
            .setCaseSensitive(True) \
            .setMaxSentenceLength(512)

        return [token_classifier]

    def __build_conll_model_stages(self) -> List:
        """Build the CoNLL BERT Classification stage.

        This includes:
        - A `BertEmbeddings` stage.
        - A `NerDLModel` stage

        Returns:
            List: The stage instances wrapped in a list.
        """
        embeddings = BertEmbeddings \
            .pretrained(name='bert_base_cased', lang='en') \
            .setInputCols(['document', 'token']) \
            .setOutputCol('embeddings')

        ner_model = NerDLModel \
            .pretrained(self.model_name, 'en') \
            .setInputCols(['document', 'token', 'embeddings']) \
            .setOutputCol('ner')

        return [embeddings, ner_model]

    def __build_pipeline(self):
        """Build the complete pipeline and model to be used for NER."""
        self.logger.info("Building NER Pipeline...")

        self.logger.info("Building Document Assembler & Tokeniser.")
        stages = self.__build_document_stages()

        if self.model_name == "xlnet_base":
            self.logger.info("Building XLNet Model.")
            stages.extend(self.__build_xlnet_model_stages())

        elif self.model_name == "ner_conll_bert_base_cased":
            self.logger.info("Building CoNLL BERT Model.")
            stages.extend(self.__build_conll_model_stages())
        else:
            # Should never get here.
            logging.fatal("No matching model name for pipeline. Should have "
                          "thrown a ValueError when setting the model name.")

        self.logger.info("Build NER Converter.")
        stages.extend(self.__build_converter_stages())

        self.__pipeline = Pipeline(stages=stages)
        self.logger.info("Pipeline built successfully.")

        self.logger.info("Building model from pipeline...")
        empty_df = self.spark.createDataFrame([['']], ["text"])
        self.__model = self.pipeline.fit(empty_df)

        self.logger.info("Model built successfully.")

    @staticmethod
    @F.udf(returnType=ArrayType(ArrayType(StringType())))
    def __extract_brands(rows: List[Row]) -> List[List[str, str]]:
        """User defined function to return a list of names and entity types."""
        return [[row.result, row.metadata['entity']] for row in rows]

    def predict_brand(self, df: DataFrame,
                      filter_non_entities: bool = False) -> DataFrame:
        """Run NER analysis on the text column of the dataframe.

        Args:
            df (DataFrame): The dataframe of articles to run NER on.
            filter_non_entities (bool): Remove rows that have no identified
                entities.

        Returns:
            DataFrame: The same dataframe, but with an extra column `entities`
                conatining the result of the NER analysis.
        """
        self.logger.info("Running NER model.")

        brand_df = self.model.transform(df) \
            .withColumn("entities", self.__extract_brands('ner_chunk')) \
            .select(*self.NER_FIELDS)

        return self.remove_articles_without_entities(brand_df) \
            if filter_non_entities else brand_df.repartition(self.partitions)

    def remove_articles_without_entities(self, df: DataFrame) -> DataFrame:
        """Filter dataframe to the rows that have detected entities.

        Args:
            df (DataFrame): The dataframe to filter.

        Returns:
            DataFrame: The dataframe without the rows that had no identified
                entities.
        """
        self.logger.info("Removing articles with no entities.")

        return df.filter(F.size(df.entities) > 0) \
            .repartition(self.partitions)
