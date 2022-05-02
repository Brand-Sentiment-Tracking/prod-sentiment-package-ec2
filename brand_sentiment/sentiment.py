import logging

from typing import List

from pyspark.sql import SparkSession, DataFrame, Row
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql import functions as F

from pyspark.sql.window import Window, WindowSpec

from sparknlp.pretrained import PretrainedPipeline


class SentimentIdentification:
    """Run Sentiment Analysis on articles using spark dataframes.

    Note:
        Currently, only `classifierdl_bertwiki_finance_sentiment_pipeline` is
        supported. Other models can be used, but the program may crash.

    Args:
        spark (SparkSession): The spark session to run the app through.
        model_name (str): The sentiment model to use.
        partitions (int): The number of partitions to store the dataframe in
            during execution. The best value is dependent on how the spark
            session has been created and the hardware of your machine.
        log_level (_Level): The severity level of logs to be reported.
    """
    SENTIMENT_FIELDS = ("text", "source_domain", "date_publish", "language",
                        "entities", "class.result")

    def __init__(self, spark: SparkSession, model_name: str,
                 partitions: int = 32, log_level: int = logging.INFO):

        self.logger = logging.getLogger("SentimentIdentification")
        self.logger.setLevel(log_level)

        self.spark = spark
        self.model_name = model_name

        self.partitions = partitions

    @property
    def model(self) -> PretrainedPipeline:
        """`PretainedPipeline`: The model to transform the dataframe by."""
        return self.__model

    @property
    def model_name(self) -> str:
        """`str`: The name of the pretrained model to build.

        Once this has been set, the model will be built.
        """
        return self.__model_name

    @model_name.setter
    def model_name(self, name: str):
        if type(name) != str:
            raise TypeError("Model name is not a string.")
        if name != "classifierdl_bertwiki_finance_sentiment_pipeline":
            self.logger.warning("Pipeline hasn't been designed for model "
                                f"'{name}'. Using this model may cause the "
                                "pipeline to crash.")

        self.__model_name = name
        self.__model = PretrainedPipeline(self.model_name, lang='en')

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
            raise TypeError("Partitions is not an integer")
        elif n <= 0:
            raise ValueError("Partitions is not greater than 0.")

        self.__partitions = n

    def __get_sentiment_scores(self, df: DataFrame,
                               window: WindowSpec) -> DataFrame:
        """Add the probabilities and the sentiment score to the df.

        This must be done once the pipeline has been run (and there is a
        `metadata` column).

        Args:
            df (DataFrame): The dataframe to add scores to.
            window (WindowSpec): The window function to apply over the
                dataframe (for recording column index when rejoining
                dataframes.)

        Returns:
            DataFrame: The same dataframe with the added scores.
        """
        return df \
            .select(F.explode(F.col("class.metadata")).alias("metadata")) \
            .select(F.col("metadata")["positive"].alias("positive"),
                    F.col("metadata")["neutral"].alias("neutral"),
                    F.col("metadata")["negative"].alias("negative")) \
            .withColumn("score", F.col("positive") - F.col("negative")) \
            .withColumn("column_index", F.row_number().over(window))

    def __reorganise_df(self, df: DataFrame, window: WindowSpec) -> DataFrame:
        """Modify dataframe in preparation for joining it with scores.

        This includes:
        - Renaming the `result` column to `sentiment`.
        - Wrapping the values within sentiment in lists (for appending NER to).
        - Adding a column index for rejoining dataframes together.

        Args:
            df (DataFrame): The dataframe to modify.
            window (WindowSpec): The window function to apply over the
                dataframe (for recording column index when rejoining
                dataframes.)

        Returns:
            DataFrame: The modified dataframe.
        """
        return df \
            .select(*self.SENTIMENT_FIELDS) \
            .withColumnRenamed("result", "sentiment") \
            .withColumn("sentiment", F.array_join("sentiment", "")) \
            .withColumn("column_index", F.row_number().over(window))

    @staticmethod
    @F.udf(returnType=ArrayType(ArrayType(StringType())))
    def __append_sentiment(entities: List[Row], sent: str) -> List[List[str]]:
        """User defined function to add sentiment to each NER result."""
        for entity in entities:
            entity.append(sent)

        return entities

    def __add_scores(self, df: DataFrame, scores: DataFrame) -> DataFrame:
        """Combine the dataframe and scores.

        This includes:
        - Joining rows of both dataframes together (based on column index).
        - Adding the sentiment score to each entity in the NER results (now
            called `Predicted_Entity_and_Sentiment`).
        - Dropping the column index columns.
        - Dropping the entities and sentiment columns

        Args:
            df (DataFrame): The modified dataframe.
            scores (DataFrame): The dataframe with expanded score columns.

        Returns:
            DataFrame: The combined dataframe.
        """
        mask = df.column_index == scores.column_index

        return df.join(scores, mask, "inner") \
            .drop(scores.column_index) \
            .drop(df.column_index) \
            .withColumn('Predicted_Entity_and_Sentiment',
                        self.__append_sentiment('entities', 'sentiment')) \
            .drop('entities', 'sentiment')

    def predict_sentiment(self, brand_df: DataFrame) -> DataFrame:
        """Annotates each NER result in the df with the sentiment.

        Note:
            The `entities` column is renamed to
            `Predicted_Entity_and_Sentiment`.

        Args:
            df : The dataframe to add sentiment analysis to. Must contain a
                `text` column and the NER results in an `entities` column.

        Returns:
            DataFrame: The dataframe with the added sentiment analysis.
        """
        self.logger.info("Running sentiment model...")
        df = self.model.transform(brand_df)

        w = Window.orderBy(F.monotonically_increasing_id())

        self.logger.info("Calculating sentiment scores.")
        scores = self.__get_sentiment_scores(df, w)

        self.logger.info("Reorganising dataframe.")
        df = self.__reorganise_df(df, w)

        self.logger.info("Adding scores to dataframe.")

        return self.__add_scores(df, scores) \
            .repartition(self.partitions)
