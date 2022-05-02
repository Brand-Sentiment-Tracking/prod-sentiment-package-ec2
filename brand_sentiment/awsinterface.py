import logging

from typing import Optional, Union

from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F


class AWSInterface:
    """Download, preprocess and upload article dataframes via AWS S3.

    Note:
        This class assumes the dataframes are from parquet files partitioned
        first by date crawled, followed by language.

    Args:
        spark (SparkSession): The spark session to run the app through.
        extraction_bucket (str): The name of the bucket to download articles
            from.
        sentiment_bucket (str): The name of the bucket to upload results to.
        partitions (int): The number of partitions to store the dataframe in
            during execution. The best value is dependent on how the spark
            session has been created and the hardware of your machine.
        extraction_date (datetime or str): The date to pull articles from for
            analysis. If None, the date will be taken to be yesterday's.
        language (str): The short-code of the language to pull articles from
            for analysis. Languages other than english are not currently
            supported.
        log_level (_Level): The severity level of logs to be reported.
    """
    def __init__(self, spark: SparkSession, extraction_bucket: str,
                 sentiment_bucket: str, partitions: int = 32,
                 extraction_date: Optional[Union[str, datetime]] = None,
                 language: str = "en", log_level: int = logging.INFO):

        self.logger = logging.getLogger("AWSInterface")
        self.logger.setLevel(log_level)

        self.spark = spark

        self.extraction_bucket = extraction_bucket
        self.sentiment_bucket = sentiment_bucket

        self.partitions = partitions

        if extraction_date is None:
            yesterday = datetime.now() - timedelta(days=1)
            self.extraction_date = yesterday
        else:
            self.extraction_date = extraction_date

        self.language = language

    @property
    def extraction_bucket(self) -> str:
        """`str`: The name of the bucket to download articles from."""
        return self.__extraction_bucket

    @extraction_bucket.setter
    def extraction_bucket(self, name: str):
        if type(name) != str:
            raise TypeError("Bucket name is not a string.")

        self.__extraction_bucket = name

    @property
    def sentiment_bucket(self) -> str:
        """`str`: The name of the bucket to upload results to."""
        return self.__sentiment_bucket

    @sentiment_bucket.setter
    def sentiment_bucket(self, name: str):
        if type(name) != str:
            raise TypeError("Bucket name is not a string.")

        self.__sentiment_bucket = name

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
            raise TypeError("Partitions is not an integer.")
        elif n <= 0:
            raise ValueError("Partitions is not greater than 0.")

        self.__partitions = n

    @property
    def extraction_date(self) -> str:
        """`str`: The date to pull articles from for analysis.

        The setter will automatically try to convert a string or datetime
        object to YYYY-MM-DD format. If the date is in the future, a
        ValueError will be raised.
        """
        return self.__extraction_date

    @extraction_date.setter
    def extraction_date(self, new_date: Union[str, datetime]):
        if type(new_date) == datetime:
            parsed_date = new_date
        elif type(new_date) == str:
            try:
                parsed_date = datetime.fromisoformat(new_date)
            except ValueError as parse_error:
                setter_error = ValueError("Extraction date isn't ISO format.")
                raise setter_error from parse_error
        else:
            raise TypeError("Extraction date is not a string or datetime.")

        if parsed_date > datetime.now():
            raise ValueError("Extraction date is in the future.")

        self.__extraction_date = parsed_date.strftime("%Y-%m-%d")

    @property
    def language(self) -> str:
        """`str`: The short-code of the language to pull articles from for
        analysis.

        Note:
            Languages other than english are not currently supported.
        """
        return self.__language

    @language.setter
    def language(self, lang: str):
        if type(lang) != str:
            raise TypeError("Lanaguage is not a string.")

        self.__language = lang

    @property
    def extraction_url(self) -> str:
        """`str`: The full URL to the parquet files to download from S3."""
        return f"s3a://{self.extraction_bucket}/" \
               f"date_crawled={self.extraction_date}/" \
               f"language={self.language}/"

    @property
    def sentiment_url(self) -> str:
        """`str`: The full URL to the folder to upload results to in S3."""
        return f"s3a://{self.sentiment_bucket}/"

    def __preprocess_dataframe(self, df: DataFrame) -> DataFrame:
        """Edit the article dataframe in preparation for sentiment analysis.

        This includes:
            - Filling in all null publish dates with the date crawled.
            - Adding the language column back into the dataframe.
            - Renaming the title column to text.
            - Repartitioning the dataframe.

        Args:
            df (DataFrame): The spark df to preprocess.

        Returns:
            DataFrame: The processed dataframe ready for sentiment analysis.
        """
        dates = F.when(df["date_publish"].isNull(), self.extraction_date) \
            .otherwise(df["date_publish"])

        return df.withColumnRenamed("title", "text") \
            .withColumn("date_publish", dates) \
            .withColumn("language", F.lit("en")) \
            .repartition(self.partitions)

    def download(self, limit: Optional[int] = None) -> DataFrame:
        """Download the parquet files from S3 and load them into a spark df.

        Args:
            limit (int): The maximum number of articles to download. If None,
                all articles available will be downloaded.

        Returns:
            DataFrame: The articles in a spark df, preprocessed.
        """
        self.logger.info(f"Downloading from '{self.extraction_url}'.")
        df = self.spark.read.parquet(self.extraction_url)

        if limit is not None:
            self.logger.info(f"Reducing dataframe to {limit} rows.")
            df = df.limit(limit)

        self.logger.debug("Setting language to 'en' and null publish"
                          f"dates to '{self.extraction_date}'.")

        return self.__preprocess_dataframe(df)

    def upload(self, df: DataFrame):
        """Upload a spark dataframe to AWS S3.

        Args:
            df (DataFrame): The spark dataframe to upload.
        """
        self.logger.info(f"Uploading results to '{self.sentiment_url}'.")
        df.write.mode('append').parquet(self.sentiment_url)

        self.logger.info("Upload successful.")
