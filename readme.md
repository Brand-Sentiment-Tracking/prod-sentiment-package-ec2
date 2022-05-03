# Article NER & Sentiment Analysis Application

## Pre-requisites & Installation
Make sure you have [Java 8](https://java.com/en/download/manual.jsp) installed and [Apache Spark](https://spark.apache.org/downloads.html) installed

To install the requirements use this command:

    python3.8 -m pip install -r requirements.txt

## Usage

### `AWSInterface` Class

Download, preprocess and upload article dataframes via AWS S3.

_Note: This class assumes the dataframes are from parquet files partitioned first by date crawled, followed by language._

### `BrandIdentification` Class

Run NER Analysis on articles using spark dataframes.

Currently, only two NER models are supported:
    - `xlnet_base` (default)
    - `ner_conll_bert_base_cased`

### `SentimentIdentification` Class

Run Sentiment Analysis on articles using spark dataframes.+

_Note: Currently, only `classifierdl_bertwiki_finance_sentiment_pipeline` is supported. Other models can be used, but the program may crash._
### Configuration

```docker
# Bucket from which the data is obtained
ENV EXTRACTION_BUCKET_NAME=extracted-news-articles
# Bucket to which the processed data is outputted
ENV SENTIMENT_BUCKET_NAME=processed-news-article
# Date of extraction that we want to process
ENV EXTRACTION_DATE=2022-04-26 

# Entity Identification model: xlnet_base | ner_conll_bert_base_cased
ENV NER_MODEL=xlnet_base
# Entity Sentiment model
ENV SENTIMENT_MODEL=classifierdl_bertwiki_finance_sentiment_pipeline 

ENV DATAFRAME_PARTITIONS=32
ENV DATAFRAME_LIMIT=100
```

## Repository Structure

## Contact
