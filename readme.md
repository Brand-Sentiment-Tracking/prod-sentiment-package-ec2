# Article NER & Sentiment Analysis Application

## Pre-requisites & Installation
Make sure you have Java 8 installed and Apache Spark installed

To install the requirements use this command:

    python3.8 -m pip install -r requirements.txt

## Usage

### `AWSInterface` Class

### `BrandIdentification` Class

### `SentimentIdentification` Class

### Configuration

```docker
ENV EXTRACTION_BUCKET_NAME=extracted-news-articles
ENV SENTIMENT_BUCKET_NAME=processed-news-articles
ENV EXTRACTION_DATE=2022-04-26

ENV NER_MODEL=xlnet_base
ENV SENTIMENT_MODEL=classifierdl_bertwiki_finance_sentiment_pipeline

ENV DATAFRAME_PARTITIONS=32
ENV DATAFRAME_LIMIT=100
```

## Repository Structure

## Contact
