# Sentiment Analysis with Transformers
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

## Classification Results

### ~4000 comments

|Model Type     |Model Name                 |Accuracy   |
|---            |---                        |---        |
|XLNet          |xlnet-base-cased           |0.784      |
|RoBERTa        |roberta-base               |**0.802**  |
|RoBERTa        |distilroberta-base         |0.763      |
|BERT           |bert-base-cased            |0.780      |
|BERT           |bert-base-uncased          |0.680      |
|BERT           |bert-large-cased           |0.772      |
|FLAIR          |LSTM                       |0.701      |
|FLAIR          |GRU                        |0.706      |
|ALBERT         |albert-base-v1             |0.668      |
|DistilBERT     |distilbert-base-cased      |0.742      |

### ~6000 comments

|Model Type     |Model Name                 |Accuracy   |
|---            |---                        |---        |
|XLNet          |xlnet-base-cased           |0.822      |
|RoBERTa        |roberta-base               |**0.844**  |

Hence, a pre-trained RoBERTa (base) transformer language model was used to train this classifier. Final model accuracy was **84.4%**.

## Docker
1. Build the image: `docker build -t dso/sa-transformers:v1.0 .`.
2. Run the container: `docker run -d -p 5128:80 --name sa-transformers-container dso/sa-transformers:v1.0`.
3. Export the image for distribution: `docker image save -o sa-transformers-v1.0.tar dso/sa-transformers:v1.0`.
    * For archiving, a gzip image is recommended: `docker save dso/sa-transformers:v1.0 | gzip -c > sa-transformers-v1.0.tar.gz`.
4. To load an image from a gzip archive: `docker load < sa-transformers-v1.0.tar.gz`.

## RESTful API

* **URL**
    /sentiment

* **Method**
    `POST`
  
*  **URL Params**
    None

* **Data Params**
    * **Format:** `application/json`
    #### Example Input
    A JSON array list of input text to analyse:
    ```
    ["Text to analyse for sentiment #1", 
    "Text to analyse for sentiment #1"]
    ```

* **Success Response**
    * **Code:** `200`
        
    #### Example Results
    A JSON array list of sentiment outputs, order is that of the input array order.
    Negative: -1, Positive: 1, Neutral: 0.
    ```
    [0,
    -1]
    ```
  
* **Invalid JSON Input**
    * **Code:** `400`
        
    #### Example Results
    ```
    {
        "ERROR": "Please provide a valid JSON. "
    }
    ```