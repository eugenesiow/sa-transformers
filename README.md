# Sentiment Analysis with Transformers
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)

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