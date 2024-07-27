<h1 align="center">News Search Engine</h1>
<h6 align="center">Spring-2023 Information Retrieval Course Final Project at Amirkabir University of Tech.</h6>


## Introduction
This project is a search engine for Persian news, intended to offer the most relevant results for each query.
The retrieval can be done in multiple modes:
1. **Simple Positional Search:** In this mode AND and NOT (!) operators can be used in the query and the documents containing the query tokens are retrieved. When an expression is placed in a double quotation the tokens within should appear as a squence, otherwise the order is irrelevant. The documents are ranked solely based on term frequencies.
2. **Ranked Search Based on Jaccard Similarity:** In this mode, each document is first represented as a vector using the td-idf measure. A similar vector is also computed for the input query. Now, we can compare these two vectors using various similarity measures. This mode uses Jaccard similarity for this purpose.
3. **Ranked Search Based on Cosine Similarity:** This mode utilizes a similar approach to the previous mode, but employs the cosine similarity algorithm for vector comparison. This addresses certain limitations of Jaccard similarity, such as its sensitivity to document length.
4. **Champion List Search**: This mode can be used with either Jaccard or Cosine similarity.  It utilizes champion lists to accelerate the search process. While this mode prioritizes speed, it may lead to slightly reduced relevance. 
<p align="center">
  <br>
  <img src="https://github.com/NegarMov/News-Search-Engine/blob/master/output.jpg" alt="Sample Output" width="800"/>
</p>

## Preprocessing
Various methods of preprocessing are applied both to the documents and the queries, including:
* Normalization
* Lemmatization
* Stop words and punctuations elimination
