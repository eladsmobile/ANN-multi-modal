## Balancing Multiple Similarity Approaches through Weighted ANN Search
This project explores advanced methods for merging multi-modal embeddings to improve accuracy and efficiency in Approximate Nearest Neighbor (ANN) search. The study evaluates fusion strategies, including normalized and scaled embeddings, robust multi-indexing, and learned weights, focusing on balancing accuracy (measured by Mean Average Precision, MAP) and computational efficiency.

Key highlights include:

* Datasets: Experiments were conducted on CIFAR-100 and Tiny ImageNet datasets, ensuring diversity in testing multi-modal embedding techniques.
* Fusion Methods: Techniques such as dimension reduction (PCA), robust multi-indexing, and tolerant search were tested alongside baseline methods like concatenation and separate indexing.
* Evaluation Metrics: Performance was measured using MAP, accuracy, recall, and computational metrics like indexing and search runtime.
* Results: Normalized embeddings and robust multi-indexing achieved the best MAP scores, significantly outperforming baseline approaches, though with increased computational costs.
* Trade-offs: While sophisticated methods improve retrieval accuracy, they require higher processing power and time, emphasizing the need for balanced trade-offs in real-world applications.

The findings demonstrate the potential of hybrid embedding techniques and advanced fusion strategies in multi-modal search tasks, paving the way for more robust and scalable solutions in ANN search.

-----------------------------
## Setup
Clone the project:
```sh
git clone https://github.com/eladsmobile/ANN-multi-modal.git
```
From the project directory, run:
```sh
pip install -r requirements.txt
```

## Run the code
Use jupyter notebook to run combine_imbeddings_with_ANN.ipynb and reproduce our results.
