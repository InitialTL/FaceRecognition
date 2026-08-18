<h1>Face Recognition</h1>
<h2>Hitboxes</h2>
<h3>Datasets are available here</h3>
<ul>
  <li>https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset?resource=download-directory&select=labels</li>
  <li>https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset?resource=download-directory&select=images</li>
</ul>
<p>Before training, download the zip files linked above, and unzip the zip-files in "hitboxes/data/"
</p>

Install the necessary packages with
```bash
pip install -r requirements.txt
```

<p>Start training the model by running <code>hitboxes/main.py</code></p>
```bash
python -m hitboxes.main
```
<p>Or with the epochs flag <code>-e</code> or <code>--epochs</code></p>

```bash
python -m hitboxes.main -e 20
```
