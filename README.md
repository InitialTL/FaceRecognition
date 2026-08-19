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

Start training the model by running <code>hitboxes/main.py</code>
```bash
python -m hitboxes.main
```

Or with the <code>-e</code> / <code>--epochs</code> flag
```bash
python -m hitboxes.main -e 5
```

<p>The -l (--load-existing) is used when wanting to further improve a model that's trained.
Takes the path to the model as parameter.</p>
<strong>NOTICE:</strong> the parameter does not automatically add the hitboxes/ preset

```bash
python -m hitboxes.main -e 5 -l hitboxes/models/model.pth
```

