# Fake News Detector API REST

[![Docker Pulls](https://img.shields.io/docker/pulls/josumsc/flask-fake-news.svg)](https://hub.docker.com/repository/docker/josumsc/flask-fake-news/general)
[![Docker Stars](https://img.shields.io/docker/stars/josumsc/flask-fake-news.svg)](https://hub.docker.com/repository/docker/josumsc/flask-fake-news/general)

This is a simple API REST to detect fake news using a machine learning model. It is based on the [DistilBERT uncased model](distilbert-base-uncased-finetuned-sst-2-english) and fine-tuned using the [Fake News Dataset](https://huggingface.co/datasets/GonzaloA/fake_news) curated by the user GonzaloW.

The model is located both in [Hugging Face's model hub](https://huggingface.co/josumsc/fake-news-detector) as a standalone PyTorch model and in [Docker Hub](https://hub.docker.com/repository/docker/josumsc/flask-fake-news/general) as a Docker image. I'd rather recommend using the Docker image, as it is easier to deploy and includes the necessary dependencies and code to be executed.

## Usage

### Docker Compose

There is a `Makefile` that calls the `docker-compose` command to run the API REST. You can use it to run the API REST in a container.

```bash
# Run the API REST
make run
```

This instruction will run the command on daemon mode, so in case you want to stop it you can run:

```bash
# Stop the API REST
make stop
```

The `make run` command will start the API REST in the port 5001. You can test it using the following command:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "This is a fake news"}' http://localhost:5001/detect_json
```

![API REST](https://github.com/josumsc/fake-news-detector/blob/master/docs/img/api-rest.png?raw=true)

At the same time, you can also use a simple HTML based interface to test the API REST. You can access it in the following URL: [http://localhost:5001](http://localhost:5001) and fill the form there to test the model.

![HTML interface](https://github.com/josumsc/fake-news-detector/blob/master/docs/img/html-interface.png?raw=true)

### Docker Image

If you want to use the Docker image on your own applications without the `docker-compose.yml` proposed, you can pull it from Docker Hub and run it.

```bash
# Pull the image
docker pull josumsc/flask-fake-news

# Run the container
docker run -p 5001:5000 josumsc/flask-fake-news
```

### Retrain the model and publish it to Hugging Face's model hub

If you want to retrain the model, you can use the `cli.py` utility with the `train` command. It will download the dataset and train the model. You can also use the `--output` argument to specify the name of the model that will be saved in the Hugging Face's model hub. This process will also print the results of the evaluation of the model.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate # Linux
venv\Scripts\activate # Windows

# Install the dependencies
pip install -r requirements.txt

# Train the model
python cli.py train --output <model_name>
```

> **Note**: In order to run PyTorch with CUDA, you need to install the CUDA Toolkit and cuDNN. You can find more information about this in the [PyTorch documentation](https://pytorch.org/get-started/locally/). It is possible that you should also remove and reinstall PyTorch after installing the CUDA Toolkit and cuDNN.

Later on, you can use the same utility to publish the model to Hugging Face's model hub. You should have logged in to Hugging Face's model hub before using this command, as it will use your credentials to upload the model.

```bash
# Login to Hugging Face's model hub
huggingface-cli login

# Publish the model
python cli.py publish --model <model_name>
```
