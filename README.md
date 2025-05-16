<a id="readme-top"></a>
<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/shafiqninaba/resonare">
    <img src="assets/images/logo.png" alt="Logo" height="400" width="400">
  </a>
<h2 align="center">Resonare</h2>

  <p align="center">
    An application that uses an Agentic RAG pipeline to retrieve information from the vector store using Qdrant that contains crawled websites via Firecrawl.
    <br />
    <a href="https://github.com/shafiqninaba/resonare/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/shafiqninaba/resonare/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
    <a href="#built-with">Built With</a>
    </li>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#description">Description</a></li>
        <li><a href="#data-pipeline">Data Pipeline</a></li>
        <li><a href="#retrieval-and-question-answering-pipeline">Retrieval and Question Answering Pipeline</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation-and-commands">Installation and Commands</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#demo">Demo</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## Built With

* [![Python][Python-img]][Python-url]
* [![uv][uv-img]][uv-url]
* [![unsloth][unsloth-img]][unsloth-url]
* [![Streamlit][streamlit-img]][streamlit-url]
* [![AWS][aws-img]][aws-url]

<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]]


### Description

#### Software Architecture

The diagram below shows the high-level architecture of the application.

![Architecture](assets/images/software-architecture.png)

##### Key Components

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DEMO -->
## Demo

The gifs below showcase the key features of the application, ...

### DEMO GIF 1

<div align="center">
  <img src="assets/gifs/firecrawler.gif" alt="Firecrawler Demo"> <p><em>Figure 1: Document Crawling and Indexing Process</em></p>
</div>

<PLACEHOLDER>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

- [Python](https://www.python.org/) 3.8 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- [Docker](https://www.docker.com/) to run the application locally
- [AWS Account](https://aws.amazon.com/) to store the crawled data in an S3 bucket

### Installation and Commands

1. Clone the repository:
```bash
git clone https://github.com/shafiqninaba/resonare.git
cd resonare
```

2. Create an .env file in the root directory and add the following environment variables:
```
AWS_ACCESS_KEY_ID=XXXXXXXX
AWS_SECRET_ACCESS_KEY=YYYYYYYYYYYYYYYY
AWS_S3_BUCKET=resonare-test-bucket
AWS_REGION=ap-southeast-1
```

3. Build and run the docker compose file:
```bash
docker-compose up --build
```

4. Open your browser and go to `http://localhost:8501` to access the web app.


5. Close the web app and stop the docker container:
```bash
docker-compose down
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

<a href="https://github.com/shafiqninaba/resonare/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=shafiqninaba/ask-the-docs" alt="contrib.rocks image" />
</a>

Shafiq Ninaba | shafiqninaba@gmail.com | [LinkedIn](https://linkedin.com/in/shafiq-ninaba)

Zack Low | lowrenhwa88@gmail.com | [LinkedIn](https://www.linkedin.com/in/ren-hwa-low-855080224/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/shafiqninaba/ask-the-docs.svg?style=for-the-badge
[contributors-url]: https://github.com/shafiqninaba/resonare/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shafiqninaba/ask-the-docs.svg?style=for-the-badge
[forks-url]: https://github.com/shafiqninaba/resonare/network/members
[stars-shield]: https://img.shields.io/github/stars/shafiqninaba/ask-the-docs.svg?style=for-the-badge
[stars-url]: https://github.com/shafiqninaba/resonare/stargazers
[issues-shield]: https://img.shields.io/github/issues/shafiqninaba/ask-the-docs.svg?style=for-the-badge
[issues-url]: https://github.com/shafiqninaba/resonare/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[product-screenshot]: assets/images/homepage.jpg
[Python-img]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[uv-img]: https://img.shields.io/badge/uv-package%20manager-blueviolet
[uv-url]: https://docs.astral.sh/uv/
[unsloth-img]: https://raw.githubusercontent.com/unslothai/unsloth/main/images/made%20with%20unsloth.png
[unsloth-url]: https://github.com/unslothai/unsloth
[streamlit-img]: https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white
[streamlit-url]: https://streamlit.io/
[aws-img]: https://www.google.com/url?sa=i&url=https%3A%2F%2Ficonduck.com%2Ficons%2F94029%2Faws&psig=AOvVaw1GvNS6KfUIPaOdVNrfzhJA&ust=1747453015928000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCKjY-_SHp40DFQAAAAAdAAAAABAK
[aws-url]: https://aws.amazon.com/
