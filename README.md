<a id="readme-top"></a>
<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/shafiqninaba/resonare">
    <img src="assets/images/resonare_banner.jpg" alt="Banner" width="70%" height="auto">
  </a>
<h2 align="center">Resonare</h2>

  <p align="center">
    End-to-end LLM Twin pipeline to make a clone of yourself based on your Telegram chat data.
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
        <li><a href="#overview">Overview</a>
          <ul>
            <li><a href="#software-architecture">Software Architecture</a></li>
            <li><a href="#key-components">Key Components</a></li>
          </ul>
        </li>
        <li><a href="#data-prep">Data-Prep</a>
          <ul>
            <li><a href="#software-architecture-1">Software Architecture</a></li>
            <li><a href="#key-components-1">Key Components</a></li>
          </ul>
        </li>
        <li><a href="#fine-tuning">Fine-Tuning</a>
          <ul>
            <li><a href="#software-architecture-2">Software Architecture</a></li>
            <li><a href="#key-components-2">Key Components</a></li>
          </ul>
        </li>
      </ul>
    </li>
    <li><a href="#demo">Demo</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation-and-commands">Installation and Commands</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## Built With

* [![Python][Python-img]][Python-url]
* [![uv][uv-img]][uv-url]
* <a href="https://github.com/unslothai/unsloth"><img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/made with unsloth.png" height="30" align="center" style="margin-bottom: 5px" /></a>
* [![Streamlit][streamlit-img]][streamlit-url]
* <a href="https://aws.amazon.com/"><img src="https://upload.wikimedia.org/wikipedia/commons/9/93/Amazon_Web_Services_Logo.svg" height="40" align="center" style="margin-bottom: 5px" /></a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

***

<!-- ABOUT THE PROJECT -->
## About The Project

<div align="center">
  <img src="assets/images/main_screenshot.jpg" alt="Product Screenshot" width="70%" height="auto">
</div>


### Description

_Resonare_ (meaning "echo" in Latin) empowers users to create a personalized LLM-powered chatbot ("LLM Twin") trained on their Telegram chat data. The pipeline automates data cleaning, formatting, fine-tuning, and deployment of a conversational AI that mimics your messaging style and personality.

The end-to-end process includes:

- **Data Preparation:** Upload, clean, and chunk your exported Telegram conversations, producing high-quality training data.
- **Model Fine-Tuning:** Fine-tune a state-of-the-art language model to generate responses that reflect your unique conversational traits.
- **Inference:** Deploy and interact with your LLM Twin via a simple web interface or API.

### Overview

#### Software Architecture
The diagram below shows the high-level architecture of the application.

<div align="center">
  <img src="assets/images/overall_architecture.jpg" alt="Overall software architecture" width="70%" height="auto">
  <p><em>Figure 1: Overall software architecture of Resonare</em></p>
</div>

#### Key Components

- **FastAPI**: The API server that handles the requests from the client and manages the job queue.
- **Job Queue**: An in-memory queue that manages the jobs and ensures that the GPU is not overloaded with multiple jobs at the same time.
- **Async Worker**: A background worker that processes the jobs in the queue. It loads the JSON file, processes it and uploads the processed files to S3.
- **S3**: An AWS S3 bucket that stores the processed files. The raw JSON file is also uploaded to S3 for backup purposes.
- **Docker Compose**: The application is containerized using Docker Compose. This allows for easy deployment and scaling of the application.

#### Visual Flow

Here’s how a typical user would use Resonare:

1. Export: User exports their Telegram chat history.
2. Upload: They upload result.json through the web UI.
3. Processing: The backend prepares processed.json and train.jsonl and uploads them to S3.
4. Fine-Tune: They submit a fine-tuning job; the system trains their LLM Twin.
5. Chat: They chat with their personalized AI, which responds in their style.

```mermaid
graph TD
    A[User Exports Telegram Data] --> B[Upload via Web/API]
    B --> C[Data Preparation / Processing]
    C --> D[train.jsonl + processed.json]
    D --> E[Fine-Tune Model]
    E --> F[Deploy for Inference]
    F --> G[User Chats with LLM Twin]
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

***

### Data-Prep

#### Software Architecture
The diagram below shows the high-level architecture of the application.

```mermaid
flowchart TD
    subgraph Client
        A[User POSTs JSON to <br/> /process]
        B["User GETs job status<br/> /jobs/{run_id}"]
        C[User GETs all jobs<br/> /jobs]
        D[User GETs queue info<br/> /queue]
        E[User GETs health<br/> /health]
    end

    subgraph FastAPI API Server
        A --> F[Validate & Save temp file]
        F --> G[Generate run_id<br/>Store job status]
        G --> H[Enqueue run_id in job_queue]
        H --> I[Return run_id to client]
        B --> J[Return job status]
        C --> K[Return all jobs status]
        D --> L[Return queue info]
        E --> M[Return health/S3 status]
    end

    subgraph Async Worker
        N[Dequeue run_id from job_queue]
        N --> O[Load JSON from temp file]
        O --> P[Processing Pipeline]
        P --> Q[Update job status: COMPLETED/FAILED]
        Q --> R[Write processed files: local/S3]
    end

    %% Processing Pipeline Steps
    subgraph Processing Pipeline
        P1[Load & parse raw JSON]
        P2[Upload raw JSON to S3]
        P3[Load tokenizer]
        P4[Build Chat & Message objects]
        P5[Chunk chats into blocks]
        P6[Merge, sanitize, add system prompt]
        P7[Compute & log statistics]
        P8[Export processed.json & train.jsonl]
        P --> P1 --> P2 --> P3 --> P4 --> P5 --> P6 --> P7 --> P8
    end

    %% Connections
    I ==> N
```

<div align="center"> <p><em>Figure 2: Data-Prep architectural flow diagram</em></p> </div>

#### Key Components

- **Processing Pipeline**: The pipeline that processes the raw JSON file and generates the processed files. It includes steps such as loading the JSON file, chunking the chats into blocks, merging and sanitizing the data, computing statistics and exporting the processed files.
- **Statistics**: The statistics about the raw JSON file are computed and logged. This includes the number of blocks, tokens, messages, chats and users in the file.
- **Health Check**: The health check endpoint checks the status of the S3 bucket and the job queue. It returns the status of the S3 bucket and the number of jobs in the queue.

More information about the data pipeline can be found in this [`README.md`](packages/data-prep/README.md).


<p align="right">(<a href="#readme-top">back to top</a>)</p>

***

### Fine-Tuning

#### Software Architecture
The diagram below shows the high-level architecture of the application.

```mermaid
flowchart TD
    subgraph Client
        A["User POSTs /finetune/submit<br/>(submit fine-tune job)"]
        B["User GETs /finetune/jobs/{run_id}<br/>(job status)"]
        C["User GETs /finetune/queue<br/>(queue info)"]
        D["User POSTs /infer<br/>(inference request)"]
        E["User GETs /system/health<br/>(health check)"]
    end

    subgraph FastAPI App
        F[API Router Layer]
        F1[finetune.router]
        F2[inference.router]
        F3[system.router]
        F --> F1
        F --> F2
        F --> F3

        F1 --> G1[JobService<br/>submit, status, queue]
        F2 --> G2[InferenceService<br/>infer]
        F3 --> G3[Health Check Logic]
    end

    subgraph Background Worker
        H["JobService.worker_loop<br/>(async job processing)"]
        H1[(S3 Upload/Download)]
        H2[Model Training]
        H3[Job Status Update]
        H --> H1
        H --> H2
        H --> H3
    end

    %% Client to API
    A --> F1
    B --> F1
    C --> F1
    D --> F2
    E --> F3

    %% API to Worker
    F1 -- Enqueue job --> H
    H -- Update status --> F1

    %% Inference
    F2 -- Load model from S3 --> H1
    F2 -- Generate reply --> G2

    %% Health
    F3 -- Check S3 & GPU --> G3

    %% S3 interactions
    G1 -- Uses --> H1
    G2 -- Uses --> H1
    G3 -- Uses --> H1
```

<div align="center"> <p><em>Figure 3: Fine-Tuning architectural flow diagram</em></p> </div>

#### Key Components

- **Unsloth**: Unsloth is used for fine-tuning the LLM. It allows models to be fine tuned on lesser GPU compute resources.
- **Fine-Tuning Pipeline**: The pipeline that fine-tunes the LLM on the processed data. It includes steps such as loading the model, training the model and uploading the model to S3.
- **Inference Pipeline**: The pipeline that generates the response from the LLM Twin. It includes steps such as loading the model, generating the response and returning the response to the client.
- **Health Check**: The health check endpoint checks the status of the S3 bucket and the job queue. It returns the status of the S3 bucket and the number of jobs in the queue.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

***

<!-- DEMO -->
## Demo

The gifs below showcase the key features of the application and showing it in action.

### Uploading and Processing Data

<div align="center">
  <img src="assets/gifs/data-prep-to-job-monitor.gif" alt="Upload Data Demo">
  <p><em>Figure 4: Uploading and Processing Data</em></p>
</div>

The gif above shows the user uploading a JSON file containing their Telegram chat data. The file is then processed and the user can monitor the job status in real-time in the Job Monitor tab. The processed data is stored in an S3 bucket.

<div align="center">
  <img src="assets/images/data-prep-stats.jpg" alt="Data Prep Statistics">
  <p><em>Figure 5: Some statistics about the raw Telegram data</em></p>
</div>

The user can also see some statistics about the raw Telegram data, such as the number of blocks, tokens etc. This is useful for the user to understand the data they are working with.

### Checking Job Queue

<div align="center">
  <img src="assets/images/job_queue.jpg" alt="Job Queue Screenshot">
  <p><em>Figure 6: Job Queue</em></p>
</div>

When another user uploads a file, the job is added to the queue. This is so that the GPU will not be overloaded with multiple jobs at the same time. The user can check the queue status in the Job Monitor tab. The job queue is managed by a in memory queue.

### Chatting with the LLM Twin

<div align="center">
  <img src="assets/gifs/chatting-with-twin.gif" alt="Chatting with LLM Twin">
  <p><em>Figure 7: Chatting with the LLM Twin</em></p>
</div>

The gif above shows the user chatting with their LLM Twin. The user can ask questions and the LLM Twin will respond with a style similar to user's messaging style.

Below are some more examples of the conversations with the LLM Twin. Note that the LLM Twin picked up on the user's Singlish and the way they type. The LLM Twin also has some information about the user, such as their favourite emojis and their favourite field in computer science.

<details>
  <summary>Favourite emojis</summary>

  <div align="center">
    <img src="assets/images/fav_emojis.jpg" alt="Chatting with LLM Twin">
    <p><em>Figure 8: What are your favourite emojis?</em></p>
  </div>

</details>

<details>
  <summary>Favourite field in computer science</summary>

  <div align="center">
    <img src="assets/images/fav_ai_field.jpg" alt="Favourite field in computer science">
    <p><em>Figure 9: What is your favourite field in computer science?</em></p>
  </div>

</details>

<details>
  <summary>Want to eat?</summary>

  <div align="center">
    <img src="assets/images/want_to_eat.jpg" alt="Want to eat conversation">
    <p><em>Figure 10: Want to eat?</em></p>
  </div>
  
</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

***

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
# HUGGINGFACE_USERNAME=
# HUGGINGFACE_TOKEN=
```
_**Note:** The `HUGGINGFACE_USERNAME` and `HUGGINGFACE_TOKEN` are optional. If used,the fine tuned model will be uploaded to your Hugging Face account._

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

***

<!-- CONTACT -->
## Contact

<a href="https://github.com/shafiqninaba/resonare/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=shafiqninaba/resonare" alt="contrib.rocks image" />
</a>

Shafiq Ninaba | shafiqninaba@gmail.com | [LinkedIn](https://linkedin.com/in/shafiq-ninaba)

Zack Low | lowrenhwa88@gmail.com | [LinkedIn](https://www.linkedin.com/in/ren-hwa-low-855080224/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/shafiqninaba/resonare.svg?style=for-the-badge
[contributors-url]: https://github.com/shafiqninaba/resonare/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/shafiqninaba/resonare.svg?style=for-the-badge
[forks-url]: https://github.com/shafiqninaba/resonare/network/members
[stars-shield]: https://img.shields.io/github/stars/shafiqninaba/resonare.svg?style=for-the-badge
[stars-url]: https://github.com/shafiqninaba/resonare/stargazers
[issues-shield]: https://img.shields.io/github/issues/shafiqninaba/resonare.svg?style=for-the-badge
[issues-url]: https://github.com/shafiqninaba/resonare/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[product-screenshot]: assets/images/main_screenshot.jpg
[Python-img]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
[uv-img]: https://img.shields.io/badge/uv-package%20manager-blueviolet
[uv-url]: https://docs.astral.sh/uv/
[streamlit-img]: https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white
[streamlit-url]: https://streamlit.io/

