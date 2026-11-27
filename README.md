# Agentic-Chatbot-using-LangGraph
Learning Agentic AI Concepts and implementing in projects

---

# Agentic-Chatbot-APP-CICD-Deployment-with-Github-Actions-on-AWS

### Description: About the deployment

1. Build docker image of the source code
2. Push your docker image to Docker Hub
3. Launch your EC2
4. Pull your image from Docker Hub in EC2
5. Launch your docker image in EC2

## 1. Login to AWS console

## 2. Create IAM user for deployment

**Policy:**

1. AmazonEC2FullAccess

## 3. Create EC2 machine (Ubuntu)

## 4. Open EC2 and install Docker in EC2 machine

```bash
# optional
sudo apt-get update -y
sudo apt-get upgrade

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

> **Note:** Do the port mapping to this port: `8501`

## 5. Configure EC2 as self-hosted runner

`Settings > Actions > Runners > New self-hosted runner >` choose OS, then run the commands one by one.

## 6. How to add secret keys to GitHub Actions

Add these under `Settings > Secrets and variables > Actions > New repository secret`.

**Required** (the workflow fails if any of these are missing):

```env
DOCKER_USERNAME=your-dockerhub-username
DOCKER_PASSWORD=your-dockerhub-access-token
IMAGE_NAME=agentic-chatbot

GOOGLE_API_KEY=your-google-api-key
TAVILY_API_KEY=your-tavily-api-key
OPENWEATHER_API_KEY=your-openweather-api-key

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=agentic-chatbot-project
```

> `GOOGLE_API_KEY` is also passed to the app as `GEMINI_API_KEY`, so you only need to set it once.

**Optional** (leave unset to disable the feature):

```env
GROQ_API_KEY=your-groq-api-key              # Groq fallback model
STOCK_API_KEY=your-stock-api-key            # stock price tool
CURRENT_LOCATION_URL=your-location-api-url  # current location tool
```

> The workflow deploys through the self-hosted runner on EC2, so it does not use AWS access keys or a `REGISTRY` secret.

> Chat history and FAISS indexes are stored in the Docker volume `agentic-chatbot-data`, so they persist across deployments.
