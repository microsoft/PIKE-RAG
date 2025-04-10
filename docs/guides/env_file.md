# .env File Format

Please follow below environment configuration variable names to create your *.env* file, we suggest you put it under
`PATH-TO-PIKE-RAG/env_configs/` which has already been added to *.gitignore* file:

## For Azure OpenAI Client

The `AzureOpenAIClient` will indeed create an `openai.AzureOpenAI` client inside. The client instance parameters can be provided either in the `client_configs` of client in yaml config or be set in environment variables. As `openai.AzureOpenAI` introduced, it automatically infers the following arguments from their corresponding environment variables if they are not provided:

- `api_key` from `AZURE_OPENAI_API_KEY`
- `organization` from `OPENAI_ORG_ID`
- `project` from `OPENAI_PROJECT_ID`
- `azure_ad_token` from `AZURE_OPENAI_AD_TOKEN`
- `api_version` from `OPENAI_API_VERSION`
- `azure_endpoint` from `AZURE_OPENAI_ENDPOINT`

To be specific, if you have the access API key, you can provide them either in `client_configs` or set them in *.env* file as follow:

    ```ini
    OPENAI_API_TYPE = "azure"
    AZURE_OPENAI_ENDPOINT = "YOUR-ENDPOINT(https://xxx.openai.azure.com/)"
    OPENAI_API_VERSION = "2024-08-01-preview"  # Just an example value here.
    AZURE_OPENAI_API_KEY = "YOUR-API-KEY"
    ```

In case you don't have the API key due to the security mechanism of Azure, the `AZURE_BEARER_TOKEN_SCOPE` should be specified, otherwise *"https://cognitiveservices.azure.com/.default"* would be used as default scope. Now your *.env* file should be set as follow:

    ```ini
    OPENAI_API_TYPE = "azure"
    AZURE_OPENAI_ENDPOINT = "YOUR-ENDPOINT(https://xxx.openai.azure.com/)"
    OPENAI_API_VERSION = "2024-08-01-preview"  # Just an example value here.
    AZURE_BEARER_TOKEN_SCOPE = "YOUR-TOKEN-SCOPE-ADDRESS"
    ```

To access with token provider, please remember to login your device to Azure CLI using your valid account first, we provided two simple scripts in *scripts/*:

    ```sh
    # Install Azure-CLI and other dependencies. Sudo permission is required.
    bash scripts/install_az.sh

    # Login Azure CLI using device code.
    bash scripts/login_az.sh
    ```

In case the `azure_deployment` is necessary to initialize the client, you can specify it in `client_configs` or set the environment variable `AZURE_DEPLOYMENT_NAME` in *.env* file, which results a *.env* below:

    ```ini
    OPENAI_API_TYPE = "azure"
    AZURE_OPENAI_ENDPOINT = "YOUR-ENDPOINT(https://xxx.openai.azure.com/)"
    OPENAI_API_VERSION = "2024-08-01-preview"  # Just an example value here.
    AZURE_BEARER_TOKEN_SCOPE = "YOUR-TOKEN-SCOPE-ADDRESS"  # Or the `AZURE_OPENAI_API_KEY`, or the `AZURE_OPENAI_AD_TOKEN` here.
    AZURE_DEPLOYMENT_NAME = "gpt-4_1106-Preview"  # Just an example value here.
    ```

## For Azure Meta LlaMa Client

Since the endpoint and API keys varied among different LlaMa models, you can add multiple
(`llama_endpoint_name`, `llama_key_name`) pairs you want to use into the *.env* file, and specify the names when
initializing `AzureMetaLlamaClient` (you can modify the llm client args in the YAML files). If `null` is set to be the
name, the (`LLAMA_ENDPOINT`, `LLAMA_API_KEY`) would be used as the default environment variable name.

    ```ini
    # Option 1: Set only one pair in one time, update these variables every time you want to change the LlaMa model.
    LLAMA_ENDPOINT = "YOUR-LLAMA-ENDPOINT"
    LLAMA_API_KEY = "YOUR-API-KEY"

    # Option 2: Add multiple pairs into the .env file, for example:
    LLAMA3_8B_ENDPOINT = "..."
    LLAMA3_8B_API_KEY = "..."

    LLAMA3_70B_ENDPOINT = "..."
    LLAMA3_70B_API_KEY = "..."
    ```

### Ways to Get the Available Azure Meta LLaMa **Endpoints**, **API Keys** and **Model Names**

The way we have implemented the LLaMa model so far involves requesting the deployed model on the GCR server. You can
find the available settings follow the steps below:

1. Open [Azure Machine Learning Studio](https://ml.azure.com/home), sign in may be required;
2. Click *Workspaces* on the left side (expand the menu by clicking the three horizontal lines in the top left corner if
you cannot find it);
3. Choose and click on a valid workspace, e.g., *gcrllm2ws*;
4. Click *Endpoints* on the left side (expand the menu by clicking the three horizontal lines in the top left corner if
you cannot find it), You can find the available model list in this page;
5. Choose and click the model you want to use, e.g., *gcr-llama-3-8b-instruct*:
    - **model** name: in tab "Details", scroll to find "Deployment summary", the *Live traffic allocation* string (e.g.,
        *meta-llama-3-8b-instruct-4*) is the model name you need to set up in your YAML file;
    - **LLAMA_ENDPOINT** & **LLAMA_API_KEY**: can be found in tab "Consume".

### Handling the Issue "Specified deployment could not be found"

If you get error message "Specified deployment could not be found", it indicates that the GCR team has changed the
server deployment location. In this case, you need to check the available model list in
[Azure Machine Learning Studio](https://ml.azure.com/home) and update the YAML config again.

*Return to the main [README](https://github.com/microsoft/PIKE-RAG/blob/main/README.md)*
