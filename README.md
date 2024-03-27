# Medical Regulation API

This is a simple API that extracts regulations related to a given drug. The API is built using FastAPI, a modern, fast (high-performance), web framework for building APIs with Python 3.7+.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Fatoumata964/MedicalReglementation.git
   cd MedicalReglementation
    ```
   
2. Download bioSentvec model from: https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin
   load it in /models.

3. Activate virtual env: if you are not using pycharm or visual studio activate venv manually
    ```bash
    python -m venv venv
   source venv/bin/activate
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Run the API

To run the API, execute the following command:

```bash
uvicorn src.medical_regulation:app --reload
```

The API will be accessible at `http://127.0.0.1:8000/`.

### API Endpoints

- **Default Endpoint:**

  - Endpoint: `/`
  - Method: `GET`
  - Description: Default endpoint for the API.
  - Response:
    ```json
    {
        "version": "0.1.0",
        "documentation": "/docs"
    }
    ```

- **Get Regulation Endpoint:**

  - Endpoint: `/apiv1/regulation/get-regulation`
  - Method: `POST`
  - Description: Accepts POST requests with JSON data containing the drug text.
  - Request:
    ```json
    {
        "drug": "Your Drug Text Here"
    }
    ```
  - Response:
    ```json
    {
        "regulation": "Extracted Regulation Text Here"
    }
    ```

## Customization

Feel free to add your implementation for the `extract_regulation` function in the `src/medical_regulation.py` file to suit your specific requirements.

## Dependencies

- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
