from qdrant_client import QdrantClient, models
import qdrant_client
from fastapi import FastAPI, HTTPException
from botocore.exceptions import ClientError
import boto3
from dotenv import load_dotenv
import os
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings
from llama_index.core import Document
from fastembed import TextEmbedding
import shutil

load_dotenv()

app = FastAPI()

# Retrieve Qdrant Cloud endpoint URL and API key from environment variables
qdrant_cloud_endpoint = os.getenv("QDRANT_CLOUD_ENDPOINT")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
collection_name = os.getenv("QDRANT_COLLECTION_NAME")

# Initialize AsyncQdrantClient with the Qdrant Cloud endpoint and API key
qdrant_client = QdrantClient(
    qdrant_cloud_endpoint,
    api_key=qdrant_api_key,
)

# Retrieve AWS credentials and S3 bucket configuration from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_FOLDER_NAME = os.getenv("S3_FOLDER_NAME")
S3_LOGFILE_NAME = "logs.txt"  

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")



def pdf_ingest():
    """
    Function to ingest PDF documents from an S3 bucket, process them, and store information in Qdrant Cloud.

    Takes:
    - No arguments.

    Returns:
    - Logs the processing status and errors if any.

    Steps:
    1. Retrieves list of all folders from the specified S3 bucket.
    2. Checks for the existence of a log file and creates one if not found.
    3. Determines unprocessed folders by comparing with those listed in the log file.
    4. Iterates through each unprocessed folder.
    5. Downloads PDF files from the folder and processes them.
    6. Stores information in Qdrant Cloud.
    7. Updates the log file with processed folder names.
    8. Handles and logs errors if any occur.
    """
    
    try:
    # List all objects in the 'test' folder
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=f"{S3_FOLDER_NAME}")

        # Extract folder names from response
        if 'Contents' in response:
            all_folders = {obj['Key'].split('/')[1] for obj in response['Contents']}
        else:
            all_folders = set()

        print("All folders in S3:", all_folders)

        # Check if the log file exists
        try:
            log_response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"{S3_FOLDER_NAME}/{S3_LOGFILE_NAME}")
        except s3_client.exceptions.NoSuchKey:
            fk=f"{S3_FOLDER_NAME}/{S3_LOGFILE_NAME}"
            empty_content = b'' # Empty content for the log file
            s3_client.put_object(
                                Bucket=S3_BUCKET_NAME,
                                Key=fk,
                                Body=empty_content
                            )
            processed_folders = set()
        else:
            lines = log_response['Body'].read().decode('utf-8').splitlines()
            processed_folders = {line.strip() for line in lines if line.strip()}

        print("Processed folders:", processed_folders)

        # Find folders not processed yet
        not_processed_folders = all_folders - processed_folders
        not_processed_folders = [folder for folder in not_processed_folders if folder.strip() and folder != "logs.txt" and folder != "error.txt" and not folder.endswith('.csv')]
        print("unprocessed folders",not_processed_folders)
        # Iterate through unprocessed folders
        for folder in not_processed_folders:
            # List objects in the current folder
            folder_response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=f"{S3_FOLDER_NAME}/{folder}")

            # Extract JSON files
            if 'Contents' in folder_response:
               pdf_files = [obj['Key'] for obj in folder_response['Contents'] if obj['Key'].endswith('.pdf')]
               print("list of all pdfs",pdf_files)
               if not os.path.exists("downloads"):
                    os.makedirs("downloads")
               for pdf_file in pdf_files:
                   # Extract file name
                    file_name = os.path.basename(pdf_file)
                    print(file_name)
                    # Download PDF file from S3
                    local_file_path = f"downloads/{file_name}"  # Change this to your desired local directory
                    s3_client.download_file(S3_BUCKET_NAME, pdf_file, local_file_path)
               documents = SimpleDirectoryReader("./downloads/").load_data()
               vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)
               storage_context = StorageContext.from_defaults(vector_store = vector_store)
               index = VectorStoreIndex.from_documents(documents, vector_store=vector_store, storage_context = storage_context, embed_model = embed_model)
               print(index)
               if os.path.exists("downloads"):
                shutil.rmtree("downloads")
               print("removed downloads")
            # After processing each folder, append its name to logs.txt
            processed_folders.add(folder)
            # Read the current content of logs.txt
            log_content = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f"{S3_FOLDER_NAME}/{S3_LOGFILE_NAME}")['Body'].read().decode('utf-8')
            # Append the new folder name to the content
            updated_content = log_content + '\n' + folder
            # Write the updated content back to logs.txt
            s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=f"{S3_FOLDER_NAME}/{S3_LOGFILE_NAME}", Body=updated_content)
        
    except ClientError as e:
        error_message = f"An error occurred while processing S3 logs: {str(e)}"
        print(error_message) # Log the error
        raise HTTPException(status_code=500, detail=error_message)
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message) # Log the error
        raise HTTPException(status_code=500, detail=error_message)
        

pdf_ingest()

