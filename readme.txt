
Step1: What will LSTM do?
      1. LSTM will accept the GitHub data from flask microservice and will forecast the data for past 1 year based on past 30 days
      2. It will also plot three different graph (i.e.  "model_loss", "lstm_generated_data", "all_issues_data")using matplot lib 
      3. This graph will be stored as image in gcloud storage.
      4. The image URL are then returned back to flask microservice.

Step2: What is Google Cloud Storage?
       Google Cloud Storage is a RESTful online file storage web service for storing and accessing data on Google Cloud
       Platform infrastructure.    


Step3: Deploying LSTM to gcloud platform
       1: You must have Docker(https://www.docker.com/get-started) and Google Cloud SDK(https://cloud.google.com/sdk/docs/install) 
           installed on your computer. Then, Create a gcloud project FOR LSTM  and enable the billing account. 
       2. Steps to follow while creating LSTM gcloud project:
            1. Go to GCP Platform, create a LSTM gcloud project.
            2. After creating the LSTM gcloud project, go to gcloud storage of LSTM gcloud project then click on 
               create bucket, then add name to your bucket click on continue, then choose where to store your data, there 
               choose location type to "region" and location to "us-central1(Iowa)", then click on continue, then Choose a default storage class for your data
               and choose option "standard" and click continue and then click on create then ypu will automatically navigated to "bucket details", there you 
               will see "objects" and in "objects" you will see "buckets" click on "buckets", you will be able to see your assigned bucket name with checkbox
               click on checkbox, then on the right side you will be able to see "permission" and "labels",in "permission" scroll down you will be able to see
               "ADD PRINCIPAL" click on "ADD PRINCIPAL" you will be able to see "new principals" and "select a role", in "new principals" type "allUsers" and in 
               "select a role" go to "cloud storage" click on "cloud storage" and select "storage object viewer" and hit on save.
            3. After this, go to "https://cloud.google.com/docs/authentication/getting-started#create-service-account-console"  and 
               there you will see creating service account, go to creating service account, click on console, and hit on creating service account
            4. Then select your LSTM created project, in service account details add service name "lstm-github-forecasting" and click on "create and 
               continue", then in Grant this service account access to project click on "select the role",choose "basic" and choose "owner"and hit on done.
            5. After that you will see your created service, click on the created service, go to keys, click on add key, and click create new key.
            6. Created key will get downloaded in .json format, copy that downloaded file in the given LSTM code
            7. Then, on cmd terminal type "set GOOGLE_APPLICATION_CREDENTIALS=KEY_PATH" (Replace KEY_PATH with the path of the JSON file that contains your service account key.)

       3: Type `docker` on cmd terminal and press enter to get all required information

       4: Type `docker build .` on cmd to build a docker image

       5: Type `docker images` on cmd to see our first docker image. After hitting enter, newest created image will be always on the top of the list

       6: Now type `docker tag <your newest image id> gcr.io/<your project-id>/<project-name>` and hit enter 
            Type `docker images` to see your image id updated with tag name

       7: Type `gcloud init` on cmd

       8: Type `gcloud auth configure-docker` on cmd

       9: Go to your GCloud account and open container registry

       10: Enable your billing account

       11: Enable your Container Registry API

       12: Go to the Cloud Build and enable Cloud Build API

       13: Type `docker push <your newest created tag>` on cmd and hit enter

       14: Go to cloud run and create new service, service name will be your GCloud project name and for container image url 
            hit select and selects your latest id and hit select and edit container port to '8080', increase the memory limit 
            to 1GiB and go to variable and secrets and click on add environment variable as follows(there will three environment variable):
                Name                                 value  
            a. GOOGLE_APPLICATION_CREDENTIALS     "<yourjsonfile>.json"
            b. BASE_IMAGE_PATH                    "https://storage.googleapis.com/your bucket name/"
            c. BUCKET_NAME                         "your bucket name"

       16: Hit the create, this will create the service on port 8080 and will generate the url, hit the url.

     
Step4: To run locally:
       1. Go to cmd terminal and type following:
        a. python -m venv env
        b. env\Scripts\activate.bat
        c. pip install -r requirements.txt
        d. python app.py  