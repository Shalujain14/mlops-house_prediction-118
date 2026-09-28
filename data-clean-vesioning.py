import pandas as pd
import boto3
from datetime import date

# load raw csv from local resource
path=r"C:\Users\HP\Downloads\Mlops_house_predication_clean_v1.csv"
data=pd.read_csv(path)
df=pd.DataFrame(data)
print("==============Before cleaning=================")
print(df.isnull().sum())
print(f"Shape Before : {df.shape}")
print()
df_clean=df.dropna()
print("===============After cleaning==================")
print(df_clean.isnull().sum())
print(f"Shape After : {df_clean.shape}")

# save clean csv locally
clean_path=r"C:\Users\HP\Downloads\Mlops_house_predication_clean_v2.csv"
df_clean.to_csv(clean_path,index=False)


# upload to s3 as processed version
s3=boto3.client('s3')
BUCKET="mlops-house-prediction-118"
def upload_proccessed_data(local_path):
    key=f"processes/{date.today()}/Mlops_house_prediction_clean_v2.csv"
    s3.upload_file(local_path,BUCKET,key)
    print(f"\nuploaded to s3://{BUCKET}/{key}")
    return key

upload_proccessed_data(clean_path)