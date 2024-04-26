import json
import os
import os.path
import constants as c
import pandas as pd
import requests
from fastapi import FastAPI
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse,FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from collections import OrderedDict
from operator import getitem

DEBUG = False
DEMO_MODE = True
if ("DEBUG" in os.environ):
    DEBUG = os.environ["DEBUG"]

df=pd.read_csv("Results/ClaimsAmountResults.csv")

# A Pydantic model
class PremiumParameter(BaseModel):
    alcohol: str ="No"
    years_of_insurance_with_us:int = 0
    regular_checkup_last_year:int = 0
    adventure_sports:str = "No"
    occupation:str = "Salaried"
    visited_doctor_last_1_year:int = 0
    cholesterol_level:str = "125 to 150"
    daily_avg_steps:int= 0
    age:int = 16
    heart_decs_history:str = "No"
    any_other_major_decs_history:str = "No"
    gender:str ="Male"
    avg_glucose_level:int =90
    smoking_status:str= "Unknown"
    weight:int = 68
    height:int = 150
    covered_by_any_other_company:str="N"
    exercise:str = "No"
class InvalidWMLzPasswordException(Exception):
    "Raised when the WMLz Password has been expired"
    pass

# Logging
LOGLEVEL = os.environ.get("LOGLEVEL", "DEBUG" if DEBUG else "INFO").upper()
logger.setLevel(LOGLEVEL)

app = FastAPI(debug=True)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    #allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")


def triton_prediction(prediction_input):
    url = "http://localhost:8000/v2/models/rf_model/infer"
    if ('SCORING_URL' in os.environ):
       url = os.environ['SCORING_URL']
    payload = {
    "inputs": [
        {
        "name": "IN0",
        "shape": [
            1,
            23
        ],
        "datatype": "BYTES",
        "data": [
            prediction_input
        ]
        }
    ],
    "outputs": [
        {
        "name": "OUT0"
        }
    ]
    }

    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.post(url, json=payload)
    resposne_result = response.json()
    
    if resposne_result["outputs"]:
        return resposne_result["outputs"][0]["data"][0]
    else:
        print("Issue with Prediction")
        exit(16)
@app.get("/predict/stats")


def get_statistics(selectedCities: str = None,
                   healthIndicator: str = None,
                   policyType: str = None,
                   policyDuration: str = None):
    
    PolicyType=["Care Supreme Plan","Basic Health Coverage","Critical Illness Plan","Care Plus Plan"]
    HealthIndicator=["Terminal Health","Critical Health","Very Poor Health","Poor Health","Average Health","Fair Health","Good Health","Very Good Health","Excellent Health"]
    CityName=["Baltimore","Toronto","Riverside","Boston","Miami","Berkley","Philadelphia","Roseville","Ontario","Columbia","Indianapolis","Madison","Memphis","Belleville","Santa Maria","New York","Dallas","Austin","San Diego","Lakewood","Springfield","Vancouver","San Jose","Pittsburg","Chicago","Houston","Los Angeles","Phoenix","Oakland","Cleveland","Seattle","West Covina","Clearwater","West Valley City","Concord","Norman"]
    PolicyDuration=["1","2","3","4","5","6","7","8","9","10","11","12","13","14","14+"]
    
    #if any of the fields in not selected by default it is All available values for 
    #for each filter
    if(selectedCities != None):
        selectedCities_sel_arr = [c.strip() for c in selectedCities.split(",")]
    else:
        selectedCities_sel_arr = CityName
    
    if(healthIndicator != None):
        healthIndicator_sel_arr = [c.strip() for c in healthIndicator.split(",")]
    else:
        healthIndicator_sel_arr = HealthIndicator
    
    if(policyType != None):
        policyType_sel_arr = [c.strip() for c in policyType.split(",")]#policyType.split(",")
    else:
        policyType_sel_arr = PolicyType

    if(policyDuration != None):
         policyDuration_sel_arr = [c.strip() for c in policyDuration.split(",")]
    else:
        policyDuration_sel_arr = PolicyDuration

    #find number of elements in selection array 
    polType_sel_len = len(policyType_sel_arr)
    HI_sel_len = len(healthIndicator_sel_arr)
    polDur_sel_len = len(policyDuration_sel_arr)
    city_sel_len = len(selectedCities_sel_arr)
    

    # print("Selected cities is :"
    # ,selectedCities_sel_arr)
    # print("Selected HI is :",healthIndicator_sel_arr)
    # print("Selected Pol.Type is :",policyType_sel_arr)
    #print("Selected Pol.Dur is :",policyDuration_sel_arr)
   
    row = df[(df["City_Name"].isin(selectedCities_sel_arr)) & (df["Health Indicator"].isin(healthIndicator_sel_arr))
              & (df["Holding_Policy_Type_Desc"].isin(policyType_sel_arr)) & (df["Holding_Policy_Duration"].isin(policyDuration_sel_arr))]
    
    
    #Number of policy holders]
    Num_policy_holders = row.shape[0]

    #Number of claims
    claims = row[(row["Response"] == 1)]
    Num_Claims= claims.shape[0]
    # print("Claims unique:")
    # print(claims.City_Name.unique())
    #Tot Claim amount
    Claim_amount = round(claims["Claim amount"].sum()/1000000,2)
    AvgClaimAmtForecast = round((claims["ClaimAmtPredicted"].sum()//Num_Claims)//1000)
    AvgClaimAmtcurrent = round((claims["Claim amount"].sum()//Num_Claims)//1000)
    AvgGap = AvgClaimAmtcurrent - AvgClaimAmtForecast
   
    #claims and claims amount by Policy Type
    if(policyType_sel_arr!=None and len(policyType_sel_arr)>0):
        PolicyType = set(policyType_sel_arr).intersection(PolicyType)
        print("Policy Type Printing:",PolicyType)

    PT_Dict = {}
    for i in PolicyType:
        claims_by_policyType = claims[(claims["Holding_Policy_Type_Desc"] == i)]
        claims_number = claims_by_policyType.shape[0]
        amt = round(claims_by_policyType['ClaimAmtPredicted'].sum()/1000000)
        PT_Dict[i] ={"number":claims_number,"amount":amt}
        PT_Dict = OrderedDict(sorted(PT_Dict.items(),key = lambda x: getitem(x[1], 'amount'),reverse=True))

    #claims and claims amount by Health Indicator
    # HealthIndicator=["X1","X2","X3","X4","X5","X6","X7","X8","X9"]
    if(healthIndicator_sel_arr!=None and len(healthIndicator_sel_arr)>0):
        HealthIndicator = set(healthIndicator_sel_arr).intersection(HealthIndicator)
    HI_Dict = {}
    for i in HealthIndicator:
        claims_by_HI = claims[(claims["Health Indicator"] == i)]
        claims_number = claims_by_HI.shape[0]

        amt = round(claims_by_HI['ClaimAmtPredicted'].sum()/1000000)
        HI_Dict[i] ={"number":claims_number,"amount":amt}
    HI_Dict = OrderedDict(sorted(HI_Dict.items(),key = lambda x: getitem(x[1], 'amount'),reverse=True))
    
    #claims and claims amount by CityName
     
    if(selectedCities_sel_arr!=None and len(selectedCities_sel_arr)>0):
        CityName = set(selectedCities_sel_arr).intersection(CityName)
    City_Dict = {}
    for i in CityName:
        claims_by_CityName = claims[(claims["City_Name"] == i)]
        claims_number = claims_by_CityName.shape[0]
        amt = round(claims_by_CityName['ClaimAmtPredicted'].sum()/1000000)
        City_Dict[i] ={"number":claims_number,"amount":amt}
    City_Dict = OrderedDict(sorted(City_Dict.items(),key = lambda x: getitem(x[1], 'amount'),reverse=True))
    


    #claims and claims amount by Policy Duration
    if(policyDuration_sel_arr!=None and len(policyDuration_sel_arr)>0):
        PolicyDuration = set(policyDuration_sel_arr).intersection(PolicyDuration)
        print("Policy Duration Printing:",PolicyDuration)

    PD_Dict = {}
    for i in PolicyDuration:
        claims_by_policyDuration = claims[(claims["Holding_Policy_Duration"] == i)]
        claims_number = claims_by_policyDuration.shape[0]
        amt = round(claims_by_policyDuration['ClaimAmtPredicted'].sum()/1000000)
        tag = i+" years"
        PD_Dict[tag] ={"number":claims_number,"amount":amt}
    PD_Dict = OrderedDict(sorted(PD_Dict.items(),key = lambda x: getitem(x[1], 'amount'),reverse=True))
    print('PD_Dict',PD_Dict)

    #code to return values , return only non null values
    #  print(numDict)
    result = {}
    statistics = {}
    result['policyHolder'] = str(Num_policy_holders)
    result['claim'] = str(Num_Claims)
    result['claimAmount'] = {"amount":Claim_amount,"unit":"$M"}
    result['current'] = {"amount":AvgClaimAmtcurrent,"unit":"$K"}
    result['forecast'] = {"amount":AvgClaimAmtForecast,"unit":"$K"}
    result['gap'] = {"amount":AvgGap,"unit":"$K"}
    
    result["statistics"]=statistics

    policyType = {}
    statistics["policyType"] = PT_Dict
   
    HealthIndicator = {}
    statistics["healthIndicator"] = HI_Dict

    city={}
    statistics["city"] = City_Dict

    policyDuration = {}
    statistics["policyDuration"] = PD_Dict

    return result
            
            
@app.get("/download/stats")
def download_statistics(selectedCities: str = None,
                        healthIndicator: str = None,
                        policyType: str = None,
                        policyDuration: str = None):
    print(selectedCities,
          healthIndicator,
          policyType,
          policyDuration)
    filename = "health_insurance_rpt.csv"
    fields = ["Number of test policy holders", "Amount"]

    # data rows of csv file
    rows = [["1234", "3435"]]

    # writing to s file
    with open(filename, "w") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(rows)
    return FileResponse(path=filename, filename=filename, media_type="application/octet-stream")

@app.post("/predict/premium")
def post_injectiondata(inputData:PremiumParameter):
    #convert cms to meters
    height = inputData.height/100
    BMI = inputData.weight/(height*height)
    applicant_id = 0
    location = "Bangalore"
    year_last_admitted = "2000"
    weight_change_in_last_one_year = 30
    fat_percentage = 20

    print('BMI parameters: ', inputData.weight, height, BMI)
    data = {
        "Alcohol":inputData.alcohol,
        "years_of_insurance_with_us":inputData.years_of_insurance_with_us,
        "regular_checkup_last_year":inputData.regular_checkup_last_year,
        "adventure_sports":inputData.adventure_sports,
        "Occupation":inputData.occupation,
        "visited_doctor_last_1_year":inputData.visited_doctor_last_1_year,
        "cholesterol_level":inputData.cholesterol_level,
        "daily_avg_steps":inputData.daily_avg_steps,
        "age":inputData.age,
        "heart_decs_history":inputData.heart_decs_history,
        "Any_other_major_decs_history":inputData.any_other_major_decs_history,
        "Gender":inputData.gender,
        "avg_glucose_level":inputData.avg_glucose_level,
        # "bmi":inputData.BMI,
        "smoking_status":inputData.smoking_status,
        "weight":inputData.weight,
        "bmi":BMI,
        "covered_by_any_other_company":inputData.covered_by_any_other_company,
        "exercise":inputData.exercise,

        "applicant_id":0,
        "fat_percentage":0,
        "Location":"Bangalore",
        "weight_change_in_last_one_year":0,
        "Year_last_admitted":0
    }

    input_data = [ str(applicant_id), str(inputData.years_of_insurance_with_us), str(inputData.regular_checkup_last_year), str(inputData.adventure_sports), str(inputData.occupation), str(inputData.visited_doctor_last_1_year), str(inputData.cholesterol_level), str(inputData.daily_avg_steps), str(inputData.age), str(inputData.heart_decs_history), str(inputData.any_other_major_decs_history), str(inputData.gender), str(inputData.avg_glucose_level), str(BMI), str(inputData.smoking_status), str(year_last_admitted), str(location), str(inputData.weight), str(inputData.covered_by_any_other_company), str(inputData.alcohol), str(inputData.exercise), str(weight_change_in_last_one_year), str(fat_percentage) ]

    headers = {"Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,POST,PATCH,OPTIONS"}
    try:
        updated_data = triton_prediction(input_data)
        predPremValue = "$"+ str(round(updated_data,2))
        predPremValue = {"premium":predPremValue}
        print(predPremValue)
    except InvalidWMLzPasswordException:
        return JSONResponse(content="Invalid WMLz Password. Please contact the administrator", status_code=400)
   
    #if predPremValue -ve return 0
    print(round(updated_data,2))
    if(round(updated_data,2) <0):
        return {"premium":"$"+str(0)}
    else:
        return predPremValue

    # return JSONResponse(content=predPremValue, status_code=200, headers=headers)
#uvicorn main:app --host 0.0.0.0 --port 80 --reloads
