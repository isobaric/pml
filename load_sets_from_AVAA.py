#!/usr/bin/env python3
"""
Script used to download and format my training and test sets.
Only tested with:
Python 3.5.2
Pandas 0.17.1
"""

import requests
import pandas
from functools import partial
from io import StringIO

template = r'http://avaa.tdata.fi/palvelut/smeardata.jsp?variables={variablenames}&table=HYY_META&from={startyear}-01-01 00:00:00.000&to={endyear}-01-01 00:00:00.000&quality=ANY&averaging=30MIN&type=ARITHMETIC'

startyear = 2000
endyear = 2016

testyear = 2016

#Because it also takes the very first measurement of the next year.
endyear = endyear + 1 

variables = {"T504": "T lower", "T672":"T", "Pamb0":"Air pressure", "Net":"Net radiation"}

#this determines the order of the received data columns
variablestring = "T504,T672,Pamb0,Net"

#It is not a large service, so trying to download too much would be inconsiderate
#The API is subject to change, and has already done so once during this project work
    
request = template.format(**{"variablenames":variablestring, "startyear":startyear, "endyear":endyear})
print(request)

response = requests.get(request)

#well, not much we can do if there is an HTTP code other than success
if response.status_code != 200:
    raise ValueError("HTTP status code abnormal: " + str(response.status_code))

#save data in current folder in case we find trouble later
with open("./raw_avaa.csv", "w") as outfile:
    outfile.write(response.text)

databuffer = StringIO(response.text)
#read into pandas dataframe, the date_parser seems somewhat inelegant, but does the job.
date_format = "%Y %m %d %H %M %S"
full_data = pandas.read_csv(databuffer, parse_dates={"Time":["Year", "Month", "Day", "Hour", "Minute", "Second"]}, date_parser=partial(pandas.to_datetime, format=date_format), index_col='Time')

#The original column names could work as well, but referring to things as tablename.variablename is a bit annoying
#None of the data I use has a . in the variablename.
def drop_table_name(name):
    if name.startswith("HYY_META"):
        return variables[name.split(".")[1]]
    else:
        return name
    
full_data.rename(columns=drop_table_name, inplace=True)

#Clean rows where any of the values are missing (because the model is useless there)

full_data.dropna(inplace=True)

#create new variables fraction of day [0,1), fraction of year [0,~1)
full_data['day fraction'] = pandas.Series((full_data.index.hour + full_data.index.minute/60.0)/24.0, index=full_data.index)
#I don't really care that on leap years this a bit over 1.
full_data['year fraction'] = full_data.index.dayofyear/365.0

#split off test year
test_data = full_data[full_data.index.year == testyear]
training_data =  full_data[full_data.index.year != testyear]

#store for training script
training_data.to_csv("./training_data.csv")
test_data.to_csv("./test_data.csv")
