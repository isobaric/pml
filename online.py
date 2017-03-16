#!/usr/bin/env python3

from datetime import datetime,date,timedelta
import numpy as np
import pandas
import urllib.request
import matplotlib as mpl
import sys
import xmlrpc.client
from functools import partial

mpl.use('Agg')
mpl.rcParams.update({'font.size': 24, "figure.figsize":(30,20)})
import matplotlib.pyplot as plt

coeff = np.array([9.97300534e-01,  -5.83935333e-04])
intercept = -0.13252711030957354

template = r'http://avaa.tdata.fi/palvelut/smeardata.jsp?variables=T672,T504,Net&table=HYY_META&from={startdate}+00:00:00.000&to={enddate}+00:00:00.000&quality=ANY&averaging=30MIN&type=ARITHMETIC'

date_format = "%Y %m %d %H %M %S"

def gather_data(current_date):
    week_past = current_date - timedelta(days=8)
    filled_request = template.format(**{"startdate":week_past, "enddate":current_date})
    print(filled_request)
    try:
        response = urllib.request.urlopen(filled_request)
    except HTTPError as error:
        success = False
        return [], success

    dataframe = pandas.read_csv(response, parse_dates={"Time":["Year", "Month", "Day", "Hour", "Minute", "Second"]},
                                date_parser=partial(pandas.to_datetime, format=date_format), index_col='Time')
    
    return dataframe, True

def make_prediction(dataframe):
    data = dataframe[["HYY_META.T504", "HYY_META.Net"]]
    prediction = np.sum(data.values * coeff[np.newaxis, :], axis=1) + intercept
    mean_diff = np.mean(np.abs(dataframe["HYY_META.T672"] - prediction))
    max_diff = np.max(np.abs(dataframe["HYY_META.T672"] - prediction))

    return prediction, mean_diff, max_diff

def make_the_plot(dataframe, prediction, mean_diff, max_diff):
    fig = plt.figure()
    plt.plot(dataframe.index, dataframe["HYY_META.T672"].values, 'rx--', label="Measured")
    plt.plot(dataframe.index, prediction, 'b+--', label="Predicted")
    plt.legend(loc='best')
    plt.ylabel("Temperature [Celcius]")
    plt.title("Prediction, Error mean: "+str(mean_diff)+",max:"+str(max_diff))
    plt.savefig("./current_result.png")
    plt.close()

def check_state(current_date):

    page_id = omitted_data["page_id"]

    proxy = xmlrpc.client.ServerProxy(omitted_data["URL"])

    auth_token = proxy.confluence2.login(omitted_data["LOGIN"],
                                         omitted_data["PASSWORD"])

    page = proxy.confluence2.getPage(auth_token, page_id)
    
    attachment = proxy.confluence2.getAttachment(auth_token, page_id, "current_result.png", "0")

    time_tuple = attachment["created"].timetuple()
    modification_date = date(time_tuple.tm_year, time_tuple.tm_mon, time_tuple.tm_mday)

    proxy.confluence2.logout(auth_token)

    if modification_date == current_date:
        return True

    return False

def send_to_webpage():

    page_id=omitted_data["page_id"]

    proxy = xmlrpc.client.ServerProxy(omitted_data["URL"])
 
    auth_token = proxy.confluence2.login(omitted_data["LOGIN"],
                                         omitted_data["PASSWORD"])

    page = proxy.confluence2.getPage(auth_token, page_id)

    proxy.confluence2.removeAttachment(auth_token, page_id, "current_result.png")
    #add new in its place
    with open("./current_result.png", "rb") as infile:
        data = infile.read()

    proxy.confluence2.addAttachment(auth_token, page_id, attachment, bytearray(data))
        
    proxy.confluence2.logout(auth_token)
        

if __name__ == "__main__":
    if len(sys.argv) > 2:
        current_date = date(*[int(item) for item in sys.argv[2:]])
    else:
        current_date = date.today()


    # Read secrets for authentication
    with open(sys.argv[1], 'r') as infile:
        split_lines = (line.strip("\n").split(",") for line in infile)
        omitted_data = {name:value for name,value in split_lines} 

    attachment = {'contentType': 'image/png',
                  'fileName': 'current_result.png',
                  'pageId': omitted_data["page_id"]}

    start_time = str(datetime.now())

    if check_state(current_date):
        print(start_time+": Already updated today.")
        sys.exit()

    df, success = gather_data(current_date)
    if success:
        prediction, mean, maximum = make_prediction(df)
        make_the_plot(df, prediction, mean, maximum)
        send_to_webpage()
        print(start_time+": Updated image.")
    else:
        print(start_time+ ": Failure in data acquisition.")
