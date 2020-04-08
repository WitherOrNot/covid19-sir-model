from wget import download
from gen_model import fit_sir
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import csv

def subtract_dates(date1, date2):
    delta = datetime.datetime.strptime(date1, "%m/%d") - datetime.datetime.strptime(date2, "%m/%d")
    return delta.days

def add_days(date, days):
    date2 = datetime.datetime.strptime(date, "%m/%d") + datetime.timedelta(days=days)
    return date2.strftime("%m/%d")

def scrape_data(country):
    if country == "United States":
        index_country = "US"
        index_pop = country
    elif country == "Iran":
        index_country = country
        index_pop = "Iran, Islamic Rep."
    else:
        index_country = country
        index_pop = country
    
    try:
        os.remove("data.csv")
        os.remove("world-population.csv")
    except OSError:
        pass

    download("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv", "data.csv", bar=lambda *_: "")
    download("https://data.opendatasoft.com/explore/dataset/world-population@kapsarc/download/?format=csv&timezone=America/New_York&use_labels_for_header=true&csv_separator=%2C", bar=lambda *_: "")

    with open("data.csv", newline='') as data_file:
        data_reader = csv.reader(data_file)
        country_rows = [list(map(int, row[4:])) for row in data_reader if row[1] == index_country]
    
    raw_data = list(map(sum, zip(*country_rows)))

    with open("world-population.csv") as pop_file:
        pop_reader = csv.reader(pop_file)
        pop_data = [list(map(float, [row[0], row[2]])) for row in pop_reader if row[1] == index_pop]
        population = max(pop_data, key=lambda i: i[0])[1]
    
    data = raw_data
    for i in range(len(raw_data)):
        if raw_data[i] != 0:
            break
        else:
            data = raw_data[i+1:]
    
    offset = len(raw_data) - len(data)
    
    return data, population, offset

if __name__ == "__main__":
    country = input("Which country? (United States): ")
    end_period = input("Model up to how many days after today? (30): ")
    country = country if country else "United States"
    end_period = int(end_period) if end_period else 30

    data, population, offset = scrape_data(country)
    model, beta, gamma, rsq = fit_sir(data, population, end_period)
    start_date = add_days("1/22", offset)
    print(start_date)

    while True:
        cmd = input(">> ")
        if cmd.startswith("predict"):
            date = cmd.split()[1]
            days = subtract_dates(date, start_date)
            print("Predicted number of cases on " + date + ": " + str(model.y[1][::50][days]))
        elif cmd.startswith("graph"):
            fig = plt.figure()
            fig.suptitle("SIR Model of COVID-19 for " + country + "\nβ = " + str(round(beta, 4)) + ", γ = " + str(round(gamma, 4)) + ", R^2 = " + str(round(rsq, 4)) + ", R = " + str(round(sqrt(rsq), 4)))
            plt.xlabel("Days since "+start_date)
            plt.ylabel("Number of cases")
            plt.plot(model.t, model.y[1], "r-", label="Model")
            plt.plot(np.arange(len(data)), data, "*:", label="Real-world Data")
            plt.legend(loc="upper left")
            fig.show()
        elif cmd.startswith("max"):
            day = add_days(start_date, np.argmax(model.y[1])/50)
            cases = np.max(model.y[1])
            print("Maximum is predicted to be on " + day + " with " + str(cases) + " cases")
