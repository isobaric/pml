#!/usr/bin/env python3
"""
Functions used to fit the models for evaluation.

"""
import pandas
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from scipy import constants

matplotlib.rcParams.update({'font.size': 24})

exampleseed = 3232043

def fit_and_test_models(randomseed):
    #load the data from previous script
    training = pandas.read_csv("./training_data.csv", parse_dates=True, index_col=0)
    test = pandas.read_csv("./test_data.csv", parse_dates=True,index_col=0)

    #rescale pressure to atm and remove 1 atm to scale/center it better
    training["Air pressure"] = (training["Air pressure"]*100.0/constants.atm)-1
    test["Air pressure"] = (test["Air pressure"]*100.0/constants.atm)-1
    
    #separate predictors and target variable
    training_x = training.drop("T", axis=1).values
    training_y = training["T"].values

    test_x = test.drop("T", axis=1).values
    test_y = test["T"].values 

    models = {"Linear":LinearRegression(), "Ridge":Ridge(alpha=0.1), "Lasso":Lasso(alpha=0.1),
              "Neural":MLPRegressor(hidden_layer_sizes=10, alpha=0.1, early_stopping=True, max_iter=3000, random_state=randomseed),
              "QuadraticLasso":make_pipeline(PolynomialFeatures(2), Lasso(alpha=0.1, max_iter=5000))}

    prediction_error = pandas.DataFrame(test["T"])
    for label,model in models.items():
        model.fit(training_x, training_y)
        prediction_error[label] = np.abs(test_y - model.predict(test_x))
        
 
    scores = {label:[model.score(test_x, test_y), np.mean(prediction_error[label]), np.max(prediction_error[label])] for label,model in models.items()}
    level_difference = np.abs(test["T lower"] - test["T"])
    scores["substitution"] = [r2_score(test["T"], test["T lower"]), np.mean(level_difference), np.max(level_difference)]

    #Testing our hypothesis = " Model better than linear interpolation for large gaps"
    #Let's do ten repeats of each gap percentage to get some statistics

    error_chances = np.linspace(0.1,1, num=20)
    repeats = 10

    comparison_to_interpolation = {i:against_interpolation(models, test, e, prediction_error, repeats) for i,e in enumerate(error_chances)}

    return models, prediction_error, scores, comparison_to_interpolation, error_chances


def against_interpolation(models, data, error_chance, prediction_errors, repeats):

    #can't remove endpoints, or it is not technically interpolation anymore
    datapoints = data.shape[0]-2
    indices = np.full(datapoints+2, False, dtype=bool)
    
    stats = {"mean":np.mean, "max":np.max, "std":np.std}
    
    errors = {model:{name:np.zeros(repeats) for name in stats} for model in models}
    errors["Interpolation"] = {name:np.zeros(repeats) for name in stats}
    
    for i in range(repeats):
        
        #choose datapoints to keep with chance 1-error_chance        
        indices[1:-1] = np.random.choice([True, False], size=datapoints, replace=True, p = [error_chance, 1-error_chance])

        fragmented_data = data.copy()
        fragmented_data[indices] = np.nan
        #also slinear should work correctly, not necessarily evenly spaced data
        fragmented_data.interpolate("time", inplace=True)

        #calculate l2-distance at indices, and reducing functions
        interpolation_errors = np.abs(fragmented_data["T"][indices]-data["T"][indices])
        for m in errors:
            if m == "Interpolation":
                for name,method in stats.items():
                    errors[m][name][i] = method(interpolation_errors)
            else:
                for name,method in stats.items():
                    errors[m][name][i] = method(prediction_errors[m][indices])



    #and since we don't really care about the individual runs, reduce again over the repeats
    #In retrospect, there might be a simpler datastructure and a way to do this with pandas
    reduced_error_stats = {model:{s:{method_name:method(values) for method_name,method in stats.items()} for s,values in d.items()} for model,d in errors.items()}

                    
    return reduced_error_stats


def make_easily_plottable_form(error_stats,error_rates):
    """
    I have to plot the errors from the resulting error data structure.
    So actually I want them to be simple arrays, which requires some reorganisation.
    ...
    If this was something I'd use constatly I would rewrite the error calculation to produce a nicer format directly,
    but since this was to be a one time script, this utility function works as well. It is not very elegant though. 
    """

    model_stats = {model:{name:np.zeros_like(error_rates) for name in ["max", "mean", "mean std", "max std"]} for model in error_stats[0]}
    
    for i,e in enumerate(error_rates):
        for model,results in error_stats[i].items():
            model_stats[model]["max"][i] = results["max"]["mean"]
            model_stats[model]["max std"][i] = results["max"]["std"]
            model_stats[model]["mean"][i] = results["mean"]["mean"]
            model_stats[model]["mean std"][i] = results["mean"]["std"]


    return model_stats


def plot_interpolation_comparison(error_stats, error_rates):
    model_stats = make_easily_plottable_form(error_stats, error_rates)
    
    #Since most of the linear models have extremely similar behaviour
    #I will only plot lasso, Quadratic Lasso and the Neural network in the comparison.

    included_models = ["Interpolation", "Lasso", "QuadraticLasso", "Neural"]

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    for model in included_models:
        ax1.errorbar(error_rates, model_stats[model]["mean"],
                     yerr=model_stats[model]["mean std"],
                     marker='x', linestyle='--', capsize=6, label=model)
        ax1.set_title("Mean")
        ax1.set_ylabel("K")
        ax1.set_xlabel("Fraction missing")
        
        ax2.errorbar(error_rates, model_stats[model]["max"],
                     yerr=model_stats[model]["max std"],
                     marker='x', linestyle='--', capsize=6, label=model)
        ax2.set_title("Max")
        ax2.set_ylabel("K")
        ax2.set_xlabel("Fraction missing")
        
        
    ax1.set_yscale("log")
    ax1.legend(loc="best")
    ax2.set_yscale("log")
    ax2.legend(loc="best")

    plt.show()
    
