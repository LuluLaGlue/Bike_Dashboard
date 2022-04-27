import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNetCV, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

warnings.filterwarnings('ignore')
st.set_page_config(layout="wide")


def one_hot_encoding(data, column):
    data = pd.concat(
        [data,
         pd.get_dummies(data[column], prefix=column, drop_first=True)],
        axis=1)
    data = data.drop([column], axis=1)
    return data


@st.cache
def format_df_for_ml(df):
    df_ml = df

    df_ml['count'] = np.log(df_ml['count'])
    df_ml['casual'] = np.log(df_ml['casual'])
    df_ml['registered'] = np.log(df_ml['registered'])

    cols = ['season', 'month', 'holiday', 'weekday', 'workingday', 'weather']

    for col in cols:
        df_ml = one_hot_encoding(df_ml, col)

    target_count = df_ml["count"]
    target_registered = df_ml["registered"]
    target_casual = df_ml["casual"]
    df_ml = df_ml.drop(["atemp", "windspeed", "casual", "registered", "count"],
                       axis=1)

    return df_ml, target_count, target_casual, target_registered


def train_reg(X_train, X_test, y_train, y_test, color):
    scaler = StandardScaler()

    std_feats = scaler.fit_transform(X_train)
    model = regression.fit(std_feats, y_train)
    pred_test = model.predict(X_test)

    error = y_test - pred_test
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(y_test, error, color=color)
    ax.axhline(lw=3, color='black')
    ax.set_xlabel('Observed')
    ax.set_ylabel('Error')

    return fig, pred_test


if __name__ == "__main__":
    # st.snow()

    st.title("Dashboard - Graphics")

    df = pd.read_csv(os.path.join("data", "Capital_Bikeshare_data.csv"),
                     sep=";")

    df.apply(lambda x: len(x.unique()))
    df.isnull().sum()

    df = df.rename(
        columns={
            'weathersit': 'weather',
            'yr': 'year',
            'mnth': 'month',
            'hr': 'hour',
            'hum': 'humidity',
            'cnt': 'count'
        })

    df = df.drop(columns=['dteday', 'year'])

    cols = ['season', 'month', 'holiday', 'weekday', 'workingday', 'weather']
    for col in cols:
        df[col] = df[col].astype('category')

    st.subheader("Bikes rented per month and week day")

    fig_month, ax_month = plt.subplots(figsize=(20, 5))
    sns.barplot(data=df, x='month', y='count', ax=ax_month)
    ax_month.set(title='Count of bikes during different months')

    fig_weekday, ax_weekday = plt.subplots(figsize=(20, 5))
    sns.barplot(data=df, x='weekday', y='count', ax=ax_weekday)
    ax_weekday.set(title='Count of bikes during different days')

    m, w = st.columns(2)
    with m:
        st.write(fig_month)
    with w:
        st.write(fig_weekday)

    st.subheader("Bikes rented per season and weather")

    fig_season, ax_season = plt.subplots(figsize=(20, 5))
    sns.barplot(data=df, x='season', y='count', ax=ax_season)
    ax_season.set(title='Count of bikes during different seasons')

    fig_weather, ax_weather = plt.subplots(figsize=(20, 5))
    sns.barplot(data=df, x='weather', y='count', ax=ax_weather)
    ax_weather.set(title='Count of bikes for different weathers')

    s, w = st.columns(2)
    with s:
        st.write(fig_season)
        st.caption(
            "Seasons equivalence: 1 - Spring, 2 - Summer, 3 - Fall, 4 - Winter"
        )
    with w:
        st.write(fig_weather)
        st.caption(
            "Weather equivalence: 1 - Clear Skies, 2 - Misty / Cloudy, 3 - Light Raining / Thunderstorm, 4 - Heavy Raining / Snow"
        )

    st.subheader("Influence of temperature on users")

    fig1, ax1 = plt.subplots(figsize=(20, 5))
    sns.regplot(x=df['temp'], y=df['count'], ax=ax1, color="orange")
    ax1.set(title="Relation between temperature and users")
    fig2, ax2 = plt.subplots(figsize=(20, 5))
    sns.regplot(x=df['atemp'], y=df['count'], ax=ax2)
    ax2.set(title="Relation between felt temperature and users")
    t, h = st.columns(2)
    with t:
        st.write(fig1)
    with h:
        st.write(fig2)

    st.subheader("Influence of humidity and windspeed on users")

    fig1, ax1 = plt.subplots(figsize=(20, 5))
    sns.regplot(x=df['humidity'], y=df['count'], ax=ax1, color="purple")
    ax1.set(title="Relation between humidity and users")
    fig2, ax2 = plt.subplots(figsize=(20, 5))
    sns.regplot(x=df['windspeed'], y=df['count'], ax=ax2, color="green")
    ax2.set(title="Relation between windspeed and users")
    t, h = st.columns(2)
    with t:
        st.write(fig1)
    with h:
        st.write(fig2)

    st.subheader("Correlation Matrix")

    corr = df.corr()
    fig = plt.figure(figsize=(15, 7))
    sns.heatmap(corr, annot=True, annot_kws={'size': 10})
    c, _ = st.columns(2)

    with c:
        st.write(fig)

    df_ml, target_count, target_casual, target_registered = format_df_for_ml(
        df)

    st.title("Machine Learning")

    algo = [(LinearRegression(), "Linear Regression"),
            (Ridge(), "Ridge Regression"), (Lasso(), "Lasso Regression"),
            (ElasticNetCV(), "Elastic Net"),
            (HuberRegressor(), "Huber Regression"),
            (RandomForestRegressor(), "Random Forest Regression"),
            (GradientBoostingRegressor(), "Gradient Boosting Regression"),
            (ExtraTreesRegressor(), "Extra Trees Regression")]
    available_algo = [
        "Linear Regression", "Ridge Regression", "Lasso Regression",
        "Elastic Net", "Huber Regression", "Random Forest Regression",
        "Gradient Boosting Regression", "Extra Trees Regression"
    ]
    to_display = []

    alg_selection = st.multiselect("Select Algorithms", available_algo,
                                   available_algo)

    for a in alg_selection:
        index = available_algo.index(a)
        to_display.append(algo[index])

    for regression, title in to_display:
        st.header(title)
        co, re, ca = st.columns(3)
        with co:
            st.subheader("*Count*")
            X_train_count, X_test_count, y_train_count, y_test_count = train_test_split(
                df_ml, target_count, test_size=0.25, random_state=42)
            fig_count, pred_test_count = train_reg(X_train_count, X_test_count,
                                                   y_train_count, y_test_count,
                                                   "green")
            st.write(fig_count)

            rmse_count = np.sqrt(
                mean_squared_error(y_test_count, pred_test_count))
            r2_count = r2_score(y_test_count, pred_test_count)

        with re:
            st.subheader("*Registered*")
            X_train_registered, X_test_registered, y_train_registered, y_test_registered = train_test_split(
                df_ml, target_registered, test_size=0.25, random_state=42)
            fig_registered, pred_test_registered = train_reg(
                X_train_registered, X_test_registered, y_train_registered,
                y_test_registered, "blue")
            st.write(fig_registered)

            rmse_registered = np.sqrt(
                mean_squared_error(y_test_registered, pred_test_registered))
            r2_registered = r2_score(y_test_registered, pred_test_registered)

        with ca:
            st.subheader("*Casual*")
            X_train_casual, X_test_casual, y_train_casual, y_test_casual = train_test_split(
                df_ml, target_casual, test_size=0.25, random_state=42)
            fig_casual, pred_test_casual = train_reg(X_train_casual,
                                                     X_test_casual,
                                                     y_train_casual,
                                                     y_test_casual, "purple")
            st.write(fig_casual)

            rmse_casual = np.sqrt(
                mean_squared_error(y_test_casual, pred_test_casual))
            r2_casual = r2_score(y_test_casual, pred_test_casual)

        z, a, b, y, c, d, x, e, f = st.columns(9)

        with z:
            st.write()
        with a:
            st.write("RMSE")
            st.write(np.round(rmse_count, 4))
        with b:
            st.write("R2")
            st.write(np.round(r2_count, 4))

        with y:
            st.write()
        with c:
            st.write("RMSE")
            st.write(np.round(rmse_registered, 4))
        with d:
            st.write("R2")
            st.write(np.round(r2_registered, 4))

        with x:
            st.write()
        with e:
            st.write("RMSE")
            st.write(np.round(rmse_casual, 4))
        with f:
            st.write("R2")
            st.write(np.round(r2_casual, 4))
