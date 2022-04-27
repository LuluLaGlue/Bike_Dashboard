import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

st.title("Dashboard - Graphics")

df = pd.read_csv(os.path.join("data", "Capital_Bikeshare_data.csv"), sep=";")

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

# fig, ax = plt.subplots(figsize=(20, 5))
# sns.pointplot(data=df, x='month', y='count', hue='weekday', ax=ax)
# ax.set(title='Count of bikes during weekdays and weekends')
# plt.show()

# st.write(df.head(10))

st.subheader("Bikes rented per month and week day")

fig_month, ax_month = plt.subplots(figsize=(20, 5))
sns.barplot(data=df, x='month', y='count', ax=ax_month)
ax_month.set(title='Count of bikes during different months')
# plt.show()

fig_weekday, ax_weekday = plt.subplots(figsize=(20, 5))
sns.barplot(data=df, x='weekday', y='count', ax=ax_weekday)
ax_weekday.set(title='Count of bikes during different days')
# plt.show()

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
    st.write(
        "Seasons equivalence: 1 - Spring, 2 - Summer, 3 - Fall, 4 - Winter")
    st.write(fig_season)
with w:
    st.write(
        "Weather equivalence: 1 - Clear Skies, 2 - Misty / Cloudy, 3 - Light Raining / Thunderstorm, 4 - Heavy Raining / Snow"
    )
    st.write(fig_weather)

st.subheader("Influence of temperature and humidity on users")

fig1, ax1 = plt.subplots(figsize=(20, 5))
sns.regplot(x=df['temp'], y=df['count'], ax=ax1, color="red")
ax1.set(title="Relation between temperature and users")
fig2, ax2 = plt.subplots(figsize=(20, 5))
sns.regplot(x=df['humidity'], y=df['count'], ax=ax2)
ax2.set(title="Relation between humidity and users")
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

df_ml = df.copy()

target_count = df_ml["count"]
target_registered = df_ml["registered"]
target_casual = df_ml["casual"]
df_ml = df_ml.drop(["casual", "registered", "count"], axis=1)

st.title("Machine Learning")

algo = [(LinearRegression(), "Linear Regression"),
        (Ridge(alpha=0.5), "Ridge Regression"),
        (Lasso(alpha=0.5), "Lasso Regression"),
        (ElasticNet(alpha=0.5), "Elastic Net")]

for regression, title in algo:
    st.subheader(title)
    co, re, ca = st.columns(3)
    with co:
        st.subheader("*Count*")
        X_train, X_test, y_train, y_test = train_test_split(df_ml,
                                                            target_count,
                                                            test_size=0.30,
                                                            random_state=40)
        scaler = StandardScaler()
        std_feats = scaler.fit_transform(X_train)
        model = regression.fit(std_feats, y_train)
        pred_test = model.predict(X_test)

        rmse_count = np.sqrt(mean_squared_error(y_test, pred_test))
        r2_count = r2_score(y_test, pred_test)

    with re:
        st.subheader("*Registered*")
        X_train, X_test, y_train, y_test = train_test_split(df_ml,
                                                            target_registered,
                                                            test_size=0.30,
                                                            random_state=40)
        scaler = StandardScaler()
        std_feats = scaler.fit_transform(X_train)
        model = regression.fit(std_feats, y_train)
        pred_test = model.predict(X_test)

        rmse_registered = np.sqrt(mean_squared_error(y_test, pred_test))
        r2_registered = r2_score(y_test, pred_test)

    with ca:
        st.subheader("*Casual*")
        X_train, X_test, y_train, y_test = train_test_split(df_ml,
                                                            target_casual,
                                                            test_size=0.30,
                                                            random_state=40)
        scaler = StandardScaler()
        std_feats = scaler.fit_transform(X_train)
        model = regression.fit(std_feats, y_train)
        pred_test = model.predict(X_test)

        rmse_casual = np.sqrt(mean_squared_error(y_test, pred_test))
        r2_casual = r2_score(y_test, pred_test)

    a, b, c, d, e, f = st.columns(6)

    with a:
        st.write("RMSE")
        st.write(np.round(rmse_count, 2))
    with b:
        st.write("R2")
        st.write(np.round(r2_count, 4))

    with c:
        st.write("RMSE")
        st.write(np.round(rmse_registered, 2))
    with d:
        st.write("R2")
        st.write(np.round(r2_registered, 4))

    with e:
        st.write("RMSE")
        st.write(np.round(rmse_casual, 2))
    with f:
        st.write("R2")
        st.write(np.round(r2_casual, 4))
