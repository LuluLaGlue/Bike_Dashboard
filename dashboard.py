import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")

st.title("Dashboard")

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
