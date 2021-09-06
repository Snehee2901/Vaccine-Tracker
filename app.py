import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as pxi
import datetime
from datetime import date,timedelta
import tabula as tb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly 
import pystan
import fbprophet
import time
from fbprophet import Prophet
from urllib.error import HTTPError
from fbprophet.plot import plot_plotly, plot_components_plotly
from threading import Timer
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent,px
from htbuilder.funcs import rgba, rgb
import requests
from bs4 import BeautifulSoup
import re
import seaborn as sns

#from st_rerun_patch import rerun
st.set_page_config(page_title='Vaccination Tracker',page_icon="https://www.heart.org/-/media/images/news/2020/july-2020/0717coronavirusvaccine_sc.jpg",layout="wide",
        initial_sidebar_state="auto")
st.title("VACCINATION TRACKER")
st.write("It helps to track the **Vaccine data** , **COVID data**, **Recovery data** and **Death data**.")
st.sidebar.title("CHOOSE YOUR OPTION")


def getdata(): 
    df = pd.read_csv("data//vaccine_data.csv")
    return df
# Get the data from govt. site
    #try:
        #r = re.compile("http://mohfw.gov.in/pdf/")
        #urlgov = "https://www.mohfw.gov.in/"
        #req = requests.get(urlgov)
       # soup = BeautifulSoup(req.content, 'html.parser')
        #links = []
        #for img in soup.select('a[href] img'):
         #   link = img.find_parent('a', href=True)
         #   links.append(link['href'])
        #url = list(filter(r.match,links))
        #url = url[0]
        #table = tb.read_pdf(url,pages=1)
        #df = table[1]
    #except HTTPError as err:
        #if err.code == 404:
            #df = pd.DataFrame() """
    


def preprocess(df): # Preproess the data and make changes
    if df.empty:
        return df
    else:
        #print(df)
        #df.drop("Unnamed: 0",axis=1,inplace=True)
        #df = df.rename(columns={'Beneficiaries vaccinated':'First Dose Administered',
                              #  'Unnamed: 0': 'Second Dose Administered','Unnamed: 1':"Total Doses Administered",
                              # 'State/UT': 'State'})###Change unnamed 2 --> 1
        #print(df)
        #df = df.drop([0])
        #df.set_index(['S. No.'],inplace=True)
        #df['State'] = df['State'].replace(['A & N Islands','Chhattisgarh*','Punjab*','Jammu & Kashmir'],['Andaman and Nicobar Islands','Chhattisgarh',
        #                                                                           'Punjab','Jammu and Kashmir'])
        #for i in range(1,len(df)+1):
        #    df["First Dose Administered"][i] = df["First Dose Administered"][i].replace(',',"")
        #    df["Second Dose Administered"][i] = df["Second Dose Administered"][i].replace(',',"")
        #    df["Total Doses Administered"][i] = df["Total Doses Administered"][i].replace(',',"")
        
        #df["First Dose Administered"] = df['First Dose Administered'].astype(int)
       # df["Second Dose Administered"] = df['Second Dose Administered'].astype(int)
        #df["Total Doses Administered"] = df['Total Doses Administered'].astype(int) 
        return df


@st.cache(show_spinner=False)
def call(): # Call functions and make the changes 
    df = pd.read_csv("data//vaccine_data.csv")
    df['Updated On'] = pd.to_datetime(df['Updated On']).dt.date
    df.to_csv("data//vaccine_data_till_yesterday.csv",index=False)
    df['Updated On'] = pd.to_datetime(df['Updated On']).dt.date
    df1 = getdata()
    df1 = preprocess(df1)
    if df1.empty:
        return df
    else:
        #df1.loc[len(df1)+1] = ['India',df1['First Dose Administered'].sum(),df1['Second Dose Administered'].sum(),
         #                               df1['Total Doses Administered'].sum()]
        #a = df1.loc[(df1['State'] == 'Dadra & Nagar Haveli') | (df1['State'] == 'Daman & Diu')]
        #df1.loc[len(df1)+1] = ['Dadra and Nagar Haveli and Daman and Diu',a["First Dose Administered"].sum(),
        #               a['Second Dose Administered'].sum(),a['Total Doses Administered'].sum()]
        #df1.drop([8,9],inplace=True)
        #df1['Updated On'] = date.today() - timedelta(days=1)
        #df3 = df.append(df1)
        #df3.drop_duplicates(inplace=True)
        #df3.to_csv("data//vaccine_data_till_yesterday.csv",index=False)
        #df3.to_csv("data//vaccine_data.csv",index=False)
        return df#df3

def state_data(): # Get everyday data for COVID cases
    df_st = pd.read_csv("data/state_data.csv")
    df_st['Date'] = pd.to_datetime(df_st['Date']).dt.date
    #data = pd.read_csv("https://raw.githubusercontent.com/shivanshsinghal107/COVID-19-India-Dataset/master/covid_india.csv")
    #data.rename(columns={"Name of State / UT":"State","Active Cases":"Active","Cured/Discharged/Migrated":"Recovered",
    #                 "Total Confirmed cases":"Confirmed"},inplace=True)
    #data.drop("S. No.",axis=1,inplace=True)
    #data.drop(["Active"],axis=1,inplace=True)
    #data.loc[len(data)+1] = ['India',data['Confirmed'].sum(),data['Recovered'].sum(),data['Deaths'].sum()]
    #data['State'] = data['State'].replace(['Andaman & Nicobar'],['Andaman and Nicobar Islands'])   
    #data["Date"] = date.today()
    #new_st = df_st.append(data)
    #new_st.drop_duplicates(inplace=True)
    #new_st.to_csv("data/state_data.csv",index=False) """
    return df_st #new_st

vac_det = pd.read_csv("data/new_data.csv")

df = call()
radio = st.sidebar.radio("Choose graph to view for COVID 19",("Vaccination","Confirmed cases","Recovered cases","Death cases"))
if radio == "Vaccination":
    da = datetime.date(2021,1,16)
else:
    da = datetime.date(2020,1,30)
start_date = st.sidebar.date_input("Start Date for analysing data ",value=datetime.date(2021,1,16),min_value=da,max_value=date.today())
end_date = st.sidebar.date_input("End Date for analysing data ",value=date.today(),min_value=start_date,
                                max_value=date.today())
state_select = st.sidebar.selectbox("Select State",vac_det['State'].unique())
#radio = st.sidebar.radio("Choose graph to view for COVID 19",("Vaccination","Confirmed cases","Recovered cases","Death cases"))
n_df = pd.read_csv('data/vaccine_data.csv').groupby(["Updated On","State"]).sum().sort_values(by="Total Doses Administered").reset_index()
sstate = n_df[n_df['State'] == state_select]
vac_state = sstate
sstate['Updated On'] = pd.to_datetime(sstate['Updated On']).dt.date
sstate = sstate.loc[(sstate['Updated On'] >= start_date)
                     & (sstate['Updated On'] <= end_date)]
sstate.sort_values(by="Updated On",inplace=True)

vac_det = vac_det[vac_det['State'] == state_select]
vac_det['Updated On'] = pd.to_datetime(vac_det['Updated On']).dt.date
vac_det.drop(vac_det[vac_det["Updated On"] >= date(2021,6,1)].index,inplace=True)
vac_det = vac_det.loc[(vac_det['Updated On'] >= start_date)
                     & (vac_det['Updated On'] <= end_date)]
vac_det.sort_values(by="Updated On",inplace=True)

population = pd.read_csv("data/statewise_population.csv")
state_pop = population[population['State'] == state_select]

def get_total_dataframe(dataset):
    total_dataframe = pd.DataFrame({
    'Status':['Confirmed', 'Recovered', 'Deaths'],
    'Number of cases':(dataset['Confirmed'].sum(),
    dataset['Recovered'].sum(), 
    dataset['Deaths'].sum())})
    return total_dataframe

def get_data(df):
    data = pd.DataFrame({'Status':['First Dose','Second Dose','Total Doses'],
                        'Number of doses':(df['First Dose Administered'].sum(),df['Second Dose Administered'].sum(),
                                            df['Total Doses Administered'].sum())})
    return data

# VACCINATION 
if radio == "Vaccination":
    st.markdown("""This is a web based interactive dashboard that changes according to the data.
                The data has been collected from https://www.kaggle.com/sudalairajkumar/covid19-in-india and the data 
                changes according to the https://api.covid19india.org/ and https://www.mohfw.gov.in/. 
                User can choose prefrences from the side bar. """,unsafe_allow_html=True)
    st.image("https://scitechdaily.com/images/COVID-19-Vaccine-Vials.gif",use_column_width=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sstate["Updated On"], y=sstate['First Dose Administered'], name='First Dose',
                         line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=sstate["Updated On"], y=sstate['Second Dose Administered'], name='Second Dose',
                         line=dict(color='blue', width=4)))
    fig.add_trace(go.Scatter(x=sstate["Updated On"], y=sstate['Total Doses Administered'], name='Total Doses',
                         line=dict(color='green', width=4)))
    st.markdown("## **Vaccination in %s**" %(state_select))
    fig.update_layout(xaxis_title='Date', yaxis_title='Vaccine Doses',width=850,height=600)
    st.plotly_chart(fig,use_container_width=True,use_column_width=True)
    st.dataframe(sstate)
    state = get_data(sstate)
    # ANALYSIS BY STATE
    if st.sidebar.checkbox("Vaccine Analysis by State", True, key=3):
        st.markdown("## **State level analysis of Vaccine**")
        st.markdown("### Overall First Dose, Second Dose and " +
        "Total Doses in %s yet" % (state_select))
        if not st.checkbox('Hide Graphs', False, key=1):
            state_total_graph = pxi.bar(
            state, 
            x='Status',
            y='Number of doses',
            labels={'Number of doses':'Number of cases in %s' % (state_select)},
            color="Status",color_discrete_map = {"First Dose":"#5412E9","Second Dose":"#6B589A","Total Doses":"#220F62"})
            st.plotly_chart(state_total_graph,use_container_width=True,use_column_width=True)

            if state_select == 'India':
                x = df.groupby("State").sum().reset_index()
                x = x[x['State'] != 'India']
                x.sort_values(by="Total Doses Administered",ascending=False,inplace=True)
                fig = pxi.bar(x.head(10), x="State", y=["First Dose Administered", "Second Dose Administered", 
                                           "Total Doses Administered"],labels={'value':'No. of Vaccine Doses',
                                           "variable":"Dose"},barmode='stack')
                st.markdown("### Comparing top 10 states according to total doses of vaccines") 
                st.plotly_chart(fig,use_container_width=True,use_column_width=True)
            # PIE CHARTS
            #if state_select == 'Miscellaneous':
                #st.image('no-data.png',use_column_width=True)
        
            st.markdown("### Number of Covaxin and CoviShield administered in %s till 31stMay"%(state_select))
            state_vac_gr= go.Figure()
            state_vac_gr.add_trace(go.Pie(labels=["Covaxin","Covishield"], values=[vac_det['Total Covaxin Administered'].sum(),
                                                vac_det['Total CoviShield Administered'].sum()], name="Vaccine Classification"))
            state_vac_gr.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                        marker=dict(colors=['#20D0B1','#D964EE','blue','green','pink'], line=dict(color='#000000', width=2)))
            st.plotly_chart(state_vac_gr,use_container_width=True)

            st.markdown("### Vaccination in %s GENDER wise till 31stMay"%(state_select))
            state_vac_ge = go.Figure()
            state_vac_ge.add_trace(go.Pie(labels=["Male","Female","Transgender"], values=[vac_det['Male(Individuals Vaccinated)'].sum(),
                                        vac_det['Female(Individuals Vaccinated)'].sum(),vac_det['Transgender(Individuals Vaccinated)'].sum()],
                                        name="Gender Based Classification"))
            state_vac_ge.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                        marker=dict(colors=['gold', 'mediumturquoise', 'darkorange', 'lightgreen'], line=dict(color='#000000', width=2)))
            st.plotly_chart(state_vac_ge,use_container_width=True,use_column_width=True)

            vac_rate = sstate
            popstate = state_pop['Population']
            vac_rate['First'] = round((vac_rate['First Dose Administered'] / float(popstate))*100,2)
            vac_rate['Both'] = round((vac_rate['Second Dose Administered'] / float(popstate))*100,2)
            rate = go.Figure()
            rate.add_trace(go.Scatter(x=vac_rate["Updated On"], y=vac_rate['Both'], name='Fully',fill='tonexty',
                            stackgroup='one',hovertemplate=
                                "Fully vaccinated: %{y}%<br>" +
                                "<extra></extra>",
                            line=dict(color='blue', width=4)))
            rate.add_trace(go.Scatter(x=vac_rate["Updated On"], y=vac_rate['First'], name='First',fill='tozeroy',
                            stackgroup='one',hovertemplate=
                                "Atleast 1 dose: %{y}%<br>" +
                                "<extra></extra>",
                            line=dict(color='firebrick', width=4)))
            st.markdown("### Vaccination rate in %s"%(state_select))
            rate.update_layout(xaxis_title='Month',yaxis_title='Vaccination rate (%)',width=850,height=600,
                                        hovermode="x unified")
            st.plotly_chart(rate,use_container_width=True,use_column_width=True)


    # PREDICTION
    if st.sidebar.checkbox("Vaccination Prediction", False, key=2):
        x = st.sidebar.date_input('End date when you want to end the prediction',value=date.today()+timedelta(days=1),
                        max_value=date.today()+timedelta(days=1460),min_value=date.today()+timedelta(days=1))
        st.markdown("## **State level prediction of Vaccine**")
        radsel = st.radio("Choose for which dose you want to predict",['First Dose','Second Dose','Total Doses'])
        if not st.checkbox('Hide Graphs', False, key=3): 
            if radsel == 'First Dose':
                vac_state['Updated On'] = pd.to_datetime(vac_state['Updated On']).dt.date
                new = vac_state.drop(columns={"Second Dose Administered","Total Doses Administered","State"},axis=1)
                new.rename(columns={'Updated On':'ds','First Dose Administered':'y'},inplace=True)
                new['ds'] = pd.to_datetime(new['ds']).dt.date
                m = Prophet(growth='linear',weekly_seasonality=False,yearly_seasonality=False,
                            changepoint_prior_scale=0.5,changepoint_range=0.9)
                m.fit(new)
                future = m.make_future_dataframe(periods = 1460)
                forecast = m.predict(future)
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
                s = str(x)
                yy,mm,dd = s.split("-")
                yy,mm,dd = int(yy),int(mm),int(dd)
                ans = forecast[forecast['ds'] == date(yy,mm,dd)]
                s = str(dd)+"/"+str(mm)+"/"+str(yy)
                an = forecast[forecast['ds'] <= date(yy,mm,dd)]
                trace1 = go.Scatter(name = 'Predicted',mode = 'lines',x = list(an['ds']),y = list(an['yhat']),
                            marker=dict(color='red',line=dict(width=3)))
                trace2 = go.Scatter(name = 'Actual', mode = 'lines',x = list(an['ds']), y = list(new['y']),
                                    marker=dict(color='blue',line=dict(width=4)))
                data = [trace1,trace2]
                st.markdown("### Vaccine Prediction for first dose in %s "%(state_select))
                layout = dict(xaxis=dict(title = 'Dates', ticklen=2, zeroline=True),
                                        yaxis=dict(title = 'Vaccine doses', ticklen=2, zeroline=True))
                f_graph=dict(data=data,layout=layout)
                pop = int(state_pop['Population'])
                datecomp = forecast[forecast['yhat'] >= pop]
                st.markdown("#### At current rate " + str(pop) + " people of " + state_select + " will be vaccinated by " +
                            str(datecomp['ds'].iloc[0].strftime("%d-%m-%Y")) + " with first dose.")
                st.plotly_chart(f_graph,use_container_width=True,use_column_width=True)
                st.markdown("The predicted number of first doses on " + str(s) + " in "+ state_select + "  would be "+str(int(ans['yhat'])))

            if radsel == 'Second Dose':
                vac_state['Updated On'] = pd.to_datetime(vac_state['Updated On']).dt.date 
                new = vac_state.drop(columns={"First Dose Administered","Total Doses Administered","State"},axis=1)
                new.rename(columns={'Updated On':'ds','Second Dose Administered':'y'},inplace=True)
                new['ds'] = pd.to_datetime(new['ds']).dt.date
                m = Prophet(growth='linear',weekly_seasonality=False,yearly_seasonality=False,
                            changepoint_prior_scale=0.5,changepoint_range=0.9)
                m.fit(new)
                future = m.make_future_dataframe(periods = 4460)
                forecast = m.predict(future)
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
                s = str(x)
                yy,mm,dd = s.split("-")
                yy,mm,dd = int(yy),int(mm),int(dd)
                ans = forecast[forecast['ds'] == date(yy,mm,dd)]
                s = str(dd)+"/"+str(mm)+"/"+str(yy)
                an = forecast[forecast['ds'] <= date(yy,mm,dd)]
                trace1 = go.Scatter(name = 'Predicted',mode = 'lines',x = list(an['ds']),y = list(an['yhat']),
                            marker=dict(color='red',line=dict(width=3)))
                trace2 = go.Scatter(name = 'Actual', mode = 'lines',x = list(an['ds']), y = list(new['y']),
                                    marker=dict(color='blue',line=dict(width=4)))
                data = [trace1,trace2]
                st.markdown("### Vaccine Prediction for second dose in %s "%(state_select))
                layout = dict(xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))
                f_graph=dict(data=data,layout=layout)
                pop = int(state_pop['Population'])
                if state_select == "India":
                    pop = pop * 0.70
                datecomp = forecast[forecast['yhat'] >= pop]
                st.markdown("#### At current rate " + str(int(pop)) + " people of " + state_select + " will be vaccinated by " +
                            str(datecomp['ds'].iloc[0].strftime("%d-%m-%Y")) + " with second dose.")
                st.plotly_chart(f_graph,use_container_width=True,use_column_width=True)
                st.markdown("The predicted number of second doses on " + str(s) + " in "+ state_select + " would be "+str(int(ans['yhat'])))
            
            if radsel == 'Total Doses':
                vac_state['Updated On'] = pd.to_datetime(vac_state['Updated On']).dt.date 
                new = vac_state.drop(columns={"First Dose Administered","Second Dose Administered","State"},axis=1)
                new.rename(columns={'Updated On':'ds','Total Doses Administered':'y'},inplace=True)
                m = Prophet(growth='linear',weekly_seasonality=False,yearly_seasonality=False,
                            changepoint_prior_scale=0.5,changepoint_range=0.9)
                m.fit(new)
                b = x - date.today()
                future = m.make_future_dataframe(periods = int(b.days)+1)
                forecast = m.predict(future)
                forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
                s = str(x)
                yy,mm,dd = s.split("-")
                yy,mm,dd = int(yy),int(mm),int(dd)
                ans = forecast[forecast['ds'] == date(yy,mm,dd)]
                s = str(dd)+"/"+str(mm)+"/"+str(yy)
                st.markdown("The predicted number of total doses on " + str(s) + "in "+ state_select+" would be "+str(int(ans['yhat'])))
                an = forecast[forecast['ds'] <= date(yy,mm,dd)]
                trace1 = go.Scatter(name = 'Predicted',mode = 'lines',x = list(an['ds']),y = list(an['yhat']),
                            marker=dict(color='red',line=dict(width=3)))
                trace2 = go.Scatter(name = 'Actual', mode = 'lines',x = list(an['ds']), y = list(new['y']),
                                    marker=dict(color='blue',line=dict(width=4)))
                data = [trace1,trace2]
                st.markdown("### Total dose prediction in %s"%(state_select))
                layout = dict(xaxis=dict(title = 'Dates', ticklen=2, zeroline=True))
                f_graph=dict(data=data,layout=layout)
            
                st.plotly_chart(f_graph,use_container_width=True,use_column_width=True)
        
        if st.checkbox('View Weekly and Yearly trends', False, key=3):
            an['ds'] = pd.to_datetime(an['ds'])
            f = plot_plotly(m, an,xlabel="Date",ylabel="Vaccination")
            st.plotly_chart(f,use_container_width=True,use_column_width=True)


    
# CONFIRMED CASES
if radio == "Confirmed cases":
    st.markdown("""This is a web based interactive dashboard that changes according to the current data.
                The data has been collected from https://www.kaggle.com/sudalairajkumar/covid19-in-india and everyday data 
                changes according to the https://raw.githubusercontent.com/shivanshsinghal107/COVID-19-India-Dataset/master/covid_india.csv. 
                User can choose prefrences from the side bar. """,unsafe_allow_html=True)
    st.image("https://i.pinimg.com/originals/c6/28/87/c62887db7cea40ab5753171c86e456ef.gif",use_column_width=True)
    #vac_pred = False
 
    df_sta = state_data()
    df_sta = df_sta[df_sta['State'] == state_select]
    df_sta = df_sta.loc[(df_sta['Date'] >= start_date)
                        & (df_sta['Date'] <= end_date)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sta["Date"], y=df_sta['Confirmed'], name='Confirmed Cases',
                            line=dict(color='blue', width=4)))
    st.markdown("## **Confirmed cases in %s** " %( state_select))
    fig.update_layout(xaxis_title='Month',
                    yaxis_title='Confirmed Cases',width=850,height=600)

        
    st.plotly_chart(fig,use_container_width=True,use_column_width=True)

    st.dataframe(df_sta)
    state_total = get_total_dataframe(df_sta)
    if st.sidebar.checkbox("Show Analysis by State", True, key=2) :
        st.markdown("## **State level analysis of COVID Cases**")
        st.markdown("### Overall Confirmed, Recovered and " +
            "Deceased cases in %s yet" % (state_select))
        if not st.checkbox('Hide Graph', False, key=1):
            state_total_graph = pxi.bar(
            state_total, 
            x='Status',
            y='Number of cases',
            labels={'Number of cases':'Number of cases in %s' % (state_select)},
            color='Status',color_discrete_map = {"Confirmed":"#5412E9","Recovered":"green","Deaths":"red"})
            st.plotly_chart(state_total_graph,use_container_width=True,use_column_width=True)
            
            if state_select == 'India':
                state = state_data()
                x = state.groupby("State").sum().reset_index()
                x = x[x['State'] != 'India']
                x.sort_values(by="Confirmed",ascending=False,inplace=True)
                fig = pxi.bar(x.head(10), x="State", y="Confirmed",hover_data={"Recovered","Deaths"},color="State",
                                labels={'value':'No. of Cases',"variable":"Case Type"})
                st.markdown("### Comparing confirmed cases in top 10 states") 
                st.plotly_chart(fig,use_container_width=True,use_column_width=True)

            # MODEL:
    if st.sidebar.checkbox("Confirmed Cases Prediction", False, key=4):
        x = st.sidebar.date_input('End date when you want to end the prediction',value=date.today()+timedelta(days=1),
                            max_value=date.today()+timedelta(days=1460),min_value=date.today()+timedelta(days=1))
        st.markdown("## **State level prediction of Confirmed cases**")
        if not st.checkbox('Hide Graphs', False, key=2):
            df_sta['Date'] = pd.to_datetime(df_sta['Date']).dt.date
            new = df_sta.drop(columns={"Recovered","Deaths","State"},axis=1)
            new.rename(columns={'Date':'ds','Confirmed':'y'},inplace=True)
            new['ds'] = pd.to_datetime(new['ds']).dt.date
            m = Prophet(growth='linear',weekly_seasonality=False,yearly_seasonality=False,
                                changepoint_prior_scale=0.9,changepoint_range=0.9,interval_width=0.95)
            m.fit(new)
            future = m.make_future_dataframe(periods = 1460)
            forecast = m.predict(future)
            forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
            s = str(x)
            yy,mm,dd = s.split("-")
            yy,mm,dd = int(yy),int(mm),int(dd)
            ans = forecast[forecast['ds'] == date(yy,mm,dd)]
            s = str(dd)+"/"+str(mm)+"/"+str(yy)
            an = forecast[forecast['ds'] <= date(yy,mm,dd)]
            trace1 = go.Scatter(name = 'Predicted',mode = 'lines',x = list(an['ds']),y = list(an['yhat']),
                                    marker=dict(color='red',line=dict(width=3)))
            trace2 = go.Scatter(name = 'Actual', mode = 'lines',x = list(an['ds']), y = list(new['y']),
                                            marker=dict(color='blue',line=dict(width=4)))
            data = [trace1,trace2]
            st.markdown("### Confirmed cases prediction in %s"%(state_select))
            layout = dict(xaxis=dict(title = 'Dates', ticklen=2, zeroline=True),
                                yaxis=dict(title = 'Confirmed Cases', ticklen=2, zeroline=True))
            f_graph=dict(data=data,layout=layout)
            st.plotly_chart(f_graph,use_container_width=True,use_column_width=True)
            st.markdown("The predicted number of confirmed cases on " + str(s) + " would be "+str(int(ans['yhat'])))

            
        if st.checkbox('View Weekly and Yearly trends', False, key=3):
            an['ds'] = pd.to_datetime(an['ds'])
            f = plot_plotly(m, an,xlabel="Date",ylabel="Cases")
            st.plotly_chart(f,use_container_width=True,use_column_width=True)

   

if radio == "Recovered cases":
    st.markdown("""This is a web based interactive dashboard that changes according to the current data.
                The data has been collected from https://www.kaggle.com/sudalairajkumar/covid19-in-india and everyday data 
                changes according to the https://raw.githubusercontent.com/shivanshsinghal107/COVID-19-India-Dataset/master/covid_india.csv. 
                User can choose prefrences from the side bar. """,unsafe_allow_html=True)
    st.image("https://i.pinimg.com/originals/c6/28/87/c62887db7cea40ab5753171c86e456ef.gif",use_column_width=True)
    #vac_pred = False
    df_sta = state_data()
    df_sta = df_sta[df_sta['State'] == state_select]
    df_sta = df_sta.loc[(df_sta['Date'] >= start_date)
                        & (df_sta['Date'] <= end_date)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sta["Date"], y=df_sta['Recovered'], name='Recovered Cases',
                            line=dict(color='green', width=4)))
    st.markdown("## **Recovered cases in %s** " %( state_select))
    fig.update_layout(xaxis_title='Month',
                    yaxis_title='Recovered Cases',width=850,height=600)
    st.plotly_chart(fig,use_container_width=True,use_column_width=True)
    st.dataframe(df_sta)
    state_total = get_total_dataframe(df_sta)
    if st.sidebar.checkbox("Show Analysis by State", True, key=2) :
        st.markdown("## **State level analysis of COVID Cases**")
        st.markdown("### Overall Confirmed, Recovered and " +
            "Deceased cases in %s yet" % (state_select))
        if not st.checkbox('Hide Graph', False, key=1):
            state_total_graph = pxi.bar(
            state_total, 
            x='Status',
            y='Number of cases',
            labels={'Number of cases':'Number of cases in %s' % (state_select)},
            color='Status',color_discrete_map = {"Confirmed":"#5412E9","Recovered":"green","Deaths":"red"})
            st.plotly_chart(state_total_graph,use_container_width=True,use_column_width=True)
            
        if state_select == 'India':
            state = state_data()
            x = state.groupby("State").sum().reset_index()
            x = x[x['State'] != 'India']
            x.sort_values(by="Recovered",ascending=False,inplace=True)
            fig = pxi.bar(x.head(10), x="State", y="Recovered",hover_data={'Confirmed',"Deaths"},color="State",
                            labels={'value':'No. of Cases',"variable":"Case Type"})
            st.markdown("### Comparing recovered cases in top 10 states") 
            st.plotly_chart(fig,use_container_width=True,use_column_width=True)
            
    if st.sidebar.checkbox("Recovered cases Prediction", False, key=4):
        x = st.sidebar.date_input('End date when you want to end the prediction',value=date.today()+timedelta(days=1),
                            max_value=date.today()+timedelta(days=1460),min_value=date.today()+timedelta(days=1))
        st.markdown("## **State level prediction of Recovered cases**")
        if not st.checkbox('Hide Graphs', False, key=2):
            df_sta['Date'] = pd.to_datetime(df_sta['Date']).dt.date
            new = df_sta.drop(columns={"Confirmed","Deaths","Confirmed","State"},axis=1)
            new.rename(columns={'Date':'ds','Recovered':'y'},inplace=True)
            new['ds'] = pd.to_datetime(new['ds']).dt.date
            m = Prophet(growth='linear',weekly_seasonality=False,yearly_seasonality=False,
                                changepoint_prior_scale=0.9,changepoint_range=0.9,interval_width=0.95)
            m.fit(new)
            future = m.make_future_dataframe(periods = 1460)
            forecast = m.predict(future)
            forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
            s = str(x)
            yy,mm,dd = s.split("-")
            yy,mm,dd = int(yy),int(mm),int(dd)
            ans = forecast[forecast['ds'] == date(yy,mm,dd)]
            s = str(dd)+"/"+str(mm)+"/"+str(yy)
            an = forecast[forecast['ds'] <= date(yy,mm,dd)]
            trace1 = go.Scatter(name = 'Predicted',mode = 'lines',x = list(an['ds']),y = list(an['yhat']),
                                    marker=dict(color='green',line=dict(width=3)))
            trace2 = go.Scatter(name = 'Actual', mode = 'lines',x = list(an['ds']), y = list(new['y']),
                                            marker=dict(color='blue',line=dict(width=4)))
            data = [trace1,trace2]
            st.markdown("### Recovered cases prediction in %s" %(state_select))
            layout = dict(xaxis=dict(title = 'Dates', ticklen=2, zeroline=True),
                                yaxis=dict(title = 'Recovered Cases', ticklen=2, zeroline=True))
            f_graph=dict(data=data,layout=layout)
            st.plotly_chart(f_graph,use_container_width=True,use_column_width=True)
            st.markdown("The predicted number of recovered cases on " + str(s) + " in "+ state_select + " would be "+str(int(ans['yhat'])))
                
        if st.checkbox('View Weekly and Yearly trends', False, key=3):
            an['ds'] = pd.to_datetime(an['ds'])
            f = plot_plotly(m, an,xlabel="Date",ylabel="Cases")
            st.plotly_chart(f,use_container_width=True,use_column_width=True)

if radio == "Death cases":
    st.markdown("""This is a web based interactive dashboard that changes according to the current data.
                The data has been collected from https://www.kaggle.com/sudalairajkumar/covid19-in-india and everyday data 
                changes according to the https://raw.githubusercontent.com/shivanshsinghal107/COVID-19-India-Dataset/master/covid_india.csv. 
                User can choose prefrences from the side bar. """,unsafe_allow_html=True)
    st.image("https://i.pinimg.com/originals/c6/28/87/c62887db7cea40ab5753171c86e456ef.gif",use_column_width=True)
    #vac_pred = False
    df_sta = state_data()
    df_sta = df_sta[df_sta['State'] == state_select]
    df_sta = df_sta.loc[(df_sta['Date'] >= start_date)
                        & (df_sta['Date'] <= end_date)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sta["Date"], y=df_sta['Deaths'], name='Deaths',
                            line=dict(color='red', width=4)))
        
    fig.update_layout( xaxis_title='Month',
                    yaxis_title='Deaths',width=850,height=600)

    st.markdown("## **Death cases in %s** " %( state_select))
    st.plotly_chart(fig,use_container_width=True,use_column_width=True)

    st.dataframe(df_sta)
    state_total = get_total_dataframe(df_sta)
    if st.sidebar.checkbox("Show Analysis by State", True, key=2) :
        st.markdown("## **State level analysis of COVID Cases**")
        st.markdown("### Overall Confirmed, Recovered and " +
            "Deceased cases in %s yet" % (state_select))
        if not st.checkbox('Hide Graph', False, key=1):
            state_total_graph = pxi.bar(
            state_total, 
            x='Status',
            y='Number of cases',
            labels={'Number of cases':'Number of cases in %s' % (state_select)},
            color='Status',color_discrete_map = {"Confirmed":"#5412E9","Recovered":"green","Deaths":"red"})
            st.plotly_chart(state_total_graph,use_container_width=True,use_column_width=True)
            
        if state_select == 'India':
            state = state_data()
            x = state.groupby("State").sum().reset_index()
            x = x[x['State'] != 'India']
            x.sort_values(by="Deaths",ascending=False,inplace=True)
            fig = pxi.bar(x.head(10), x="State", y="Deaths",hover_data={'Confirmed',"Recovered"},color="State",
                            labels={'value':'No. of Cases',"variable":"Case Type"})
            st.markdown("### Comparing deaths in top 10 states") 
            st.plotly_chart(fig,use_container_width=True,use_column_width=True)

    if st.sidebar.checkbox("Death Cases Prediction", False, key=4):
        x = st.sidebar.date_input('End date when you want to end the prediction',value=date.today()+timedelta(days=1),
                            max_value=date.today()+timedelta(days=1460),min_value=date.today()+timedelta(days=1))
        st.markdown("## **State level prediction of Confirmed cases**")
        if not st.checkbox('Hide Graphs', False, key=2):
            df_sta['Date'] = pd.to_datetime(df_sta['Date']).dt.date
            new = df_sta.drop(columns={"Recovered","Confirmed","Confirmed","State"},axis=1)
            new.rename(columns={'Date':'ds','Deaths':'y'},inplace=True)
            new['ds'] = pd.to_datetime(new['ds']).dt.date
            m = Prophet(growth='linear',weekly_seasonality=False,yearly_seasonality=False,
                                changepoint_prior_scale=0.9,changepoint_range=0.9,interval_width=0.95)
            m.fit(new)
            future = m.make_future_dataframe(periods = 1460)
            forecast = m.predict(future)
            forecast['ds'] = pd.to_datetime(forecast['ds']).dt.date
            s = str(x)
            yy,mm,dd = s.split("-")
            yy,mm,dd = int(yy),int(mm),int(dd)
            ans = forecast[forecast['ds'] == date(yy,mm,dd)]
            s = str(dd)+"/"+str(mm)+"/"+str(yy)
            an = forecast[forecast['ds'] <= date(yy,mm,dd)]
            trace1 = go.Scatter(name = 'Predicted',mode = 'lines',x = list(an['ds']),y = list(an['yhat']),
                                    marker=dict(color='red',line=dict(width=3)))
            trace2 = go.Scatter(name = 'Actual', mode = 'lines',x = list(an['ds']), y = list(new['y']),
                                            marker=dict(color='blue',line=dict(width=4)))
            data = [trace1,trace2]
            layout = dict(xaxis=dict(title = 'Dates', ticklen=2, zeroline=True),
                                yaxis=dict(title = 'Deaths', ticklen=2, zeroline=True))
            f_graph=dict(data=data,layout=layout)
            st.markdown("### Death cases prediction in %s"%(state_select))
            st.plotly_chart(f_graph,use_container_width=True,use_column_width=True)
            st.markdown("The predicted number of deaths cases on " + str(s) +" in "+ state_select +" would be "+str(int(ans['yhat'])))
                
        if st.checkbox('View Weekly and Yearly trends', False, key=3):
            an['ds'] = pd.to_datetime(an['ds'])
            f = plot_plotly(m, an,xlabel="Date",ylabel="Cases")
            st.plotly_chart(f,use_container_width=True,use_column_width=True)

#SETTING TIMER TO RUN THE APP EVERYDAY
dat_to = datetime.datetime.today()
dat_tom = dat_to.replace(day=dat_to.day+1, hour=dat_to.hour, minute=dat_to.minute, second=dat_to.second)
delta_t=dat_tom-dat_to
secs=delta_t.seconds+1
t = Timer(secs, call)
state_call = Timer(secs,state_data)
t.start()   
state_call.start()

def imag(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def links(l, text, **style):
    return a(_href=l, _target="_blank", style=styles(**style))(text)

def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
    </style>
    """

    style_div = styles(
        left=0,
        bottom=0,
        width=percent(100),
        color="black",
        text_align="center",
        height=0,
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )
    body = p()
    foot = div(style=style_div)(hr(style=style_hr), body)

    st.sidebar.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)
        elif isinstance(arg, HtmlElement):
            body(arg)
    st.sidebar.markdown(foot, unsafe_allow_html=True)

def footer():
    myargs = [
        "<b>Technologies</b>:",br()," Python 3.8 ",
        links("https://www.python.org/", imag("https://i.imgur.com/ml09ccU.png",
        	width=px(18), height=px(18), margin= "0em")),
        ", Streamlit ",
        links("https://streamlit.io/", imag('https://docs.streamlit.io/en/stable/_static/favicon.png',
        	width=px(24), height=px(25), margin= "0em")),
        br(),
        "<b>Made by Snehee Patel</b>",br(),
        "Connect with me on :",br(),
        links("https://github.com/Snehee2901", imag('https://image.flaticon.com/icons/png/512/25/25231.png',
            width=px(24), height=px(25), margin= "0.25em")),
        links("https://www.linkedin.com/in/SneheePatel/", imag('https://image.similarpng.com/very-thumbnail/2020/07/Linkedin-logo-on-transparent-Background-PNG-.png',
        	width=px(24), height=px(25), margin= "0.25em")),
        links("https://www.kaggle.com/sneheepatel", imag('https://thumbnail.imgbin.com/5/4/0/imgbin-kaggle-predictive-modelling-data-science-business-predictive-analytics-4Mh2z1pTSSFjmKfX09tHiQrz7_t.jpg',
        	width=px(34), height=px(25), margin= "0.25em")),
        br(),
    ]
    layout(*myargs)

footer()
