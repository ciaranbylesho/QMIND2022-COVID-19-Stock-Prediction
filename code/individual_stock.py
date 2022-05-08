from datetime import datetime
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib as mlt
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

class IndividualStock():

    def __init__(self, index, stock):
        self.path=os.getcwd()
        self.index=index
        self.stock = stock
        try:
            self.covid_df = pd.read_csv("../datasets/COVID-19_Daily_Counts_of_Cases__Hospitalizations__and_Deaths.csv")
        except:
            self.covid_df = pd.read_csv("../datasets/Covid-19_data/COVID-19_raw_data.csv")
        self.stock_df = pd.read_csv("../datasets/stock_market_data/" + self.index + "/csv/"+self.stock+".csv")

        self.stock_formatdf = self.formatStockData()
        self.covid_formatdf = self.covid_data()
        self.label_df = self.label_data(self.stock_df)

        self.stock_with_labels = self.merge(self.stock_formatdf,self.label_df) #DF1
        self.dropRows(self.stock_with_labels,25)
        self.covid_with_labels = self.merge(self.covid_formatdf,self.label_df) #DF2
        #self.dropRows(self.covid_with_labels,25)
        self.covid_with_stock = self.merge(self.covid_formatdf,self.stock_formatdf)
        self.both_with_labels = self.merge(self.covid_with_stock,self.label_df) #DF3
        self.dropRows(self.both_with_labels,25)
        

    def formatStockData (self, printDataset=False, dropOld = True):
        df = self.stock_df
        # convert to datetime
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
        
        # drop anything lower than Feb 1st 2020 and reset the index of the dataframe
        df = df.loc[(df['Date'] > '2020-02-01')]
        # prev day shift columns
        # df = df[1:]
        # make label do before shift
        df['Prev Low'] = df['Low'].shift()
        df['Prev Open'] = df['Open'].shift()
        df['Prev Volume'] = df['Volume'].shift()
        df['Prev High'] = df['High'].shift()
        df['Prev Close'] = df['Close'].shift()

        df['SMA5'] = df['Close'].rolling(5).mean().shift()
        df['SMA5'] = df['Close'].rolling(5).mean().shift()
        df['SMA5'] = df['Close'].rolling(5).mean().shift()
        df['SMA5'] = df['Close'].rolling(5).mean().shift()
        df['SMA5'] = df['Close'].rolling(5).mean().shift()
        # add STD feature then reindex
        # drop anything lower than March 1st 2020 and reset the index of the dataframe
        df = df.loc[(df['Date'] > '2020-03-01')].reset_index().drop(['index'], axis=1)

        self.days_stat(df, 'Prev Close', 10)
        self.rollingData(df, 'Prev Close')

        # add std, skew and kurtosis IMPORTANT Take not of unecessary data loss...both functions remove 7 days
        df=self.STD_kurt_skew(df, columnName="Close")
        # dfLabel = self.label_data(df)
        #drop old columns
        if dropOld == True:
            df=df.drop(columns=['Low', 'Open', 'Volume', 'High', 'Close', 'Adjusted Close'])

        #df = self.stockDailyChange(df)

        # if the parameter dictates it, print the dataset
        if printDataset==True:
            print(df)

        return df

    def covid_data(self):
        df = self.covid_df
        df = df[['DATE_OF_INTEREST','CASE_COUNT', 'DEATH_COUNT', 'HOSPITALIZED_COUNT']] # only include "DATE_OF_INTEREST" & "CASE_COUNT" columns
        df['CURRENT_CASE']=df['CASE_COUNT'].diff() # subtract the row in column "CASE_COUNT" with the previous row

        df['Date'] = df['DATE_OF_INTEREST'].replace('/', '-', regex=True)
        df['Date'] = pd.to_datetime(df['Date'], format="%m-%d-%Y", yearfirst=True)
        
        df=self.STD_kurt_skew(df,columnName="CURRENT_CASE")

        df = df.iloc[1:, 1:]

        # vaccination
        vac = pd.read_csv("../datasets/Covid-19_data/vaccination_data.csv").iloc[:, :5]
        vac['Date'] = pd.to_datetime(vac['DATE'], format="%Y-%m-%d", yearfirst=True)
        vac = vac.drop(['DATE'], axis=1).rename({"ADMIN_DOSE1_DAILY":"dose1_daily", 
                                                            "ADMIN_DOSE1_CUMULATIVE":"dose1_total", 
                                                            "ADMIN_DOSE2_DAILY": "dose2_daily", 
                                                            "ADMIN_DOSE2_CUMULATIVE": "dose2_total"},
                                                            axis=1)

        df = df.merge(vac, on='Date', how='left')
        
        df[df.filter(regex='dose').columns]= df.filter(regex='dose').fillna(0)
        #self.days_stat(df, 'CURRENT_CASE', 25)
        #self.rollingData(df, 'CURRENT_CASE')

        # df = df.drop(['DATE_OF_INTEREST'], axis=1)
        return df


    def STD_kurt_skew(self,df,columnName, days=7):
    
        # make new column and shift all the cells so current cell is not included
        df[str(days)+'-DayStd'] = df[columnName].rolling(days).std().shift()
        df[str(days)+'-Skew'] = df[columnName].rolling(days).skew().shift()
        df[str(days)+'-Kurt'] = df[columnName].rolling(days).kurt().shift()

        

        return df

    def label_data(self,df):

        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

        # drop anything lower than Feb 1st 2020 and reset the index of the dataframe
        df = df.loc[(df['Date'] > '2020-02-01')]

        # add STD feature then reindex
        # drop anything lower than March 1st 2020 and reset the index of the dataframe
        df = df.loc[(df['Date'] > '2020-03-01')].reset_index().drop(['index'], axis=1)

        # df['time'] = pd.to_datetime(df['time'],unit='s')
        # df['time'] = df['time'].apply(lambda x: x.date())

        df['Label']=df['Close']-df['Close'].shift()
        #Comment out to get previous function VVVVV next 2 lines
        df['Label']=df['Label'].rolling(7).mean()
        df=df.iloc[7:]
        df['Label'][df['Label']>0]=1
        df['Label'][df['Label']<=0]=0
        # df['time'] = pd.to_datetime(df['time'],infer_datetime_format=True)
        # df['Date'] = df['time']
        dfLabel = df[['Date','Label']]
        return dfLabel


    def merge(self, df1, df2):
        df = df1.merge(df2, on= 'Date')
        first_col = df['Date']
        df.drop('Date', axis=1, inplace=True)
        df.insert(0, 'Date', first_col.values)
        return df

    def days_stat(self, df, columnName, days):  
            for y in range(days):      
                df[columnName + "_days" + str(y+1)] = df[columnName].shift(int(1+y))

    def rollingData (self, df, columnName, days=5):
        for y in range(days):      
            df[columnName + "_" + str((y+1)*5)] = df[columnName].rolling((y+1)*5).mean().shift()
    
    def dropRows (self, df, days):
        df.drop(df.head(days).index, inplace = True)

    def visualize_data(self, max_date=0, n=0):        
        max=self.stock_df['Close'].max()
        df = self.both_with_labels.copy()

        df["Close"]=df["Prev Close"].shift(-1)/(max)

        df=df[['Date', 'Close']]
        df=df.merge(self.label_df, how='inner',on='Date')
        
        plt.figure(figsize=(40,10))
        if n == 0:
            n = len(df)
        xdate=np.array(mlt.dates.date2num(df['Date']))
        xdate_orig = np.copy(xdate)

        xdate=(xdate-xdate[0])/(xdate[-1]-xdate[0])
        for i in range(len(xdate)-1):
            if df['Label'][i]==0:
                plt.axhspan(0, 1, xdate[i], xdate[i+1],facecolor='r', alpha=0.25)
            else:
                plt.axhspan(0, 1, xdate[i], xdate[i+1], facecolor='g', alpha=0.25)

        min_date = df.iloc[0, 0]
        self.min_date = min_date
        if max_date != 0:
            covid_filtered = self.covid_df.loc[pd.to_datetime(self.covid_df['DATE_OF_INTEREST'], format="%m/%d/%Y") < max_date]
            xlim_max = max_date
            df = df.loc[df['Date'] > min_date].loc[df['Date'] < max_date]
        else:
            covid_filtered = self.covid_df
            xlim_max = xdate_orig[-1]
                                           
        covid_graph = covid_filtered.loc[pd.to_datetime(covid_filtered['DATE_OF_INTEREST'], format="%m/%d/%Y") > min_date, 'CASE_COUNT_7DAY_AVG']
        covid_graph2 = covid_filtered.loc[pd.to_datetime(covid_filtered['DATE_OF_INTEREST'], format="%m/%d/%Y") > min_date, 'CASE_COUNT']
        covid_date = covid_filtered.loc[pd.to_datetime(covid_filtered['DATE_OF_INTEREST'], format="%m/%d/%Y") > min_date, 'DATE_OF_INTEREST']
        
        
        
        plt.plot(df['Date'], (df['Close']-np.min(df['Close']))/(np.max(df['Close'])-np.min(df['Close']))*0.7+0.15, 'k', lw=5, label='Stock Price')
        plt.plot(
                pd.to_datetime(
                    covid_date, format="%m/%d/%Y", yearfirst=True), 
                ((covid_graph2-np.min(covid_graph2))/(np.max(covid_graph2)-np.min(covid_graph2))*0.7)+0.15, 'tab:blue', lw=5, label='COVID-19 Cases (Daily)')
        plt.plot(
                pd.to_datetime(
                    covid_date, format="%m/%d/%Y", yearfirst=True), 
                ((covid_graph-np.min(covid_graph))/(np.max(covid_graph)-np.min(covid_graph))*0.7)+0.15, 'tab:purple', lw=5, label='COVID-19 Cases (7-Day Avg.)')



        plt.title(f"Stock Activity Labeling for {self.stock}", fontsize=56, pad=25)
        plt.xticks(rotation=90, fontsize=36)
        plt.yticks(fontsize=36)
        plt.xlabel("Date", fontsize=48)
        plt.xlim([xdate_orig[0], xlim_max])
        plt.ylim([0, 1])
        plt.ylabel('Normalized Data', fontsize=40)
        plt.legend(fontsize=30, loc=2)

    def fifty_trials(self, data="both"):
        if data == 'both':
            df = self.both_with_labels.dropna()
        elif data == 'covid':
            df = self.covid_with_labels.dropna()
        else:
            df = self.stock_with_labels.dropna()

        acc=[]
        for i in range(50):
            x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, 1:-1], df.iloc[:, -1], shuffle=True, test_size=0.2)

            scaler = StandardScaler()

            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

            rf = RandomForestClassifier()
            rf.fit(x_train, y_train)
            y_pred = rf.predict(x_test)
            acc.append(accuracy_score(y_pred, y_test))
        return np.mean(acc), np.std(acc), (sum(y_train)+sum(y_test))/(len(y_train)+len(y_test))
    
    def kFold(self,data='both'):
        if data == 'both':
            df = self.both_with_labels.dropna()
        elif data == 'covid':
            df = self.covid_with_labels.dropna()
        else:
            df = self.stock_with_labels.dropna()
        X = df.iloc[:,1:-1].values
        y = df.iloc[:,-1].values
    
        cv = KFold(n_splits=5,random_state=1,shuffle=True)
        model = RandomForestClassifier()
        scores = cross_val_score(model,X,y,scoring='accuracy',cv=cv,n_jobs=-1)
        return scores.mean(), scores.std(), sum(y)/len(y)
    
    def modelTestRun(self):
        df = self.both_with_labels.dropna()
        try:
            x_train = df.iloc[:250, 1:-1]
            x_test = df.iloc[250:, 1:-1]
            y_train = df.iloc[:250, -1]
            y_test = df.iloc[250:, -1]
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, shuffle=True, test_size=0.1)
        except:
            raise ValueError('Not enough datapoints in the dataset.')
        # x_train, x_val, y_train, y_val = train_test_split(df[:, 1:-1], df[:, -1], shuffle=False, test_size=0.445)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_val)
        acc_val = accuracy_score(y_pred, y_val)
        y_pred = rf.predict(x_test)
        acc_test = accuracy_score(y_pred, y_test)

        
        
        min_date = df.iloc[0, 0]
        self.min_date = min_date
        
        stock_price = self.stock_df['Close'].loc[self.stock_df['Date'] >= df['Date'][250]]
        # print(stock_price.iloc[0])
        
        current_shares = 1
        current_amt = stock_price.iloc[0]
        y_prev=0
        if y_pred[0] == 1:
            current_shares = 1
            amt_valid = False
        else:
            current_amt = stock_price.iloc[0]
            amt_valid = True
        

        
        for i in range(len(x_test)):
            # buy
            if y_pred[i] == 1 and y_prev == 0:
                current_shares = current_amt/stock_price.iloc[i]
                amt_valid = False
            # sell
            elif y_pred[i] == 0 and y_prev == 1:
                current_amt = stock_price.iloc[i]*current_shares
                amt_valid = True
        
        if amt_valid == False:
            current_amt = current_shares*stock_price.iloc[i]
        
        return current_amt, acc_val, acc_test, stock_price.iloc[0]
