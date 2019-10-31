#!/usr/bin/env python

# import standard libraries
import logging, warnings, argparse, datetime

# import non-standard libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from imblearn.over_sampling import SMOTE

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--transactions", "-t", help="transactions .csv file path", default="transactions_obf.csv")
parser.add_argument("--labels", "-l", help="labels .csv file path", default="labels_obf.csv")
parser.add_argument("--sample_data", "-i", help="sample_data .csv file path", default="sample_data.csv")
parser.add_argument("--output", "-o", help="output .log file path")
args = parser.parse_args()

# add logging
if args.output:
    logfile = args.output
else:
    logfile = "flag_for_fraud_v3_" + str(datetime.datetime.now().strftime('%Y-%m-%d')) + ".log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(logfile)
logger.addHandler(handler)

# ignore warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
warnings.filterwarnings("ignore", category=Warning)

# suppress scientific notation
pd.set_option('display.float_format', lambda x: '%.5f' % x)


'''
This is a machine learning model for fraud prediction. 

In addition to the transactions_file and the labels_file, the model takes 
2 additional inputs: "churn_threshold" and "smote_ratio". 

The model is currently designed to maximize recall score, 
while keeping the ratio of false positives to true positives at reasonable levels. 

We want to max recall over other metrics since our goal is flagging for further 
examination, rather than trying to get the highest accuracy in fraud predictions. 

The best performing models captured 85% - 93% of fraudulent transactions (recall), while 
keeping the percentage of transactions flagged for fraud in the 3% - 10% range.

The final model is an ensemble of tree based methods. 

"churn_threshold" is used for regression-based models which round up the output to turn it 
into a classifier. A typical classification model would use 0.5, but this model can reasonably 
use values in the 0.02 - 0.50 range, with lower values flagging more transactions. 

"smote_ratio" is used for oversampling; it determines the ratio of the minority class to the 
majority class. SMOTE uses synethic data samples to help make the minority class easier to detect.
Reasonable smote_ratio values for this model seem to be in the 0.05 - 0.50 range. 

Default churn_threshold is set to 0.2 and default smote_ratio is set to 0.1.
'''


def transform_train_data(transactions_file, labels_file):
    
    # load transaction and labels data
    transactions = pd.read_csv(transactions_file)
    labels = pd.read_csv(labels_file)

    # set all fraudulent transaction to value of "1"
    labels['fraud'] = 1
        
    # merge data
    df = transactions.merge(labels, how='outer', on='eventId')
    
    # set non-fraud transactions to value of "0"
    df['fraud'] = df['fraud'].fillna(0).astype('uint8')
    
    # drop reported time column
    df = df.drop(['reportedTime'], axis=1)
        
    # split transactionTime into month, day of week, hour, etc
    df['date'] = pd.to_datetime(df['transactionTime'])
    df['month'] = df['date'].dt.month
    df['dayOfWeek'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['dayOfYear'] = df['date'].dt.dayofyear
    df['minute'] = df['date'].dt.minute
    df['year'] = df['date'].dt.year
    
    ## create hour and month blocks for better metrics on activity by merchant ids
    df['hourBlock'] = ((df['year'] - 2017) * 8760) + ((df['dayOfYear'] - 1) * 24) + df['hour']
    df['monthBlock'] = ((df['year'] - 2017) * 12) + df['month']
    
    # sort dataframe by datetime
    df = df.sort_values(by='date',ascending=True)
    df = df.reset_index(drop=True)
    
    # convert categoricals into numeric values
    df['merchantIdCat'] = df['merchantId'].astype('category').cat.codes
    
    # create fraud dataframe
    fraud_df = df[df['fraud'] == 1]
    
    # add feature for transactions as % of available cash
    df['pctAvailCash'] = df['transactionAmount'] / df['availableCash']
    
    # add feature for fraud as % of POS method    
    fraudPctPosDict = (fraud_df['posEntryMode'].value_counts() / df['posEntryMode'].value_counts()).fillna(0).to_dict()
    df['fraudPctPosEntry'] = df['posEntryMode'].map(fraudPctPosDict)
    
    # add merchant country stats
    fraudPctMerchantCountryDict = (fraud_df['merchantCountry'].value_counts() / df['merchantCountry'].value_counts()).fillna(0).to_dict()
    df['fraudPctMerchantCountry'] = df['merchantCountry'].map(fraudPctMerchantCountryDict)
    
    # add merchant ID stats
    df['merchantIdCount'] = df['merchantId'].map(df['merchantId'].value_counts().to_dict())
    df['merchantIdUnder10'] = (df['merchantIdCount'] < 10).astype('uint8')
    df['merchantIdUnique'] = (df['merchantIdCount'] == 1).astype('uint8')
    
    # add fraud % by merchant id
    fraudPctMerchantIdDict = (fraud_df['merchantId'].value_counts() / df['merchantId'].value_counts()).fillna(0).to_dict()
    df['fraudPctMerchantId'] = df['merchantId'].map(fraudPctMerchantIdDict)
    
    # create merchant id by hour metrics
    df['merchantIdHour'] = df['merchantIdCat'].map(str) + "_" + df['hourBlock'].map(str)
    df['merchantIdHourCount'] = df['merchantIdHour'].map(df['merchantIdHour'].value_counts().to_dict())
    
    # Note: consider taking hourBlock -1 and hourBlock +1 as well
    
    # create merchant id by month metrics
    df['merchantIdMonth'] = df['merchantIdCat'].map(str) + "_" + df['monthBlock'].map(str)
    df['merchantIdMonthCount'] = df['merchantIdMonth'].map(df['merchantIdMonth'].value_counts().to_dict())
    
    # orders in hour block as percentage of month for merchant id
    df['merchantIdPctHour'] = df['merchantIdHourCount'] / df['merchantIdMonthCount']   
    
    # drop transactionTime and datetime
    df = df.drop(['transactionTime', 'date'], axis=1)
    
    # filter only int / float columns   
    df_num = df._get_numeric_data()
    df_num['eventId'] = df['eventId']

    return df_num

def transform_test_data(input_data):
    
    train_df = transform_train_data(args.transactions, args.labels)

    # load transaction and labels data
    df = pd.read_csv(input_data)
   
    # split transactionTime into month, day of week, hour, etc
    df['date'] = pd.to_datetime(df['transactionTime'])
    df['month'] = df['date'].dt.month
    df['dayOfWeek'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['dayOfYear'] = df['date'].dt.dayofyear
    df['minute'] = df['date'].dt.minute
    df['year'] = df['date'].dt.year
    
    ## create hour and month blocks for better metrics on activity by merchant ids
    df['hourBlock'] = ((df['year'] - 2017) * 8760) + ((df['dayOfYear'] - 1) * 24) + df['hour']
    df['monthBlock'] = ((df['year'] - 2017) * 12) + df['month']
    
    # sort dataframe by datetime
    df = df.sort_values(by='date',ascending=True)
    df = df.reset_index(drop=True)
    
    # convert categoricals into numeric values
    df['merchantIdCat'] = df['merchantId'].astype('category').cat.codes
    
    # create fraud dataframe
    fraud_df = train_df[train_df['fraud'] == 1]
    
    # add feature for transactions as % of available cash
    df['pctAvailCash'] = df['transactionAmount'] / df['availableCash']
    
    # add feature for fraud as % of POS method    
    fraudPctPosDict = (fraud_df['posEntryMode'].value_counts() / df['posEntryMode'].value_counts()).fillna(0).to_dict()
    df['fraudPctPosEntry'] = df['posEntryMode'].map(fraudPctPosDict)
    
    # add merchant country stats
    fraudPctMerchantCountryDict = (fraud_df['merchantCountry'].value_counts() / df['merchantCountry'].value_counts()).fillna(0).to_dict()
    df['fraudPctMerchantCountry'] = df['merchantCountry'].map(fraudPctMerchantCountryDict)
    
    # add merchant ID stats
    df['merchantIdCount'] = df['merchantId'].map(df['merchantId'].value_counts().to_dict())
    df['merchantIdUnder10'] = (df['merchantIdCount'] < 10).astype('uint8')
    df['merchantIdUnique'] = (df['merchantIdCount'] == 1).astype('uint8')
    
    # add fraud % by merchant id
    fraudPctMerchantIdDict = (fraud_df['merchantId'].value_counts() / df['merchantId'].value_counts()).fillna(0).to_dict()
    df['fraudPctMerchantId'] = df['merchantId'].map(fraudPctMerchantIdDict)
    
    # create merchant id by hour metrics
    df['merchantIdHour'] = df['merchantIdCat'].map(str) + "_" + df['hourBlock'].map(str)
    df['merchantIdHourCount'] = df['merchantIdHour'].map(df['merchantIdHour'].value_counts().to_dict())
    
    # Note: consider taking hourBlock -1 and hourBlock +1 as well
    
    # create merchant id by month metrics
    df['merchantIdMonth'] = df['merchantIdCat'].map(str) + "_" + df['monthBlock'].map(str)
    df['merchantIdMonthCount'] = df['merchantIdMonth'].map(df['merchantIdMonth'].value_counts().to_dict())
    
    # orders in hour block as percentage of month for merchant id
    df['merchantIdPctHour'] = df['merchantIdHourCount'] / df['merchantIdMonthCount']   
    
    # drop transactionTime and datetime
    df = df.drop(['transactionTime', 'date'], axis=1)
    
    # filter only int / float columns   
    df_num = df._get_numeric_data()
    df_num['eventId'] = df['eventId']

    return df_num


def build_model(train_data, churn_threshold=0.2, smote_ratio=0.1):

    df = train_data
  
    # split data into X and y
    X = df.drop(['fraud', 'eventId'], axis=1)
    y = df['fraud']
    
    # split data based on time period; training data on 1st half; validation on 2nd half  
    rows = len(df)
    rows_half = np.round(rows / 2, 0).astype('int')
    X_train = X[:rows_half]
    X_test = X[rows_half:]
    y_train = y[:rows_half]
    y_test = y[rows_half:]
    
    # use SMOTE to oversample minority values in training data
    sm = SMOTE(sampling_strategy=smote_ratio)
    X_train, y_train = sm.fit_sample(X_train, y_train)    
  
    # build random forest classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train,y_train)    
    rfc_pred = rfc.predict(X_test)

    # set parameter for classification threshold on regressor models
    churn_req = churn_threshold
    
    # build random forest regressor
    rfr = RandomForestRegressor()
    rfr_model = rfr.fit(X_train,y_train)    
    rfr_pred = rfr.predict(X_test)        
    rfr_result = rfr_model.score(X_test, y_test)
    rfr_pred_binom = (rfr_pred > churn_req) * 1    
    
    # build Gradient Boosting Regressor
    gbr = GradientBoostingRegressor()
    gbr_model = gbr.fit(X_train,y_train)
    gbr_pred = gbr.predict(X_test)
    gbr_result = gbr_model.score(X_test, y_test)
    gbr_pred_binom = (gbr_pred > churn_req) * 1

    # build ensemble model    
    pred_df = pd.DataFrame({'rfc': rfc_pred,'rfr': rfr_pred_binom, 'gbr': gbr_pred_binom})
    pred_df['sum'] = pred_df['rfc'] + pred_df['rfr'] + pred_df['gbr']
    pred_df['preds'] = pred_df['sum'].replace([2,3],1)

    # create confusion matrix
    conf_matx = confusion_matrix(y_test, pred_df['preds'])

    # create classification report
    clasf_rep = classification_report(y_test, pred_df['preds'])

    # total number of flagged transactions
    flagged_trans = conf_matx[0][1] + conf_matx[1][1]

    # percentage of flagged transactions out of total transactions tested
    flagged_pct = str(np.round(flagged_trans / len(y_test),4) * 100) + "%"
    
    # log info
    logger.info("=== Confusion Matrix ===")
    logger.info(conf_matx)

    logger.info("\n")
    logger.info("=== Total # of Flagged Transactions ===")
    logger.info(flagged_trans)

    logger.info("\n")
    logger.info("=== % of Flagged Transactions ===")
    logger.info(flagged_pct)

    logger.info("\n")
    logger.info("=== Classification Report ===")
    logger.info(clasf_rep)
    
    logger.info("\n")
    logger.info('=== RUN COMPLETE AT ' + str(datetime.datetime.now().strftime('%Y-%m-%d')) + ' ===')


def flag_for_fraud(train_data, test_data, churn_threshold=0.2, smote_ratio=0.1):

    df = train_data
  
    # split train data into X and y
    X_train = df.drop(['fraud', 'eventId'], axis=1)
    y_train = df['fraud']

    # set up X-test
    X_test = test_data.drop(['eventId'], axis=1)
      
    # use SMOTE to oversample minority values in training data
    sm = SMOTE(sampling_strategy=smote_ratio)
    X_train, y_train = sm.fit_sample(X_train, y_train)    
  
    # build random forest classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train,y_train)    
    rfc_pred = rfc.predict(X_test)

    # set parameter for classification threshold on regressor models
    churn_req = churn_threshold
    
    # build random forest regressor
    rfr = RandomForestRegressor()
    rfr_model = rfr.fit(X_train,y_train)    
    rfr_pred = rfr.predict(X_test)        
    rfr_pred_binom = (rfr_pred > churn_req) * 1    
    
    # build Gradient Boosting Regressor
    gbr = GradientBoostingRegressor()
    gbr_model = gbr.fit(X_train,y_train)
    gbr_pred = gbr.predict(X_test)
    gbr_pred_binom = (gbr_pred > churn_req) * 1

    # build ensemble model    
    pred_df = pd.DataFrame({'eventId': test_data['eventId'] ,'rfc': rfc_pred, 'rfr': rfr_pred_binom, 'gbr': gbr_pred_binom})
    pred_df['sum'] = pred_df['rfc'] + pred_df['rfr'] + pred_df['gbr']
    pred_df['preds'] = pred_df['sum'].replace([2,3],1)

    return pred_df


    


if __name__ == "__main__":
    train_df = transform_train_data(args.transactions, args.labels)
    test_df = transform_test_data(args.sample_data)
    flag_for_fraud(train_df, test_df)
