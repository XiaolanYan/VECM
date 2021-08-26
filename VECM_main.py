#utf-8
#@author: Xiaolan Yan
#@Created on 12/24/2020
#reference:
#https://www.statsmodels.org/stable/vector_ar.html
#box test
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import VECM
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import select_order
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
from statsmodels.tsa.vector_ar.irf import *
#helper
def get_most_frequent_order(ic_list):
    from collections import Counter
    d = Counter(ic_list)
    max_value = max(d.values())
    return [k for k,v in d.items() if v == max_value]

def calc_R_Squared(pred, actual):

    if len(pred) != len(actual):
        print('not same length')
        return np.nan


    # RR means R_Square
    RR = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - np.mean(actual,axis=0)) ** 2)
    return RR

def calc_adjusted_R_Squared(pred, actual, feature_dimension):

    # RR means R_Square
    RR = calc_R_Squared(pred,actual)

    n = len(pred);
    p = feature_dimension
    Adjust_RR = 1 - (1 - RR) * (n - 1) / (n - p - 1)
    # Adjust_RR means Adjust_R_Square

    return Adjust_RR

def calc_F_stat(pred,actual,feature_dimension):


    # F = RSS/ESS *(n-p-1)/p, RSS:残差平方和， ESS: 回归平方和，n:样本量，p：参数个数
    RSS = np.sum((actual - pred) ** 2)
    ESS = np.sum((actual - np.mean(actual,axis=0)) ** 2)
    n = len(pred)
    p = feature_dimension
    return RSS/ESS *(n-p-1)/p


# ----configuration----
__FILE__PATH = "input_data.csv"
class VECM:
    max_lag_input =10
    information_criteria = "aic" #'aic','bic', 'hqic', 'fpe'
    periods =20
    P = 0.7
    predict_num = 12
    order_coint_method =2 #定阶的方法，选择1与R里一致，2直接选用py statsmodels的定阶方法

def VECM_main(config):
    output = {}

    # ----data input and processing----
    data  = pd.read_csv(__FILE__PATH)
    data_cols = ["X"+i for i in list(data.columns) ]
    data_cols[-1] = "Y"
    data.columns = data_cols
    # ----determine order----
    # Note: In R, deterministic type = c("const", "trend", "both", "none"), and the function VARselect is based on VAR
    # In Python, deterministic str {"nc", "co", "ci", "lo", "li"}, based on VECM
    # it should be constant and outside of the cointegration->"co"
    # R command: Lag_Select_Info <- VARselect(VEC_Asset_Data, lag.max = lagInput, type = "const", season = NULL, exogen = NULL)
    deterministic_param ="co"

    ###----method 1----
    # this method translates from corresponding R code.
    if config.order_coint_method ==1:
        VARmodel = VAR(data)
        VARmodel.select_order(maxlags= config.max_lag_input,trend="c")
        IC_selection_list = []
        for i in range(2,config.max_lag_input+1):
            lag_select_info = VARmodel.select_order( maxlags= i , trend = "c").selected_orders
            selected= lag_select_info[config.information_criteria]
            IC_selection_list.append(selected)
        lag_determined = get_most_frequent_order(IC_selection_list)[0]

        #johansen procedure
        #确定协整数
        johansen_result =coint_johansen(data,det_order=0,k_ar_diff =max(lag_determined-1,2))
        test_stat = johansen_result.trace_stat
        critical_vals = johansen_result.trace_stat_crit_vals
        max_len = len(test_stat)
        coint_temp = 11
        for i in range(max_len-1,0,-1):
            if test_stat[i] ==None:
                continue
            if test_stat[i] >critical_vals[i,2]:
                coint_temp = i
                break
        coint_temp = max_len-i
        coint_number = min(data.shape[1]-coint_temp,data.shape[1]-1)


    # vecm.select_order(data,maxlags=config.max_lag_input,deterministic="co",seasons= 0,exog= None)

    else:
        ###----method 2----
        # this method takes statsmodel's functions of VECM to calculate order and cointegration number directly
        lag_select_info = select_order(data, maxlags=config.max_lag_input, deterministic = deterministic_param)
        lag_determined =lag_select_info.aic
        coint_number = select_coint_rank(data,det_order= 0,k_ar_diff=lag_determined).rank
        coint_number = min(coint_number,data.shape[1]-1)

    # ----build model and fit----
    VECMmodel = VECM(endog=data,coint_rank = coint_number,k_ar_diff=lag_determined,deterministic = deterministic_param)
    VECMresult = VECMmodel.fit(method="ml")
    # print(VECMresult.summary())
    # VECMresult.fittedvalues
    # VECMresult.det_coef

    #Forecast Y
    VECMpredict = VECMresult.predict(steps=config.predict_num,alpha= config.P)
    pred_res = {}
    pred_res["val"] = VECMpredict[0][:,-1]
    pred_res["lower"] = VECMpredict[1][:,-1]
    pred_res["upper"] = VECMpredict[2][:,-1]
    pred_res = pd.DataFrame(pred_res)
    output["ForecastY"] =pred_res


    #residual standard error
    resid = VECMresult.resid
    resid_standard_error = np.std(resid[:,-1])
    output["residucal_std"] = resid_standard_error

    # Adjusted R
    pred = VECMresult.fittedvalues
    actual = pred+resid
    pred_y = pred[:,-1]
    actual_y = actual[:,-1]

    #???
    #对于VECM而言，计算adj R squard 的维数应该是什么？？ie. 是滞后的阶数K or ncol or k*ncol ???
    #以及算残差是所有变量的残差都要计算吗？or 可以只考虑我们需要的y吗？

    adj_RR_ = calc_adjusted_R_Squared(pred,actual,lag_determined)
    adj_RR_Y = calc_adjusted_R_Squared(pred_y,actual_y,data.shape[1]-1)

    RR_  = calc_R_Squared(pred,actual)
    RR_Y = calc_R_Squared(pred_y,actual_y)

    output["adj_R_squared"] =adj_RR_Y

    #Ftest的问题跟adj 一样
    #F = RSS/ESS *(n-p-1)/p, RSS:残差平方和， ESS: 回归平方和，n:样本量，p：参数个数
    F_stat = calc_F_stat(pred,actual,lag_determined)
    output["F_stat"] =F_stat


    # granger causality test
    #R里面没有
    granger_test = VECMresult.test_granger_causality(-1)


    # Box test residual p value
    # ‘portmanteau’ tests.
    # method 1
    # y_resid = VECMresult.resid[:,-1]
    # residual_test_result = lb_test(x = y_resid,lags = round(np.log(VECMresult.resid.shape[0])),return_df=True)
    # residual_test_p_value = residual_test_result['lb_pvalue'][round(np.log(VECMresult.resid.shape[0]))]
    # output["residual_p_value"] = residual_test_p_value

    # method 2
    residual_test_result = VECMresult.test_whiteness(nlags=round(np.log(VECMresult.resid.shape[0])), signif=0.05, adjusted=False)
    residual_test_p_value = residual_test_result.pvalue
    output["residual_p_value"] = residual_test_p_value
    #f-test 有两类 normality test 和granger causality test


    # ----impulse response----
    irf_res  = VECMresult.irf(periods= config.periods)
    # irf_res= IRAnalysis(periods=  config.periods, svar = True,vecm =True)
    IRFtemp = irf_res.irfs
    irf_output = []
    #---这里的问题：
    # R里面只有X对Y 的脉冲响应，但python里做了Y对Y的，Python这个类的return没有详细的，所以不清楚是选列or行
    for i in range(config.periods):
        irf_output.append(IRFtemp[i][0:data.shape[1]-1,-1])
    output["IRF"] = np.array(irf_output)

    return output


if __name__ =="__main__":
    config =Config()
    output = VECM_main(config)
    print(output)
