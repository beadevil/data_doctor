import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.formula.api import ols
from bioinfokit.analys import stat

st.title("DATA DOCTOR")

st.write("### ( INFORMATION | PLOTING | ANOVA | REGRESSION )")
st.write("")
st.write("")

st.write("#### UPLODE DATA")
csv_file=st.file_uploader("",type=['csv'])

def cl_data(df):
    aaa=1
    if st.button("DATA INFORMATION After Cleaning"):
        aaa=0
    if aaa==0:
        st.write("#### DATA INFORMATION after data cleaning")
        st.write(df)
        st.write("")
        st.write("Column name after data cleaning : ",df.columns)
        st.write("")
        st.write("Rows after data cleaning : ",len(df.index))
        st.write("Columns after data cleaning : ",len(df.columns))
        st.write("")
        df_c_o=df.select_dtypes(include=['object']).columns
        df_c_n=df.select_dtypes(include=['int64','float64']).columns
        st.write("Columns with Numerical data after data cleaning : ",len(df_c_n),df_c_n)
        st.write("Columns with Objective data after data cleaning : ",len(df_c_o),df_c_o)
        st.write("")
        st.write("Total Null values after data cleaning : ",sum(df.isnull().sum()))
        st.write("Column wise null value after data cleaning : ",df.isnull().sum())
        sns.heatmap(df.isnull())
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write("")
        st.write("Null value Plot after data cleaning :")
        st.pyplot()
    if st.button("Close DATA INFORMATION "):
        aaa=1


if csv_file is not None:
    df=pd.read_csv(csv_file)
    st.write('#### DATA VIEW :')
    st.write(df)
    st.write("")
    st.write("")
    st.write("")
    df_c_o=df.select_dtypes(include=['object']).columns
    df_c_n=df.select_dtypes(include=['int64','float64']).columns
    st.write("#### DATA INFORMATION :")
    st.write("")
    st.write("")
    bbb=1
    if st.button("DATA INFORMATION"):
        bbb=0
    if bbb==0:
        st.write("")
        st.write("")
        st.write("Column name : ",df.columns)
        st.write("")
        st.write("Rows : ",len(df.index))
        st.write("Columns : ",len(df.columns))
        st.write("")
        st.write("Columns with Numerical data : ",len(df_c_n),df_c_n)
        st.write("Columns with Objective data : ",len(df_c_o),df_c_o)
        st.write("")
        st.write("Total Null values : ",sum(df.isnull().sum()))
        st.write("Column wise null value : ",df.isnull().sum())
        sns.heatmap(df.isnull())
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write("")
        st.write("Null value Plot of raw data :")
        st.pyplot()
        st.write("")
        st.write("")
    if st.button("Close DATA INFORMATION"):
        aaa=1
    st.write("#### DATA Cleaning :")
    st.write("")
    dt_c=st.selectbox("Select cleaning process :",("None","Drop null value","Delete Row / Column"))
    if dt_c=="None":
        df.to_csv('original.csv', index=False)
    if dt_c==("Drop null value"):
        if sum(df.isnull().sum())>0:
            df=pd.read_csv("original.csv")
            df=df.dropna()
            df.to_csv('original.csv', index=False)
            st.write("Null value droped.")
        else:
            st.write("There is no null value.")
    if dt_c==("Delete Row / Column"):
        df=pd.read_csv("original.csv")
        del_row=st.selectbox("Select Row :",(df.index))
        if st.button("Delete Row"):
            df.drop(del_row,inplace = True)
            df.to_csv('original.csv', index=False)
        del_col=st.selectbox("Select Column :",(df.columns))
        if st.button("Delete Column"):
            df.drop(del_col,inplace = True,axis=1)
            df.to_csv('original.csv', index=False)

    cl_data(df)
    st.write("")
    st.write("")
    st.write("#### PLOT Your DATA :")
    st.write("")
    dt_p=st.selectbox("Select PLOT Type :",("None","BOX PLOT","BAR PLOT","PIE PLOT"))
    if dt_p=="BOX PLOT":
        box_x=st.selectbox("Select X axis (Column name) :",df.columns)
        box_y=st.selectbox("Select Y axis (Column name) :",df.drop(box_x,axis=1).columns)
        bp =sns.boxplot(x=box_x, y=box_y, data=df, color='#99c2a2')
        bp =sns.swarmplot(x=box_x, y=box_y, data=df, color='#7d0013')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.show()
        st.pyplot()
    if dt_p=="BAR PLOT":
        bar_x=st.selectbox("Select X axis (Column name) :",df.columns)
        bar_y=st.selectbox("Select Y axis (Column name) :",df.drop(bar_x,axis=1).columns)
        if bar_x and bar_y in df.columns:
            plt.bar(df[bar_x],df[bar_y],color="r", width = 0.1)
            plt.xlabel(bar_x)
            plt.ylabel(bar_y)
            plt.show()
            st.pyplot()
        else:
            st.write("This Column is not avalable.")
    if dt_p=="PIE PLOT":
        pie_c=st.selectbox("Select Column name :",df.columns)
        if pie_c in df.columns:
            if sum(df.isnull().sum())==0:
                if pie_c in df.select_dtypes(include=['int64','float64']).columns:
                    count = 0
                    for number in df[pie_c]:
                        if number < 0:
                            count += 1
                        if count==0:
                            bp =plt.pie(df[pie_c],labels =df[pie_c])
                            plt.show()
                            st.pyplot()
                        else:
                            st.write("This Column have -ve data.")
                else:
                    st.write("This Column have object type data.")
            else:
                st.write("Please clean your data.")
        else:
            st.write("This Column is not avalable.")
    if sum(df.isnull().sum())==0:
        df_c_o=df.select_dtypes(include=['object']).columns
        for i in df_c_o:
            label=LabelEncoder()
            df[i]=label.fit_transform(df[i])
    st.write("")
    st.write("#### ANOVA :")
    an_va=st.selectbox("",("None","ANOVA"))
    if an_va=="ANOVA":
        if sum(df.isnull().sum())==0:
            anova_x=st.selectbox("Select Column name :",df.columns)
            anova_y=st.selectbox("Select Column name :",df.drop(anova_x,axis=1).columns)
            mod=ols(anova_x+'~'+anova_y,data=df).fit()
            aov_table=sm.stats.anova_lm(mod,type=2)
            st.write("ANOVA TABLE :",aov_table)
            res = stat()
            res.tukey_hsd(df=df, res_var=anova_x,xfac_var=anova_y, anova_model=anova_x+"~"+anova_y)
            st.write("ANOVA SUMMARY TABLE :",res.tukey_summary)
        else:
            st.write("Please clean your data.")
    st.write("")
    st.write("")
    st.write("### Select your REGRESSION")
    reg_t=st.selectbox("",("None","Linear REGRESSION","Logistic REGRESSION"))

    if reg_t==("Logistic REGRESSION"):
        if sum(df.isnull().sum())==0:
            st.write("#### Model Details : ")
            st.write("")
            col=df.columns
            colx=col.drop(col[len(col)-1])
            coly=col[len(col)-1]
            x=df[colx]
            y=df[coly]
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
            model = LogisticRegression()
            model.fit(x_train, y_train)
            predictions=model.predict(x_test)
            st.write("REGRESSION Coefficient",model.coef_)
            st.write("")
            st.write(classification_report(y_test,predictions))
            st.write("")
            st.write(confusion_matrix(y_test,predictions))
            st.write("")
            st.write("Model Score : ",model.score(x_train, y_train))
            st.write("")
            st.write(model.predict_log_proba(x_test))
            st.write("")
        else:
            st.write("Please clean your data.")

    if reg_t==("Linear REGRESSION"):
        if sum(df.isnull().sum())==0:
            st.write("#### Model Details : ")
            st.write("")
            col=df.columns
            colx=col.drop(col[len(col)-1])
            coly=col[len(col)-1]
            x=df[colx]
            y=df[coly]
            if len(y.unique())==2:
                st.write("Linear REGRESSION not applicable on this data.")
            else:
                x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
                model=linear_model.LinearRegression()
                model.fit(x,y)
                predictions=model.predict(x_test)
                st.write("REGRESSION Coefficient",model.coef_)
                st.write("")
                st.write(classification_report(y_test,predictions))
                st.write("")
                st.write(confusion_matrix(y_test,predictions))
                st.write("")
                st.write("Model Score : ",model.score(x_train, y_train))
                st.write("")
        else:
            st.write("Please clean your data.")
