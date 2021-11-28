#!/usr/bin/env python
# coding: utf-8

# # Predict student grade



import streamlit as st
import pandas as pd
import pickle
import numpy as np
import time
import random
import sys
import datetime
from sklearn.model_selection import train_test_split

st.title("Student At Risk Detection Web App")
st.subheader("Please choose the prediction method")
options1 = st.selectbox("Please Select", ['Select','Only Historic Attributes','Complete Attributes (Historic & Checkpoints)'])
st.subheader("This tool provides 2 solutions \n 1.Batch Prediction \n  2.Single Prediction")
options = st.selectbox("Please Select", ['Select','Batch Prediction','Single Prediction'])

if(options1 =='Only Historic Attributes'):
    loaded_model = pickle.load(open('streamlit_student_grade_prediction2.pkl', 'rb'))
else:
     loaded_model = pickle.load(open('streamlit_student_grade_prediction1.pkl', 'rb'))

dict_mean = {}
dict_mean["AgeCourseStart_mean"]= 19.57
dict_mean["AGE_ADMITTED_mean"]= 17.47
dict_mean["Reg_Hrs_mean"]= 13.5
dict_mean["HS_GPA_mean"]= 0.89
dict_mean["MATH_mean"]= 0.83
dict_mean["PHYS_mean"]= 0.82
dict_mean["Repeated_Grade_ITBP119_CSBP119_mean"]= 0.2
dict_mean["ITBP119_CSBP119_max_mean"]= 0.77
dict_mean["Repeated_Grade_ITBP219_CSBP219_mean"]= 0.24
dict_mean["ESPN_mean"]= 0.78
dict_mean["CSBP121_mean"]= 0.76
dict_mean["MATH105_mean"]= 0.73
dict_mean["CENG205_mean"]= 0.78
dict_mean["PHYS105_mean"]= 0.7
dict_mean["CENG202_mean"]= 0.75
dict_mean["Quiz1Norm_mean"]= 0.63
dict_mean["Quiz2Norm_mean"]= 0.57
dict_mean["HW1Norm_mean"]= 0.76
dict_mean["MTNorm_mean"]= 0.69

if(options =='Single Prediction'):
    
    def load_data1():
        df1 = pd.DataFrame({'CIT': ['Yes','No'],
                           'AcademicStanding': ['Goodstanding', 'Probation'],
                           'Citizenship':['Citizen','Non-Citizen'],
                           'Gender':['Male','female'],
                           'Sponsor': ['Yes', 'No'],
                           'AlainResident':['Yes','No']
                          })
        return df1
    df1 = load_data1()

    st.sidebar.subheader("User Input Parameters")
    AgeCourseStart = st.sidebar.slider("Age of the student when he/she took CSBP219 course", 0.00, 30.00)
    AGE_ADMITTED=st.sidebar.slider("Age of the student when he/she admitted to university", 0.00, 30.00)
    Reg_Hrs=st.sidebar.slider("Registered Hours in the current semester", 0, 23) 
    HS_GPA=st.sidebar.slider("HighSchoolGPA", 0.0, 1.0) 
    MATH=st.sidebar.slider("Math Grade in High School", 0.0, 1.0) 
    PHYS=st.sidebar.slider("Physics Grade in High School", 0.0, 1.0) 
    if(options1 =='Complete Attributes (Historic & Checkpoints)'):
        Quiz1Norm =st.sidebar.slider("Grade of Quiz 1", 0.0, 1.0) 
        Quiz2Norm =st.sidebar.slider("Grade of Quiz 2 ", 0.0, 1.0) 
        HW1Norm =st.sidebar.slider("Grade of HW 1", 0.0, 1.0) 
        MTNorm=st.sidebar.slider("Midterm exam Grade", 0.0, 1.0) 
    
    ITBP119_CSBP119_max=st.sidebar.slider("Grade of CSBP119 course", 0.0, 1.0) 
    ESPN=st.sidebar.slider("Grade of Academic English", 0.0, 1.0) 
    CSBP121=st.sidebar.slider("Grade of Programming Lab I", 0.0, 1.0)
    MATH105=st.sidebar.slider("Grade of Calculus I", 0.0, 1.0)
    CENG205=st.sidebar.slider("Grade of Digital Design & Computer Organization", 0.0, 1.0)
    PHYS105=st.sidebar.slider("Grade of General Physics I", 0.0, 1.0)
    CENG202=st.sidebar.slider("Grade of Discrete Mathematics", 0.0, 1.0)
    Repeated_Grade_ITBP119_CSBP119 =st.sidebar.slider("How many times student repeated CSBP119 course", 0, 3)  
    Repeated_Grade_ITBP219_CSBP219  =st.sidebar.slider("How many times student repeated CSBP219 course", 0, 3)
    CIT_q=st.sidebar.selectbox("Student in College of IT ", df1['CIT'].unique())
    if CIT_q == 'Yes':
        CIT = 1  
    else:
        CIT = 0 
    Gender_q=st.sidebar.selectbox("Gender", df1['Gender'].unique())
    if Gender_q == 'Female':
        Gender =1
    else:
        Gender =0
    AcademicStanding_q=st.sidebar.selectbox("Academic Standing", df1['AcademicStanding'].unique())
    if AcademicStanding_q == 'Goodstanding':
        AcademicStanding =1
    else:
        AcademicStanding =0
    Citizenship_q=st.sidebar.selectbox("Citizenship", df1['Citizenship'].unique())
    if Citizenship_q == 'Citizen':
        Citizenship =1
    else:
        Citizenship =0
    Sponsor_q=st.sidebar.selectbox("Sponsor", df1['Sponsor'].unique())
    if Sponsor_q == 'Yes':
        Sponsor =1
    else:
        Sponsor =0
    AlainResident_q=st.sidebar.selectbox("Resident of Al Ain", df1['AlainResident'].unique())
    if AlainResident_q == 'Yes':
        Resident_in_city =1
    else:
        Resident_in_city =0
    if(options1 =='Complete Attributes (Historic & Checkpoints)'):
        features =[AgeCourseStart,Quiz1Norm, Quiz2Norm, HW1Norm, MTNorm,
           AGE_ADMITTED, Reg_Hrs, HS_GPA, MATH, PHYS,
           Repeated_Grade_ITBP119_CSBP119, ITBP119_CSBP119_max,
           Repeated_Grade_ITBP219_CSBP219, ESPN, CSBP121, MATH105,
           CENG205, PHYS105, CENG202, CIT, AcademicStanding,
           Citizenship, Gender, Sponsor, Resident_in_city]

        feature_names = ['AgeCourseStart','Quiz1Norm', 'Quiz2Norm', 'HW1Norm', 'MTNorm', 
           'AGE_ADMITTED', 'Reg_Hrs', 'HS_GPA', 'MATH', 'PHYS',
           'Repeated_Grade_ITBP119_CSBP119', 'ITBP119_CSBP119_max',
           'Repeated_Grade_ITBP219_CSBP219', 'ESPN', 'CSBP121', 'MATH105',
           'CENG205', 'PHYS105', 'CENG202', 'CIT', 'AcademicStanding',
           'Citizenship', 'Gender', 'Sponsor', 'Resident_in_city']
    else:
        features =[AgeCourseStart,
        AGE_ADMITTED, Reg_Hrs, HS_GPA, MATH, PHYS,
        Repeated_Grade_ITBP119_CSBP119, ITBP119_CSBP119_max,
        Repeated_Grade_ITBP219_CSBP219, ESPN, CSBP121, MATH105,
        CENG205, PHYS105, CENG202, CIT, AcademicStanding,
        Citizenship, Gender, Sponsor, Resident_in_city]

        feature_names = ['AgeCourseStart',
        'AGE_ADMITTED', 'Reg_Hrs', 'HS_GPA', 'MATH', 'PHYS',
        'Repeated_Grade_ITBP119_CSBP119', 'ITBP119_CSBP119_max',
        'Repeated_Grade_ITBP219_CSBP219', 'ESPN', 'CSBP121', 'MATH105',
        'CENG205', 'PHYS105', 'CENG202', 'CIT', 'AcademicStanding',
        'Citizenship', 'Gender', 'Sponsor', 'Resident_in_city']

    final_features = np.array(features).reshape(1, -1)
    final_features_df=pd.DataFrame(final_features,columns=feature_names)
    excludedList=['CIT','AcademicStanding','Citizenship','Gender','Sponsor','Resident_in_city']

    if st.sidebar.button('Predict'):  
        prediction =  loaded_model.predict(final_features)
        studRisk='<p style="font-family:Courier; color:Black; font-size: small;font-weight: bold;">Total Grade would be '+str(float(prediction[0]))+'</p>'
        st.write(studRisk, unsafe_allow_html=True)
        st.write('Please see the student attributes below : ')
        for feat in feature_names:
            fmean=feat+"_mean"
            if(feat=="Repeated_Grade_ITBP219_CSBP219" or feat=="Repeated_Grade_ITBP119_CSBP119"):
                    if float(final_features_df[feat])>0:
                        studRisk='<p style="font-family:Courier; color:Black; font-size: small;">'+feat+': '+str(int(final_features_df[feat]))+'</p>'
                        st.write(studRisk, unsafe_allow_html=True)
            elif(feat in excludedList):
                    if int(final_features_df[feat])==1:
                        val="Yes"
                        if feat=="Gender":
                            val="Female"
                        if feat=="AcademicStanding":
                             val="Goodstanding"
                    else:
                        val="No"
                        if feat=="Gender":
                            val="Male"
                        if feat=="AcademicStanding":
                            val="Probation"
                    studRisk='<p style="font-family:Courier; color:Black; font-size: small;">'+feat+': '+val+'</p>'
                    st.write(studRisk, unsafe_allow_html=True)
            else:
                if float(final_features_df[feat]) < np.round(dict_mean[fmean],2):
                    studRisk='<p style="font-family:Courier; color:Black; font-size: small;">'+feat+' :'+str(float(final_features_df[feat]))+' < Avg :'+str(np.round(dict_mean[fmean],2))+'</p>'
                    st.write(studRisk, unsafe_allow_html=True)
                if int(final_features_df[feat]) > np.round(dict_mean[fmean],2):
                    studRisk='<p style="font-family:Courier; color:Black; font-size: small;">'+feat+' :'+str(float(final_features_df[feat]))+' > Avg :'+str(np.round(dict_mean[fmean],2))+'</p>'
                    st.write(studRisk, unsafe_allow_html=True)
                    
elif(options =='Batch Prediction'):
    
    
    if(options1 =='Complete Attributes (Historic & Checkpoints)'):
        link = '[Download .csv files](https://github.com/balqism/student-at-risk-prediction/blob/main/CombinedFeatures.csv)'
        st.markdown(link, unsafe_allow_html=True)
        key_features=['AgeCourseStart', 'Quiz1Norm', 'Quiz2Norm', 'HW1Norm', 'MTNorm',
           'AGE_ADMITTED', 'Reg_Hrs', 'HS_GPA', 'MATH', 'PHYS',
           'Repeated_Grade_ITBP119_CSBP119', 'ITBP119_CSBP119_max',
           'Repeated_Grade_ITBP219_CSBP219', 'ESPN', 'CSBP121', 'MATH105',
           'CENG205', 'PHYS105', 'CENG202', 'CIT', 'AcademicStanding',
           'Citizenship', 'Gender', 'Sponsor', 'Resident_in_city']
    else:
        link = '[Download .csv files](https://raw.githubusercontent.com/balqism/student-at-risk-prediction/main/HistoricAttributes.csv)'
        st.markdown(link, unsafe_allow_html=True)
        key_features=['AgeCourseStart',
           'AGE_ADMITTED', 'Reg_Hrs', 'HS_GPA', 'MATH', 'PHYS',
           'Repeated_Grade_ITBP119_CSBP119', 'ITBP119_CSBP119_max',
           'Repeated_Grade_ITBP219_CSBP219', 'ESPN', 'CSBP121', 'MATH105',
           'CENG205', 'PHYS105', 'CENG202', 'CIT', 'AcademicStanding',
           'Citizenship', 'Gender', 'Sponsor', 'Resident_in_city']
        
    lower_target=st.sidebar.slider("Lowerlevel Target%", 0.0, 1.0)  
    upper_target=st.sidebar.slider("Upperlevel Target%", 0.0, 1.0) 
    uploaded_file = st.sidebar.file_uploader("Choose your csv file (File format: .csv)",type=['csv'])
    if uploaded_file is not None:
        student_por_df = pd.read_csv(uploaded_file)
        outcome = 'TGNorm'
        features = [feat for feat in list(student_por_df) if feat not in outcome]
        test_dataset=student_por_df[features]
        if st.sidebar.button('Predict'):  
            prediction =  loaded_model.predict(test_dataset)
            preds_final=pd.DataFrame(prediction,columns=['TGNorm'])
            condition1=preds_final['TGNorm'] <=upper_target
            condition2=preds_final['TGNorm'] >=lower_target
            predicted_students_in_trouble=preds_final[condition1 & condition2]
            st.write("No. of students in the specified range: ",len(predicted_students_in_trouble))
            excludedList=['CIT','AcademicStanding','Citizenship','Gender','Sponsor','Resident_in_city']
            # See which feature they landed well below or well above peers
            for index, row in predicted_students_in_trouble.iterrows():
                studDet='<p style="font-family:Courier; color:Black; font-size: small;"><u>Student ID :'+str(index+1)+' Total Grade:'+str(np.round(row['TGNorm'],2))+'</u></p>'
                st.write(studDet,unsafe_allow_html=True)
                for feat in key_features:
                    fmean=feat+"_mean"
                    row_df=student_por_df[features][index:index+1]
                    if(feat=="Repeated_Grade_ITBP219_CSBP219" or feat=="Repeated_Grade_ITBP119_CSBP119"):
                        if float(row_df[feat])>0:
                            studRisk='<p style="font-family:Courier; color:Black; font-size: small;">'+feat+': '+str(int(row_df[feat]))+'</p>'
                            st.write(studRisk, unsafe_allow_html=True)
                    elif(feat in excludedList):
                        if int(row_df[feat])==1:
                            val="Yes"
                            if feat=="Gender":
                                val="Female"
                            if feat=="AcademicStanding":
                                 val="Goodstanding"
                        else:
                            val="No"
                            if feat=="Gender":
                                val="Male"
                            if feat=="AcademicStanding":
                                val="Probation"
                        studRisk='<p style="font-family:Courier; color:Black; font-size: small;">'+feat+': '+val+'</p>'
                        st.write(studRisk, unsafe_allow_html=True)
                    else:
                        if float(row_df[feat]) < np.round(dict_mean[fmean],2):
                            studRisk='<p style="font-family:Courier; color:Black; font-size: small;">'+feat+' :'+str(float(row_df[feat]))+' < Avg :'+str(np.round(np.mean(student_por_df[feat]),2))+'</p>'
                            st.write(studRisk, unsafe_allow_html=True)
                        if float(row_df[feat]) > np.round(dict_mean[fmean],2):
                            studRisk='<p style="font-family:Courier; color:Black; font-size: small;">'+feat+' :'+str(float(row_df[feat]))+' > Avg :'+str(np.round(np.mean(student_por_df[feat]),2))+'</p>'
                            st.write(studRisk, unsafe_allow_html=True)

    else:
        st.warning("you need to upload a csv file.")
else:
    pass    

    