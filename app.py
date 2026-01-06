import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge , Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score

#PAGE CONFIGURE
st.set_page_config(
    page_title='Avocado Price Prediction',
    layout='wide'
)
st.title('Avocado Price Prediction Using Machine Learning')
st.markdown(
    """
    After uploading the Avocado Dataset the will automatically work and generate:
    - EDA
    - Price Prediction
    - Regression Model Comparison
    - Best Model Selection
    - Business Insights
    """
)

upload_file = st.file_uploader('Upload Avocado File Here', type=['csv'])

if upload_file is None:
    st.warning('Upload the appropriate Avocado Dataset to continue')
    st.stop()
    
df = pd.read_csv(upload_file)
    
# SIDE-BAR NAVIGATION
    
st.sidebar.title('NAVIGATION')
page = st.sidebar.radio(
    'Select Section',
    ['EDA Analysis','Price Prediction','Model Comparison','Insights']
)
    
# EDA Section
if page == 'EDA Analysis':
    st.header('Exploratory Analysis')
        
    st.subheader('Dataset Review')
    st.dataframe(df.head())
        
    st.subheader('Dataset Shape')
    st.write(f'Rows:{df.shape[0]},columns : {df.shape[1]}')
        
    st.subheader('Missing Values')
    st.write(df.isnull().sum())
        
    st.subheader('Statistical Summary')
    st.dataframe(df.describe())
        
    col1,col2 = st.columns(2)
        
    with col1:
        st.subheader('Average Price Distribution')
        fig, ax = plt.subplots()
        sns.histplot(df['AveragePrice'], kde=True ,ax=ax)
        st.pyplot(fig)
            
    with col2:
        st.subheader('Average Price by Type')
        fig ,ax = plt.subplots()
        sns.boxplot(x='type',y='AveragePrice', data=df ,ax=ax)
        st.pyplot(fig)
            
    st.subheader('Average Price Trend Over Years')
    fig, ax = plt.subplots(figsize=(10,4))
    df.groupby('year')['AveragePrice'].mean().plot(ax=ax)
    st.pyplot(fig)
        
# PRICE PREDICTION

elif page == 'Price Prediction':
    st.header('Price Prediction using Linear Regression')
        
    data = df.copy()
    data = pd.get_dummies(data , columns=['type','region'],drop_first=True)
        
    x = data.drop(['AveragePrice','Date'],axis=1)
    y = data['AveragePrice']
    
    x_train, x_test, y_train, y_test = train_test_split( x, y,test_size=0.2, random_state=42)
        
    model = LinearRegression()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
        
    st.subheader('Prediction Sample')
    comparison = pd.DataFrame({
         'Actual Price' : y_test.values,
        'Predicted Price' :y_pred
    })
    st.dataframe(comparison.head(10))
        
    st.subheader('Model Performance')
    st.metric('MAE',round(mean_absolute_error(y_test,y_pred),3))
    st.metric('MSE', round(mean_squared_error(y_test,y_pred),3))
    st.metric('R2_score', round(r2_score(y_test,y_pred),3)) 
        
# MODEL COMPARISON
elif page =='Model Comparison':
    st.header('Regression Model Comparison')
        
    data = df.copy()
    data = pd.get_dummies(data , columns=['type','region'],drop_first=True)
        
    x = data.drop(['AveragePrice','Date'],axis=1)
    y = data['AveragePrice']
        
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
        
    models ={
        'Linear Regression' : LinearRegression(),
        'Ridge Regression' : Ridge(),
        'Lasso Regression' : Lasso(),
        'Decision Tree' : DecisionTreeRegressor(random_state=42),
        'Random Forest' : RandomForestRegressor(n_estimators=100,random_state=42) 
    }
    results=[]
    for name, model in models.items():
        model.fit(x_train,y_train)
        pred = model.predict(x_test)
            
        results.append({
            'Model': name,
            'MAE' : mean_absolute_error(y_test,pred),
            'MSE' : mean_squared_error(y_test,pred),
            'R2 Score' : r2_score(y_test,pred)
        })
        
    results_df = pd.DataFrame(results).sort_values(by='R2 Score',ascending=False)
    
    st.subheader('Model Comparison Table')
    st.dataframe(results_df)
    
    
    best_model = results_df.iloc[0]['Model']
    st.success(f'Best Performance Model : **{best_model}**')
    
#INSIGHT
elif page == 'Insights':
    st.header('Analysis & Insights')
    st.markdown(
    """ #Key Observations
- Avocado prices vary significantly by **region** and **type**
- Yearly price trends show gradual changes over time
- Volume-related features strongly influence pricing

#Model Performance
- Linear models perform well for simple relationships
- Tree-based models handle non-linear patterns better

#Best Model
- **Random Forest Regressor** consistently provides:
  - Highest R² Score
  - Lowest prediction errors
#Business Insights
- Helps suppliers forecast avocado prices
- Useful for demand planning & regional pricing
- Demonstrates full ML lifecycle from EDA to deployment
#Conclusion
This project showcases a **complete data science pipeline**
with real-world pricing prediction using machine learning.
    """
    )
#FOOTER
st.markdown('------------------------------------------------')
st.markdown('Developed by Neha Yenuganti')
    
        
    

        
            
        