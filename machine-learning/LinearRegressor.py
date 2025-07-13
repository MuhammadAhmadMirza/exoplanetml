import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile

# Page title
st.set_page_config(page_title='ExoplanetML', page_icon=':alien:')
st.title(':alien: Linear Regression')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app allows users to build a machine learning (ML) model for Exoplanet target variable prediction using Linear Regression.')
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
    Here's a useful tool for data curation [CSV only]: <a href="https://aivigoratemitotool.streamlit.app/" target="_blank">AI-powered Data Curation Tool</a>. Tip: Ensure that your CSV file doesn't have any NaNs.
    </div>
    <br>
    """, unsafe_allow_html=True)

    st.markdown('**How to use the app?**')
    st.warning('To work with the app, go to the sidebar and select a dataset. Adjust the model parameters, run the model, and evaluate its performance.')

# Sidebar for input
with st.sidebar:
    st.header('1. Input data')

    st.markdown('**1.1 Use custom data**')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False)

    @st.cache_data
    def convert_df(input_df):
        return input_df.to_csv(index=False).encode('utf-8')

    example_csv = pd.read_csv('https://drive.google.com/uc?export=download&id=1J1f_qSHCYdfqiqQtpoda_Km_VmQuI4UX')
    csv = convert_df(example_csv)
    st.download_button(
        label="Download example CSV",
        data=csv,
        file_name='hwc-pesi.csv',
        mime='text/csv',
    )

    st.markdown('**1.2. Use example data**')
    example_data = st.toggle('PHL Habitable Worlds Catalog (HWC)')
    if example_data:
        df = pd.read_csv('https://drive.google.com/uc?export=download&id=1J1f_qSHCYdfqiqQtpoda_Km_VmQuI4UX')

    st.header('2. Set Parameters')
    parameter_split_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    sleep_time = st.slider('Sleep time', 0, 3, 0)

# Model building process
if uploaded_file or example_data:
    with st.status("Running ...", expanded=True) as status:

        st.write("Loading data ...")
        time.sleep(sleep_time)

        st.write("Preparing data ...")
        time.sleep(sleep_time)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        st.write("Splitting data ...")
        time.sleep(sleep_time)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - parameter_split_size) / 100, random_state=42)

        st.write("Model training ...")
        time.sleep(sleep_time)
        model = LinearRegression()
        model.fit(X_train, y_train)

        st.write("Applying model to make predictions ...")
        time.sleep(sleep_time)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        st.write("Evaluating performance metrics ...")
        time.sleep(sleep_time)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        model_results = pd.DataFrame({
            'Method': ['Linear Regression'],
            'Training MSE': [train_mse],
            'Training R2': [train_r2],
            'Test MSE': [test_mse],
            'Test R2': [test_r2]
        }).round(3)

    status.update(label="Status", state="complete", expanded=False)

    # Display data info
    st.header('Input data', divider='rainbow')
    col = st.columns(4)
    col[0].metric(label="No. of samples", value=X.shape[0], delta="")
    col[1].metric(label="No. of X variables", value=X.shape[1], delta="")
    col[2].metric(label="No. of Training samples", value=X_train.shape[0], delta="")
    col[3].metric(label="No. of Test samples", value=X_test.shape[0], delta="")

    with st.expander('Initial dataset', expanded=True):
        st.dataframe(df, height=210, use_container_width=True)
    with st.expander('Train split', expanded=False):
        train_col = st.columns((3, 1))
        with train_col[0]:
            st.markdown('**X**')
            st.dataframe(X_train, height=210, hide_index=True, use_container_width=True)
        with train_col[1]:
            st.markdown('**y**')
            st.dataframe(y_train, height=210, hide_index=True, use_container_width=True)
    with st.expander('Test split', expanded=False):
        test_col = st.columns((3, 1))
        with test_col[0]:
            st.markdown('**X**')
            st.dataframe(X_test, height=210, hide_index=True, use_container_width=True)
        with test_col[1]:
            st.markdown('**y**')
            st.dataframe(y_test, height=210, hide_index=True, use_container_width=True)

    # Coefficients as importance substitute
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    bars = alt.Chart(coef_df).mark_bar(size=40).encode(
        x='Coefficient:Q',
        y=alt.Y('Feature:N', sort='-x')
    ).properties(height=250)

    performance_col = st.columns((2, 0.2, 3))
    with performance_col[0]:
        st.header('Model performance', divider='rainbow')
        st.dataframe(model_results.T.reset_index().rename(columns={'index': 'Parameter', 0: 'Value'}))
    with performance_col[2]:
        st.header('Feature coefficients', divider='rainbow')
        st.altair_chart(bars, theme='streamlit', use_container_width=True)

    # Prediction results
    st.header('Prediction results', divider='rainbow')
    s_y_train = pd.Series(y_train, name='actual').reset_index(drop=True)
    s_y_train_pred = pd.Series(y_train_pred, name='predicted').reset_index(drop=True)
    df_train = pd.DataFrame(data=[s_y_train, s_y_train_pred], index=None).T
    df_train['class'] = 'train'

    s_y_test = pd.Series(y_test, name='actual').reset_index(drop=True)
    s_y_test_pred = pd.Series(y_test_pred, name='predicted').reset_index(drop=True)
    df_test = pd.DataFrame(data=[s_y_test, s_y_test_pred], index=None).T
    df_test['class'] = 'test'

    df_prediction = pd.concat([df_train, df_test], axis=0)

    prediction_col = st.columns((2, 0.2, 3))

    with prediction_col[0]:
        st.dataframe(df_prediction, height=320, use_container_width=True)

    with prediction_col[2]:
        scatter = alt.Chart(df_prediction).mark_circle(size=60).encode(
            x='actual',
            y='predicted',
            color='class'
        )
        st.altair_chart(scatter, theme='streamlit', use_container_width=True)

    # Save trained model
    model_filename = 'linear_model.joblib'
    joblib.dump(model, model_filename)

    with open(model_filename, 'rb') as f:
        st.download_button(
            label='Download Trained Model',
            data=f,
            file_name=model_filename,
            mime='application/octet-stream'
        )

    # Apply to new dataset
    st.header('Apply Trained Model to New Dataset')
    new_file = st.file_uploader("Upload a new CSV for prediction", type=["csv"], key='predict')

    if new_file is not None:
        new_data = pd.read_csv(new_file)

        with open(model_filename, 'rb') as f:
            saved_model = joblib.load(f)

        model_features = saved_model.feature_names_in_
        missing_features = set(model_features).difference(new_data.columns)

        if len(missing_features) == 0:
            new_X = new_data[model_features]
            predictions = saved_model.predict(new_X)
            new_data['Predictions'] = predictions
            st.write(new_data.head())

            csv_pred = convert_df(new_data)
            st.download_button(
                label="Download Predictions",
                data=csv_pred,
                file_name='predictions.csv',
                mime='text/csv'
            )
        else:
            st.error("The dataset is missing the following features: " + ", ".join(missing_features))
else:
    st.warning('👈 Upload a CSV file or click *"Load example data"* to get started!')
