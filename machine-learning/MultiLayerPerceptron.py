import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import altair as alt
import time

# Page title
st.set_page_config(page_title='ExoplanetML', page_icon=':alien:')
st.title(':alien: Multi-Layer Perceptron (MLP) Regression')

with st.expander('About this app'):
    st.markdown('**What can this app do?**')
    st.info('This app trains an MLPRegressor (Multi-Layer Perceptron) to predict a target variable in exoplanet datasets.')
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
    Here's a useful tool for data curation [CSV only]: <a href="https://aivigoratemitotool.streamlit.app/" target="_blank">AI-powered Data Curation Tool</a>. Tip: Ensure that your CSV file doesn't have any NaNs.
    </div>
    <br>
    """, unsafe_allow_html=True)

    st.markdown('**How to use the app?**')
    st.warning('Upload a dataset, configure MLP settings in the sidebar, train the model, and analyze performance.')

with st.sidebar:
    st.header('1. Input data')

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

    st.markdown('**Use example data**')
    example_data = st.toggle('PHL Habitable Worlds Catalog (HWC)')
    if example_data:
        df = pd.read_csv('https://drive.google.com/uc?export=download&id=1J1f_qSHCYdfqiqQtpoda_Km_VmQuI4UX')

    st.header('2. MLP Parameters')
    parameter_split_size = st.slider('Train/test split %', 10, 90, 80, 5)
    hidden_layer_sizes = st.text_input('Hidden layers (comma-separated)', value='64,128,64')
    max_iter = st.slider('Max iterations', 100, 2000, 500, step=100)
    alpha = st.number_input('L2 penalty (alpha)', min_value=0.0001, max_value=1.0, value=0.0001, step=0.0001, format="%f")
    learning_rate = st.selectbox('Learning rate', ['constant', 'invscaling', 'adaptive'])
    sleep_time = st.slider('Sleep time (seconds)', 0, 3, 0)

if uploaded_file or example_data:
    with st.status("Running MLP training...", expanded=True) as status:
        st.write("Preparing data ...")
        time.sleep(sleep_time)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        st.write("Splitting dataset ...")
        time.sleep(sleep_time)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - parameter_split_size) / 100, random_state=42)

        st.write("Normalizing features ...")
        time.sleep(sleep_time)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.write("Training MLP model ...")
        time.sleep(sleep_time)
        hidden_layers = tuple(int(x) for x in hidden_layer_sizes.split(','))
        model = MLPRegressor(hidden_layer_sizes=hidden_layers,
                             max_iter=max_iter,
                             alpha=alpha,
                             learning_rate=learning_rate,
                             early_stopping=True,
                             n_iter_no_change=20,
                             validation_fraction=0.1,
                             random_state=42)
        model.fit(X_train_scaled, y_train)

        st.write("Generating predictions ...")
        time.sleep(sleep_time)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        result_df = pd.DataFrame({
            'Method': ['MLP Regressor'],
            'Training MSE': [train_mse],
            'Training R2': [train_r2],
            'Test MSE': [test_mse],
            'Test R2': [test_r2]
        }).round(3)

    status.update(label="Complete", state="complete", expanded=False)

    st.header('Input data overview', divider='rainbow')
    col = st.columns(4)
    col[0].metric("Samples", X.shape[0])
    col[1].metric("Features", X.shape[1])
    col[2].metric("Train samples", X_train.shape[0])
    col[3].metric("Test samples", X_test.shape[0])

    st.dataframe(df, height=210, use_container_width=True)

    st.header('Model Performance', divider='rainbow')
    st.dataframe(result_df.T.reset_index().rename(columns={'index': 'Metric', 0: 'Value'}))

    st.header('Prediction Results', divider='rainbow')
    pred_df = pd.DataFrame({
        'actual': pd.concat([y_train, y_test]).reset_index(drop=True),
        'predicted': np.concatenate([y_train_pred, y_test_pred]),
        'class': ['train'] * len(y_train) + ['test'] * len(y_test)
    })

    col_pred = st.columns((2, 0.2, 3))
    with col_pred[0]:
        st.dataframe(pred_df, height=300)

    with col_pred[2]:
        scatter = alt.Chart(pred_df).mark_circle(size=60).encode(
            x='actual',
            y='predicted',
            color='class'
        )
        st.altair_chart(scatter, theme='streamlit', use_container_width=True)

    model_filename = 'mlp_model.joblib'
    joblib.dump(model, model_filename)
    joblib.dump(scaler, 'mlp_scaler.joblib')

    with open(model_filename, 'rb') as f:
        st.download_button(
            label='Download Trained Model',
            data=f,
            file_name=model_filename,
            mime='application/octet-stream'
        )

    st.header('Apply Trained Model')
    new_file = st.file_uploader("Upload CSV for Prediction", type=['csv'], key='predict')
    if new_file is not None:
        new_data = pd.read_csv(new_file)
        with open(model_filename, 'rb') as f:
            saved_model = joblib.load(f)
        with open('mlp_scaler.joblib', 'rb') as f:
            loaded_scaler = joblib.load(f)
        required_features = X.columns.tolist()
        missing = set(required_features).difference(new_data.columns)
        if not missing:
            new_X = new_data[required_features]
            new_X_scaled = loaded_scaler.transform(new_X)
            pred = saved_model.predict(new_X_scaled)
            new_data['Predictions'] = pred
            st.write(new_data.head())
            pred_csv = convert_df(new_data)
            st.download_button(
                label='Download Predictions',
                data=pred_csv,
                file_name='mlp_predictions.csv',
                mime='text/csv'
            )
        else:
            st.error("Missing features: " + ", ".join(missing))
else:
    st.warning('👈 Upload a CSV file or enable example data to get started.')
