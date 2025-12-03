"""
Combined Streamlit application integrated with data processing, fault detection, and LLM tool calling.
Provides a web-based GUI for uploading CSV files, viewing fault detection results via LLM decision (direct or LSTM tool),
and chatting with the Llama model for further inquiries.
Merged from API_tool.py and API_Llama_GUI_0_2.py for enhanced integration.
"""

# Importing Libraries and Setting Paths

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import pickle
import json
from openai import OpenAI
import json
import streamlit as st

LSTM_AE_path = r"path_to/Models/best_model.h5"
minmax_scaler_path = r"path_to/Extra files/minmax_scaler.pkl"
anomaly_threshold_path = r"path_to/Extra files/anomaly_threshold.txt"

best_model = load_model(LSTM_AE_path, custom_objects={'mse': mean_squared_error})

# Set white background and ensure visibility via custom CSS (light theme style)
st.markdown(
    """
    <style>
        .stApp {
            background-color: #FFFFFF;
            color: #000000; /* Black text for visibility */
        }
        .stButton > button {
            background-color: #F0F2F6; /* Light gray to match Browse Files button */
            color: #000000; /* Black text */
            border: 1px solid #cccccc; /* Light gray border to match */
            padding: 8px 12px; /* Consistent padding with Browse Files */
            font-size: 14px; /* Default font size */
            border-radius: 4px; /* Rounded corners to match */
        }
        .stButton > button:hover {
            background-color: #E0E4E8; /* Slightly darker gray on hover */
        }
        .stTextInput > div > div > input {
            background-color: #F0F2F6; /* Light gray input background */
            color: #000000; /* Black text in inputs */
            border: 1px solid #cccccc; /* Light border */
        }
        .stExpander > div > div > div > button {
            background-color: #F9F9F9; /* Off-white expander background */
            color: #000000; /* Black text */
            border: 1px solid #dddddd;
        }
        .stMarkdown {
            color: #000000; /* Black markdown text */
        }
        .stSpinner > div {
            color: #000000;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# tool use
def move_column(df, col_name, new_idx):
    """
    This function returns a new DataFrame where the column col_name is moved to position new_idx.
    The relative order of the other columns is preserved.
    """
    cols = list(df.columns)
    cols.insert(new_idx, cols.pop(cols.index(col_name)))
    return df[cols]

def data_preprocessing(data):
    """
    This function preprocesses the input data by adding a time column, handling missing values, 
    applying moving averages, and downsampling.
    """
    # Add a time column
    data['time_in_seconds'] = data.groupby(['unit', 'cycle']).cumcount()
    
    # Handle missing values and apply rolling mean
    sensor_features = ['T24', 'T30', 'T48', 'T50', 'P15', 'P24', 'Ps30', 'P40', 'P50', 'Nc', 'Nf', 'Wf',
                       'T40', 'P30', 'P45', 'W21', 'W22', 'W25', 'W31', 'W32', 'W48', 'W50', 'SmFan', 'SmLPC', 'SmHPC', 'phi']
    window_size = 10  # 10-second moving average

    for feature in sensor_features:
        if feature in data.columns:
            # Handle missing values
            if data[feature].isnull().any():
                data[feature] = (
                    data
                    .groupby(['unit', 'cycle'])[feature]
                    .transform(lambda x: x.fillna(x.rolling(window=window_size, min_periods=1).mean()))
                )
            # Apply rolling mean
            data[feature] = (
                data
                .groupby(['unit', 'cycle'])[feature]
                .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
            )
    
    # Downsample the data
    seconds = 60
    data['time_bin'] = data.groupby(['unit', 'cycle'])['time_in_seconds'].transform(lambda x: (x // seconds).astype(int) + 1)
    downsampled_data = []
    for (unit, cycle, time_bin), group in data.groupby(['unit', 'cycle', 'time_bin']):
        sensor_means = group[sensor_features].mean()
        non_sensor_cols = [col for col in data.columns if col not in sensor_features + ['time_in_seconds', 'time_bin']]
        non_sensor_values = group[non_sensor_cols].iloc[-1]
        row = pd.concat([non_sensor_values, sensor_means])
        row['time_in_seconds'] = (time_bin - 1) * seconds
        downsampled_data.append(row)
    data_downsampled = pd.DataFrame(downsampled_data)
    original_cols = [col for col in data.columns if col != 'time_bin']
    data_downsampled = data_downsampled[original_cols]
    data = data_downsampled
    data['Time_Downsample'] = data.groupby(['unit', 'cycle']).cumcount()
    data = data.drop(columns=['time_in_seconds'])
    data = move_column(data, "Time_Downsample", 3)
    
    # Min-max scaling
    feature_cols = data.columns[4:]
    with open(minmax_scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    data_mm = scaler.transform(data[feature_cols])
    data_mm = pd.DataFrame(data_mm, columns=feature_cols, index=data.index)
    for col in data_mm:
        data[col] = data_mm[col]
    
    data = data.drop('hs', axis=1, errors='ignore')  # Drop 'hs' if it exists

    return data

def detect_fault(data):
    """
    This function detects faults in the input data using a trained LSTM AutoEncoder model.
    It loads the model, preprocesses the data, and returns a binary fault prediction.
    """
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=1)  # Shape: (rows, 1, cols)
    prediction = best_model.predict(data, verbose=0)
    mse = np.mean((data.squeeze() - prediction.squeeze())**2, axis=1)
    with open(anomaly_threshold_path, 'r') as f:
        threshold = float(f.read().strip())
    fault = (mse > threshold).astype(int)
    return fault, mse

def results(data_input):
    """
    This function processes the input data, detects faults, and returns a dictionary with the results.
    """

    processed_data = data_preprocessing(data_input)
    faults = detect_fault(processed_data)[0]
    last_fault = faults[-1]

    pred = []
    if last_fault == 1:
        if data_input['HPT_eff_mod'].iloc[-1] != 0:
            pred.append([1, 'HPT'])
        if data_input['LPT_eff_mod'].iloc[-1] != 0 or data_input['LPT_flow_mod'].iloc[-1] != 0:
            pred.append([1, 'LPT'])
    else:
        pred.append([0, 'No fault'])
    
    result_dict = {
        'unit': int(data_input['unit'].iloc[-1]),
        'cycle': int(data_input['cycle'].iloc[-1]),
        'fault': pred
    }
    return json.dumps(result_dict)

# prompt
def create_last_row_text(data_df):
    row = data_df.iloc[-1]
    time = row['Time_Downsample']
    sensors = list(row.index)[4:]  # sensor columns start from column 5 (index 4)
    sensors_str = ', '.join([f"sensor{j+1} {sensors[j]}: {row[sensors[j]]:.2f}" for j in range(len(sensors))])
    not_sensors = list(row.index)[:4]
    not_sensors_str = ', '.join([f"{not_sensors[j]}: {row[not_sensors[j]]:.2f}" for j in range(4)])
    
    # Compute anomaly score for the last row
    score = detect_fault(data_df)[1][-1]
    
    # Prompt text
    prompt = f"Analyze NASA C-MAPSS engine data after downsampling to minutes and minmax normalization. Elapsed time since flight start: {time} minutes. Not sensor measurements: {not_sensors_str}. Sensor measurements: {sensors_str}. LSTM anomaly score: {score:.4f}. Identify health state and possible faults in HPT or LPT."
    
    return prompt

# --- Streamlit GUI  ---

st.title("Anomaly Detection")

st.write("Upload a CSV file containing sensor data to analyze for faults.")

# Guidance for large uploads
st.info("Note: Streamlit's default upload limit is 200MB. To upload larger files, run the app with 'streamlit run app.py --server.maxUploadSize=1000' for 1GB limit, or adjust in .streamlit/config.toml.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

tool_result = None
summary = None

if uploaded_file is not None:
    try:
        # Read in chunks for large files
        chunks = pd.read_csv(uploaded_file, chunksize=100000, low_memory=False)
        data_input_df = pd.concat(chunks, ignore_index=True)
    except Exception as e:
        st.error(f"Failed to read CSV file: {str(e)}")
    else:
        if st.button("Analyze Data"):
            with st.spinner("Analyzing..."):
                try:
                    tool_result = results(data_input_df)
                    data_json = data_input_df.to_json(orient='records')

                    processed_data = data_preprocessing(data_input_df)
                    last_row_text = create_last_row_text(processed_data)
                    
                    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
                    model = "llama_3_8b_fine_tuned"
                    
                    # Define the tool for LSTM prediction
                    tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_lstm_fault_prediction",
                                "description": "Use the LSTM AE model to analyze the sensor data and predict faults.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                }
                            }
                        }
                    ]
                    
                    # Initial messages
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a fault detection assistant. You can either predict faults yourself based on the provided data or use the 'get_lstm_fault_prediction' tool to get an analysis from the LSTM model. Summarize the faults in your final response."
                        },
                        {
                            "role": "user",
                            "content": f"Analyze this sensor data for faults: {last_row_text}"
                        }
                    ]
                    
                    # First call to the model
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto"
                    )
                    
                    response_message = response.choices[0].message

                    # Check if the model wants to call a tool
                    if response_message.tool_calls:
                        # Append the assistant's message with tool calls
                        messages.append(response_message)
                        
                        # Process each tool call
                        for tool_call in response_message.tool_calls:
                            if tool_call.function.name == "get_lstm_fault_prediction":
                                try:
                                    tool_response = results(data_input_df)
                                except Exception as e:
                                    tool_response = json.dumps({"error": str(e)})
                                # Append the tool response
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call.id,
                                        "name": tool_call.function.name,
                                        "content": tool_response
                                    }
                                )
                        
                        # Second call to the model with tool response
                        second_response = client.chat.completions.create(
                            model=model,
                            messages=messages
                        )
                        final_content = second_response.choices[0].message.content
                    else:
                        # No tool call, use the direct response
                        final_content = response_message.content
                    
                    summary = final_content
                    st.subheader("Analysis Result")
                    st.text_area("Summary", summary, height=200)
                    
                    st.subheader("Explanation")
                    st.markdown("""
                    - **Fault Detection**: Uses a trained LSTM AutoEncoder to identify anomalies in sensor data.
                    - **Preprocessing**: Includes downsampling, scaling, and handling missing values.
                    - **Results**: Based on the last detected fault, with details on HPT/LPT if applicable.
                    """)
                except ValueError as ve:
                    st.error(f"Data processing error: {str(ve)}")
                except Exception as ex:
                    st.error(f"Unexpected error during analysis: {str(ex)}")
