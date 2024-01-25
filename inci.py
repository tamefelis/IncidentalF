import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re

# Function to convert columns to datetime and clean text
def preprocess_data(df):
    # Keep the original 'clinical parameters' data
    df['Original Clinical Parameters'] = df['clinical parameters']
    df['Processed Clinical Parameters'] = df['clinical parameters'].apply(lambda x: str(x).split(',') if pd.notna(x) else [])
    # Create 'Findings_List' column for incidental findings
    df['Findings_List'] = df['clinical parameters'].apply(lambda x: str(x).split(',') if pd.notna(x) else [])
    # Preprocess for a new variable
    df['Processed Clinical Parameters'] = df['clinical parameters'].apply(lambda x: str(x).split(',') if pd.notna(x) else [])
    df['contacting participation by PI verification?'] = df['contacting participation by PI verification?'].apply(clean_text)
    df['severity'] = df['severity'].apply(clean_text)
    return df

def clean_text(text):
    if pd.isna(text):  # Check if the value is NaN
        return ""  # Return an empty string for NaN values
    text = str(text)  # Convert to string in case it's not
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)  # Replace non-alphabetic characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces
    return text.strip()

# Data Loading Function
@st.cache_data
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file, sheet_name='VISIT')
    return preprocess_data(df)

def create_longitudinal_view(data):
    # Assuming 'Visit' and 'Study ID' are columns in your data
    max_visits = data['Visit_Type'].nunique()
    transformed_data = pd.DataFrame()
    for i in range(1, max_visits + 1):
        visit_data = data[data['Visit_Type'] == f'Visit {i}']
        visit_data = visit_data[['Study ID', 'clinical parameters']].rename(columns={'clinical parameters': f'Clinical Parameters Visit {i}'})
        if i == 1:
            transformed_data = visit_data
        else:
            transformed_data = pd.merge(transformed_data, visit_data, on='Study ID', how='outer')
    return transformed_data

def create_longitudinal_view(data):
    # Create a pivot table with 'Study ID' as index and 'Visit_Type' as columns
    transformed_data = data.pivot_table(index='Study ID', columns='Visit_Type', values='clinical parameters', aggfunc='first')
    return transformed_data


# Streamlit Layout
st.title('Participant Incidental Findings Dashboard')

# Upload File
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type="xlsx")

if uploaded_file is not None:
    data = load_data(uploaded_file)

    # Assuming 'Date' column is already in the correct format
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y', errors='coerce')

    # Date Range Filter
    min_date = data['Date'].min()
    max_date = data['Date'].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    end_date = end_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Adjust to include the end date fully
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
     # Ensure that 'Findings_List' is created in the filtered_data
    filtered_data = preprocess_data(filtered_data)
    # Filters for PI Verification and Severity
    pi_verification = st.sidebar.multiselect('Filter by PI Verification', options=filtered_data['contacting participation by PI verification?'].unique())
    severity = st.sidebar.multiselect('Filter by Severity', options=filtered_data['severity'].unique())

    if pi_verification:
        filtered_data = filtered_data[filtered_data['contacting participation by PI verification?'].isin(pi_verification)]
    if severity:
        filtered_data = filtered_data[filtered_data['severity'].isin(severity)]

    # Prepare Data for Treemap
    all_findings = filtered_data.explode('Findings_List')
    all_findings['Findings_List'] = all_findings['Findings_List'].str.strip().str.lower()
    filtered_findings = all_findings[all_findings['Findings_List'] != 'nil']
    filtered_findings['Category'] = 'Incidental Findings'
    treemap_data = filtered_findings.groupby(['Category', 'Findings_List']).size().reset_index(name='Counts')

    formatted_start_date = start_date.strftime('%Y-%m-%d')
    formatted_end_date = end_date.strftime('%Y-%m-%d')
    # Calculate the total number of visits that are not NIL and NIL
    total_visits = len(filtered_data)
    non_nil_visits = filtered_data['clinical parameters'].apply(lambda x: str(x).strip().lower() != 'nil').sum()
    nil_visits = total_visits - non_nil_visits
    with st.expander("filtered_data"):
        st.write(filtered_data, "this is a dataframe of all the incidental findings from filtered date range")
    findings_summary = data['clinical parameters'].apply(lambda x: 'NIL' if str(x).strip().lower() == 'nil' else 'Not NIL').value_counts()
    st.subheader('Summary of Incidental Findings ')
    st.write(f'Data from {formatted_start_date} to {formatted_end_date}' )
    findings_summary = filtered_data['clinical parameters'].apply(lambda x: 'NIL' if str(x).strip().lower() == 'nil' else 'Not NIL').value_counts()    
    
    formatted_start_date = start_date.strftime('%Y-%m-%d')
    formatted_end_date = end_date.strftime('%Y-%m-%d')
    # Create a bar chart for the summary
    fig_summary = px.bar(findings_summary, x=findings_summary.index, y=findings_summary.values, title='Incidental Findings Summary')
     # Create a bar chart for the summary with aesthetic enhancements
    fig_summary = px.bar(
        findings_summary, 
        x=findings_summary.index, 
        y=findings_summary.values, 
        title=f'No. of records that has asterisk sign flagged in the blood profile from {formatted_start_date} to {formatted_end_date}',
        color=findings_summary.index,
        text=findings_summary.values,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    # Customize layout
    fig_summary.update_layout(
        xaxis_title='Finding Type',
        yaxis_title='Count',
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        showlegend=False
    )

    # Customize bar labels
    fig_summary.update_traces(
        texttemplate='%{text}', 
        textposition='outside',
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        opacity=0.6
    )
    st.plotly_chart(fig_summary)
    
    # Prepare Data for Treemap and Summary
    all_findings = filtered_data.explode('Findings_List')
    all_findings['Findings_List'] = all_findings['Findings_List'].str.strip().str.lower()
    filtered_findings = all_findings[all_findings['Findings_List'] != 'nil']
    
    # Count the frequency of each clinical parameter
    parameter_frequencies = filtered_findings['Findings_List'].value_counts()

    # Get the most frequent clinical parameter and its count
    most_frequent_parameter = parameter_frequencies.idxmax()
    most_frequent_count = parameter_frequencies.max()
    # Write the legend and summary
    
    # Custom HEX color scale
    hex_light = ["#326193", "#2F91E5", "#61B9FF", "#B2D5FF", "#E2E2E2", 
                 "#FFC6A7", "#FF9D55", "#CE7729", "#87512B"]

       # Create the Treemap
    fig1 = px.treemap(
        treemap_data, 
        path=['Category', 'Findings_List'], 
        values='Counts', 
        title=f'Categorical Type of Incidental Findings from {formatted_start_date} to {formatted_end_date}', 
        color='Counts', 
        color_continuous_scale=hex_light

    )
    st.plotly_chart(fig1)
    st.caption(f"Out of {total_visits} total visits from {formatted_start_date} to {formatted_end_date}, {non_nil_visits} are not 'NIL' and {nil_visits} are 'NIL'.")
    st.caption(f"The most frequent clinical parameter across all visits is **'{most_frequent_parameter}'** "
             f"with a total occurrence of {most_frequent_count} times.")

    
    # Prepare and count individual clinical parameters
    split_params = data['clinical parameters'].str.split(',').explode()
    split_params = split_params.str.strip().str.lower()
    param_counts = split_params.value_counts().reset_index()
    param_counts.columns = ['Clinical Parameter', 'Frequency']
    # Create a Bar Chart for parameter frequencies
    fig_bar = px.bar(param_counts, x='Clinical Parameter', y='Frequency', title='Frequency of Clinical Parameters')
    #st.plotly_chart(fig_bar)

    # Create a Pie Chart for parameter frequencies
    fig_pie = px.pie(param_counts, values='Frequency', names='Clinical Parameter', title='Distribution of Clinical Parameters in Pie Chart')
    #st.plotly_chart(fig_pie)
    # Create the Treemap for Clinical Parameters
    if 'Processed Clinical Parameters' in data.columns:  # Check if the column exists
        clinical_params = data['Processed Clinical Parameters'].explode()
        clinical_params = clinical_params[clinical_params.str.strip().str.lower() != 'nil']
        clinical_params_counts = clinical_params.value_counts().reset_index()
        clinical_params_counts.columns = ['Clinical Parameter', 'Counts']
        fig_treemap = px.treemap(clinical_params_counts, path=['Clinical Parameter'], values='Counts', title='Distribution of Clinical Parameters in Treemap')
        #st.plotly_chart(fig_treemap)
    else:
        st.write("Processed Clinical Parameters column not found in the data.")
    
    
    st.markdown("***")

    st.subheader("Longitudinal view on specific subject")
    # Generate Longitudinal View DataFrame
    longitudinal_data = create_longitudinal_view(data)

    # Input widget for selecting a Study ID
    study_id = st.selectbox("Please, Select a Study ID that you would like to view specificly", options=longitudinal_data.index.unique())

    # Filter data for the selected Study ID
    selected_participant_data = longitudinal_data.loc[[study_id]].dropna(axis=1, how='all')  # Drop columns if all values are NaN

    # Display data for the selected Study ID
    if not selected_participant_data.empty:
        st.write(f"Clinical Parameters for Study ID: *{study_id}*")
        st.dataframe(selected_participant_data)
        
        # Prepare data for bar chart
        # Count non-'NIL' findings per visit
        counts_per_visit = selected_participant_data.applymap(
            lambda x: 0 if isinstance(x, float) and np.isnan(x) or str(x).strip().lower() == 'nil' else len(str(x).split(','))
        ).iloc[0]
        
        # Prepare hover data for bar chart
        hover_data = selected_participant_data.applymap(lambda x: '' if isinstance(x, float) and np.isnan(x) or str(x).strip().lower() == 'nil' else x).iloc[0]
        
        # Create the bar chart
        fig = px.bar(
        counts_per_visit,
        x=counts_per_visit.index,
        y=counts_per_visit.values,
        title=f'Number of Clinical Parameters per Visit for Study ID: {study_id}',
        labels={'y': 'Number of Parameters', 'index': 'Visit'},
        text=counts_per_visit.values,
        hover_data=[hover_data]# Display counts as text on the bars
        )

        # Customize the bar chart's aesthetics
        fig.update_layout(
            yaxis=dict(
                range=[0, max(counts_per_visit.values) + 1],  # Set the y-axis range dynamically based on the data
                dtick=1,  # Set the interval between tick marks as 1 to avoid decimal points
                showgrid=True,  # Show gridlines for better readability
            ),
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        )

        # Update the bar color, border color, and hover info
        fig.update_traces(
            marker_color='rgb(55, 83, 109)',  # Set a specific color for all bars
            # Other trace customizations...
            texttemplate='%{text:.0f}',  # No decimal places in the bar labels
            textposition='outside',  # Display text outside of bars for clarity
        )
        st.plotly_chart(fig)

        
    else:
        st.write("No data available for the selected Study ID")
    
else:
    st.write("Please upload a dataset to begin.")
