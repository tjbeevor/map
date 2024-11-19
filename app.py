import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
import numpy as np
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="SA Crime Statistics Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pre-defined coordinates for major South Australian suburbs
SUBURB_COORDINATES = {
    "ADELAIDE": (-34.9285, 138.6007),
    "NORTH ADELAIDE": (-34.9066, 138.5944),
    "WEST ADELAIDE": (-34.9285, 138.5607),
    "PORT ADELAIDE": (-34.8474, 138.5079),
    "GLENELG": (-34.9820, 138.5160),
    "MODBURY": (-34.8329, 138.6834),
    "ELIZABETH": (-34.7117, 138.6696),
    "SALISBURY": (-34.7583, 138.6417),
    "MARION": (-35.0159, 138.5562),
    "MORPHETT VALE": (-35.1271, 138.5237),
    "GOLDEN GROVE": (-34.7889, 138.7258),
    "MOUNT BARKER": (-35.0667, 138.8560),
    "VICTOR HARBOR": (-35.5524, 138.6174),
    "MURRAY BRIDGE": (-35.1197, 139.2750),
    "PORT PIRIE": (-33.1858, 138.0173),
    "WHYALLA": (-33.0379, 137.5753),
    "PORT AUGUSTA": (-32.4936, 137.7743),
    "PORT LINCOLN": (-34.7217, 135.8559),
    "MOUNT GAMBIER": (-37.8283, 140.7828),
    "NOARLUNGA": (-35.1397, 138.4973)
}

@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and process the uploaded CSV file with optimization for large datasets."""
    try:
        # Read CSV in chunks for large files
        chunks = []
        for chunk in pd.read_csv(uploaded_file, chunksize=50000):
            chunks.append(chunk)
        df = pd.concat(chunks)
        
        # Convert date column
        df['Reported Date'] = pd.to_datetime(df['Reported Date'], format='%d/%m/%Y')
        
        # Aggregate data for faster processing
        df['Month'] = df['Reported Date'].dt.to_period('M')
        monthly_data = df.groupby(['Month', 'Suburb - Incident', 
                                 'Offence Level 1 Description',
                                 'Offence Level 2 Description'])['Offence count'].sum().reset_index()
        
        return df, monthly_data
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None

def create_map(df):
    """Create a folium map with crime incidents using pre-cached coordinates."""
    m = folium.Map(location=[-34.9285, 138.6007], zoom_start=10)
    
    # Create a feature group for better performance
    marker_cluster = folium.FeatureGroup(name="Crime Incidents")
    
    # Group by suburb and count incidents
    suburb_counts = df.groupby('Suburb - Incident')['Offence count'].sum()
    
    # Add markers only for suburbs with known coordinates
    mapped_suburbs = 0
    total_suburbs = len(suburb_counts)
    
    for suburb, count in suburb_counts.items():
        if suburb in SUBURB_COORDINATES:
            mapped_suburbs += 1
            coords = SUBURB_COORDINATES[suburb]
            folium.CircleMarker(
                location=coords,
                radius=np.log(count + 1) * 5,
                popup=f"{suburb}<br>Total incidents: {count:,}",
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6
            ).add_to(marker_cluster)
    
    marker_cluster.add_to(m)
    
    # Display mapping coverage
    st.caption(f"Mapped {mapped_suburbs} out of {total_suburbs} suburbs. Some suburbs may not be shown due to missing coordinates.")
    
    return m

@st.cache_data
def create_time_series(_df, monthly_data):
    """Create time series analysis using aggregated monthly data."""
    monthly_crimes = monthly_data.groupby('Month')['Offence count'].sum().reset_index()
    monthly_crimes['Month'] = monthly_crimes['Month'].astype(str)
    
    fig = px.line(monthly_crimes,
                  x='Month',
                  y='Offence count',
                  title='Monthly Crime Incidents',
                  template='plotly_white')
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Number of Incidents",
        showlegend=False,
        hovermode='x unified'
    )
    return fig

@st.cache_data
def create_crime_type_breakdown(monthly_data):
    """Create breakdown of crime types using aggregated data."""
    crime_types = monthly_data.groupby('Offence Level 2 Description')['Offence count'].sum()
    crime_types = crime_types.sort_values(ascending=True)
    
    fig = px.bar(crime_types,
                 orientation='h',
                 title='Crime Types Distribution',
                 template='plotly_white',
                 labels={'value': 'Number of Incidents',
                        'index': 'Crime Type'})
    fig.update_layout(
        showlegend=False,
        xaxis_title="Number of Incidents",
        yaxis_title="Crime Type",
        height=600
    )
    return fig

@st.cache_data
def create_top_suburbs(df):
    """Create analysis of top suburbs by crime count."""
    suburb_crimes = df.groupby('Suburb - Incident')['Offence count'].sum()
    top_suburbs = suburb_crimes.nlargest(10).sort_values(ascending=True)
    
    fig = px.bar(top_suburbs,
                 orientation='h',
                 title='Top 10 Suburbs by Crime Incidents',
                 template='plotly_white',
                 labels={'value': 'Number of Incidents',
                        'index': 'Suburb'})
    fig.update_layout(
        showlegend=False,
        xaxis_title="Number of Incidents",
        yaxis_title="Suburb"
    )
    return fig

def main():
    st.title("🚨 South Australia Crime Statistics Dashboard")
    
    st.sidebar.header("About")
    st.sidebar.info(
        "This dashboard visualizes crime statistics in South Australia. "
        "Upload a CSV file containing crime data to begin analysis."
    )
    
    uploaded_file = st.file_uploader("Upload your crime statistics CSV file", type="csv")
    
    if uploaded_file is not None:
        with st.spinner('Processing data... This may take a moment.'):
            df, monthly_data = load_and_process_data(uploaded_file)
            
        if df is not None and monthly_data is not None:
            # Display basic statistics in the sidebar
            st.sidebar.header("Summary Statistics")
            total_incidents = df['Offence count'].sum()
            unique_suburbs = df['Suburb - Incident'].nunique()
            date_range = f"{df['Reported Date'].min().strftime('%d/%m/%Y')} - {df['Reported Date'].max().strftime('%d/%m/%Y')}"
            
            st.sidebar.metric("Total Incidents", f"{total_incidents:,}")
            st.sidebar.metric("Unique Suburbs", unique_suburbs)
            st.sidebar.metric("Date Range", date_range)
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["📍 Map", "📈 Trends", "🔍 Analysis"])
            
            with tab1:
                st.subheader("Geographic Distribution of Crime")
                m = create_map(df)
                folium_static(m, width=1000, height=600)
            
            with tab2:
                st.subheader("Crime Trends Over Time")
                time_series_fig = create_time_series(df, monthly_data)
                st.plotly_chart(time_series_fig, use_container_width=True)
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    crime_types_fig = create_crime_type_breakdown(monthly_data)
                    st.plotly_chart(crime_types_fig, use_container_width=True)
                
                with col2:
                    top_suburbs_fig = create_top_suburbs(df)
                    st.plotly_chart(top_suburbs_fig, use_container_width=True)

if __name__ == "__main__":
    main()
