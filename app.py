import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
from datetime import datetime
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time

# Configure Streamlit page
st.set_page_config(
    page_title="SA Crime Statistics Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for coordinates cache
if 'suburb_coordinates' not in st.session_state:
    st.session_state.suburb_coordinates = {}

@st.cache_data
def geocode_suburb(suburb, state="South Australia", country="Australia"):
    """Geocode a suburb name using Nominatim with caching."""
    if suburb in st.session_state.suburb_coordinates:
        return st.session_state.suburb_coordinates[suburb]
    
    geolocator = Nominatim(user_agent="sa_crime_stats_dashboard")
    try:
        query = f"{suburb}, {state}, {country}"
        location = geolocator.geocode(query)
        time.sleep(1)  # Respect Nominatim's usage policy
        
        if location:
            coords = (location.latitude, location.longitude)
            st.session_state.suburb_coordinates[suburb] = coords
            return coords
    except (GeocoderTimedOut, GeocoderUnavailable):
        pass
    return None

@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and process the uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Convert date column
        df['Reported Date'] = pd.to_datetime(df['Reported Date'], format='%d/%m/%Y')
        
        # Ensure required columns exist
        required_columns = [
            'Reported Date', 'Suburb - Incident', 'Offence Level 1 Description',
            'Offence Level 2 Description', 'Offence Level 3 Description', 'Offence count'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def create_map(df):
    """Create a folium map with crime incidents."""
    # Center the map on Adelaide
    m = folium.Map(location=[-34.9285, 138.6007], zoom_start=10)
    
    # Group by suburb and count incidents
    suburb_counts = df.groupby('Suburb - Incident')['Offence count'].sum()
    
    # Add markers for each suburb
    for suburb, count in suburb_counts.items():
        coords = geocode_suburb(suburb)
        if coords:
            folium.CircleMarker(
                location=coords,
                radius=np.log(count + 1) * 5,
                popup=f"{suburb}<br>Total incidents: {count}",
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6
            ).add_to(m)
    
    return m

def create_time_series(df):
    """Create time series analysis of crime incidents."""
    daily_crimes = df.groupby('Reported Date')['Offence count'].sum().reset_index()
    
    fig = px.line(daily_crimes, 
                  x='Reported Date', 
                  y='Offence count',
                  title='Daily Crime Incidents Over Time')
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Incidents",
        showlegend=False
    )
    return fig

def create_crime_type_breakdown(df):
    """Create breakdown of crime types."""
    crime_types = df.groupby('Offence Level 2 Description')['Offence count'].sum()
    crime_types = crime_types.sort_values(ascending=True)
    
    fig = px.bar(crime_types,
                 orientation='h',
                 title='Crime Types Distribution',
                 labels={'value': 'Number of Incidents', 
                        'index': 'Crime Type'})
    return fig

def create_suburb_analysis(df):
    """Create analysis of crime by suburb."""
    suburb_crimes = df.groupby('Suburb - Incident')['Offence count'].sum()
    suburb_crimes = suburb_crimes.sort_values(ascending=True).tail(10)
    
    fig = px.bar(suburb_crimes,
                 orientation='h',
                 title='Top 10 Suburbs by Crime Incidents',
                 labels={'value': 'Number of Incidents', 
                        'index': 'Suburb'})
    return fig

def main():
    st.title("🚨 South Australia Crime Statistics Dashboard")
    
    st.sidebar.header("About")
    st.sidebar.info(
        "This dashboard visualizes crime statistics in South Australia. "
        "Upload a CSV file with crime data to begin analysis."
    )
    
    st.sidebar.header("Instructions")
    st.sidebar.markdown(
        """
        1. Upload your crime statistics CSV file
        2. View the map of crime incidents
        3. Analyze trends over time
        4. Explore crime patterns
        
        Required CSV columns:
        - Reported Date (DD/MM/YYYY)
        - Suburb - Incident
        - Offence Level 1 Description
        - Offence Level 2 Description
        - Offence Level 3 Description
        - Offence count
        """
    )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your crime statistics CSV file",
        type="csv",
        help="Upload a CSV file containing crime statistics data"
    )
    
    if uploaded_file is not None:
        # Load and process data
        df = load_and_process_data(uploaded_file)
        
        if df is not None:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📍 Map View", "📈 Time Analysis", "🔍 Crime Patterns"])
            
            with tab1:
                st.header("Geographic Distribution of Crime")
                with st.spinner("Loading map... This may take a few moments."):
                    m = create_map(df)
                    folium_static(m, width=1000, height=600)
            
            with tab2:
                st.header("Temporal Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Time series plot
                    time_series_fig = create_time_series(df)
                    st.plotly_chart(time_series_fig, use_container_width=True)
                
                with col2:
                    # Day of week analysis
                    df['Day of Week'] = df['Reported Date'].dt.day_name()
                    dow_crimes = df.groupby('Day of Week')['Offence count'].sum()
                    dow_fig = px.bar(dow_crimes, 
                                   title='Crimes by Day of Week',
                                   labels={'value': 'Number of Incidents', 
                                          'index': 'Day of Week'})
                    st.plotly_chart(dow_fig, use_container_width=True)
            
            with tab3:
                st.header("Crime Patterns Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Crime types breakdown
                    crime_types_fig = create_crime_type_breakdown(df)
                    st.plotly_chart(crime_types_fig, use_container_width=True)
                
                with col2:
                    # Suburb analysis
                    suburb_fig = create_suburb_analysis(df)
                    st.plotly_chart(suburb_fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                total_incidents = df['Offence count'].sum()
                unique_suburbs = df['Suburb - Incident'].nunique()
                most_common_crime = df.groupby('Offence Level 2 Description')['Offence count'].sum().idxmax()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Incidents", f"{total_incidents:,}")
                col2.metric("Unique Suburbs", unique_suburbs)
                col3.metric("Most Common Crime Type", most_common_crime)

if __name__ == "__main__":
    main()
