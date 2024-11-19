import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import calendar

# Configure Streamlit page
st.set_page_config(
    page_title="SA Crime Statistics Analysis Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pre-defined coordinates for major South Australian suburbs
SUBURB_COORDINATES = {
    "ADELAIDE": (-34.9285, 138.6007),
    "NORTH ADELAIDE": (-34.9066, 138.5944),
    "ADELAIDE": (-34.9285, 138.6007),
    "NORTH ADELAIDE": (-34.9066, 138.5944),
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
    "NOARLUNGA": (-35.1397, 138.4973),
    "SEMAPHORE": (-34.8393, 138.4830),
    "HENLEY BEACH": (-34.9208, 138.4945),
    "GRANGE": (-34.9041, 138.4892),
    "UNLEY": (-34.9500, 138.6000),
    "NORWOOD": (-34.9211, 138.6326),
    "PROSPECT": (-34.8853, 138.5945),
    "BURNSIDE": (-34.9333, 138.6500),
    "MAGILL": (-34.9167, 138.6833),
    "CAMPBELLTOWN": (-34.8833, 138.6667),
    "PARADISE": (-34.9000, 138.6833),
    "TEA TREE GULLY": (-34.8167, 138.7000),
    "HOPE VALLEY": (-34.8333, 138.7000),
    "HOLDEN HILL": (-34.8500, 138.6667),
    "POORAKA": (-34.8167, 138.6167),
    "MAWSON LAKES": (-34.8081, 138.6126),
    "PARAFIELD": (-34.7989, 138.6330),
    "PARALOWIE": (-34.7667, 138.6167),
    "INGLE FARM": (-34.8333, 138.6333),
    "PARA HILLS": (-34.8000, 138.6500),
    "MODBURY HEIGHTS": (-34.8167, 138.6833),
    "GREENWITH": (-34.7500, 138.7167),
    "WYNN VALE": (-34.7833, 138.7000),
    "SALISBURY EAST": (-34.7667, 138.6667),
    "SALISBURY NORTH": (-34.7500, 138.6333),
    "BURTON": (-34.7333, 138.6167),
    "EDINBURGH": (-34.7111, 138.6206),
    "ELIZABETH SOUTH": (-34.7333, 138.6667),
    "ELIZABETH VALE": (-34.7167, 138.6667),
    "ELIZABETH DOWNS": (-34.7000, 138.6833),
    "CRAIGMORE": (-34.7000, 138.7000),
    "BLAKEVIEW": (-34.6833, 138.7167),
    "GAWLER": (-34.5981, 138.7450),
    "GAWLER EAST": (-34.5933, 138.7567),
    "EVANSTON": (-34.6167, 138.7333),
    "WILLASTON": (-34.5833, 138.7333),
    "ANGLE VALE": (-34.6549, 138.6607),
    "VIRGINIA": (-34.6667, 138.5500),
    "MUNNO PARA": (-34.6833, 138.6667),
    "SMITHFIELD": (-34.6833, 138.6833),
    "DAVOREN PARK": (-34.7000, 138.6667),
    "ANDREWS FARM": (-34.6833, 138.6833),
    "LEWISTON": (-34.6333, 138.5833),
    "TWO WELLS": (-34.5833, 138.5167),
    "MALLALA": (-34.4500, 138.5167),
    "DUBLIN": (-34.4500, 138.3500)
    # ... [previous coordinates remain the same]
}

@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and process the uploaded CSV file with comprehensive data analysis."""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Convert date column
        df['Reported Date'] = pd.to_datetime(df['Reported Date'], format='%d/%m/%Y')
        
        # Add temporal features
        df['Year'] = df['Reported Date'].dt.year
        df['Month'] = df['Reported Date'].dt.month
        df['Day'] = df['Reported Date'].dt.day
        df['Day_of_Week'] = df['Reported Date'].dt.day_name()
        df['Month_Name'] = df['Reported Date'].dt.month_name()
        df['Quarter'] = df['Reported Date'].dt.quarter
        df['Is_Weekend'] = df['Reported Date'].dt.dayofweek.isin([5, 6])
        
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def create_crime_heatmap(df):
    """Create a heatmap of crime patterns by day and month."""
    # Aggregate data by month and day of week
    heatmap_data = df.groupby(['Day_of_Week', 'Month_Name'])['Offence count'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Day_of_Week', 
                                      columns='Month_Name', 
                                      values='Offence count')
    
    # Reorder days and months
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months_order = list(calendar.month_name)[1:]
    heatmap_pivot = heatmap_pivot.reindex(days_order)[months_order]
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title='Crime Heatmap by Day and Month',
        xaxis_title='Month',
        yaxis_title='Day of Week'
    )
    return fig

def create_crime_type_analysis(df):
    """Create detailed analysis of crime types and their patterns."""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Crime Categories Distribution', 
                       'Top 10 Specific Offences',
                       'Crime Types by Day of Week',
                       'Monthly Trend by Major Category'),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. Overall distribution of major crime categories
    crime_cats = df.groupby('Offence Level 1 Description')['Offence count'].sum()
    fig.add_trace(
        go.Pie(labels=crime_cats.index, values=crime_cats.values, showlegend=True),
        row=1, col=1
    )
    
    # 2. Top 10 specific offences
    top_crimes = df.groupby('Offence Level 3 Description')['Offence count'].sum().nlargest(10)
    fig.add_trace(
        go.Bar(x=top_crimes.values, y=top_crimes.index, orientation='h'),
        row=1, col=2
    )
    
    # 3. Crime types by day of week
    dow_crimes = df.groupby(['Day_of_Week', 'Offence Level 1 Description'])['Offence count'].sum().reset_index()
    for crime_type in dow_crimes['Offence Level 1 Description'].unique():
        data = dow_crimes[dow_crimes['Offence Level 1 Description'] == crime_type]
        fig.add_trace(
            go.Bar(name=crime_type, x=data['Day_of_Week'], y=data['Offence count']),
            row=2, col=1
        )
    
    # 4. Monthly trend by major category
    monthly_trend = df.groupby(['Year', 'Month', 'Offence Level 1 Description'])['Offence count'].sum().reset_index()
    monthly_trend['Date'] = pd.to_datetime(monthly_trend[['Year', 'Month']].assign(DAY=1))
    
    for crime_type in monthly_trend['Offence Level 1 Description'].unique():
        data = monthly_trend[monthly_trend['Offence Level 1 Description'] == crime_type]
        fig.add_trace(
            go.Scatter(name=crime_type, x=data['Date'], y=data['Offence count'], mode='lines'),
            row=2, col=2
        )
    
    fig.update_layout(height=1000, showlegend=True)
    return fig

def create_geographic_analysis(df):
    """Create geographic analysis with multiple layers of insight."""
    # Base map
    m = folium.Map(location=[-34.9285, 138.6007], zoom_start=10)
    
    # Create different feature groups for different crime types
    crime_groups = {}
    for crime_type in df['Offence Level 1 Description'].unique():
        crime_groups[crime_type] = folium.FeatureGroup(name=crime_type)
    
    # Add crime incidents to respective groups
    for suburb, crime_data in df.groupby('Suburb - Incident'):
        if suburb in SUBURB_COORDINATES:
            coords = SUBURB_COORDINATES[suburb]
            
            for crime_type, type_data in crime_data.groupby('Offence Level 1 Description'):
                count = type_data['Offence count'].sum()
                
                folium.CircleMarker(
                    location=coords,
                    radius=np.log(count + 1) * 3,
                    popup=f"{suburb}<br>{crime_type}<br>Incidents: {count:,}",
                    color=get_crime_color(crime_type),
                    fill=True,
                    fill_opacity=0.7
                ).add_to(crime_groups[crime_type])
    
    # Add all feature groups to map
    for group in crime_groups.values():
        group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def get_crime_color(crime_type):
    """Return color based on crime type."""
    colors = {
        "OFFENCES AGAINST PROPERTY": "red",
        "OFFENCES AGAINST THE PERSON": "darkred",
        "DRUG OFFENCES": "orange",
        "PUBLIC ORDER OFFENCES": "blue",
        "OTHER OFFENCES": "purple"
    }
    return colors.get(crime_type, "gray")

def create_trend_analysis(df):
    """Create comprehensive trend analysis."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Crime Trend', 
                       'Weekend vs. Weekday Distribution',
                       'Quarterly Comparison',
                       'Time of Year Analysis'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. Monthly trend with moving average
    monthly_crimes = df.groupby(['Year', 'Month'])['Offence count'].sum().reset_index()
    monthly_crimes['Date'] = pd.to_datetime(monthly_crimes[['Year', 'Month']].assign(DAY=1))
    fig.add_trace(
        go.Scatter(x=monthly_crimes['Date'], y=monthly_crimes['Offence count'], 
                  name='Monthly Total'),
        row=1, col=1
    )
    
    # Add 3-month moving average
    monthly_crimes['MA3'] = monthly_crimes['Offence count'].rolling(3).mean()
    fig.add_trace(
        go.Scatter(x=monthly_crimes['Date'], y=monthly_crimes['MA3'], 
                  name='3-Month Moving Average',
                  line=dict(dash='dash')),
        row=1, col=1
    )
    
    # 2. Weekend vs. Weekday comparison
    weekend_comp = df.groupby(['Is_Weekend', 'Offence Level 1 Description'])['Offence count'].mean().reset_index()
    for crime_type in weekend_comp['Offence Level 1 Description'].unique():
        data = weekend_comp[weekend_comp['Offence Level 1 Description'] == crime_type]
        fig.add_trace(
            go.Bar(name=crime_type, x=['Weekday', 'Weekend'], 
                  y=data['Offence count']),
            row=1, col=2
        )
    
    # 3. Quarterly comparison
    quarterly = df.groupby(['Year', 'Quarter'])['Offence count'].sum().reset_index()
    fig.add_trace(
        go.Bar(x=quarterly.apply(lambda x: f"{x['Year']} Q{x['Quarter']}", axis=1),
               y=quarterly['Offence count']),
        row=2, col=1
    )
    
    # 4. Time of year analysis (month-over-month)
    monthly_avg = df.groupby('Month_Name')['Offence count'].mean().reindex(calendar.month_name[1:])
    fig.add_trace(
        go.Scatter(x=monthly_avg.index, y=monthly_avg.values, 
                  mode='lines+markers',
                  name='Average by Month'),
        row=2, col=2
    )
    
    fig.update_layout(height=1000, showlegend=True)
    return fig

def main():
    st.title("🚨 Advanced Crime Statistics Analysis Dashboard")
    
    uploaded_file = st.file_uploader("Upload your crime statistics CSV file", type="csv")
    
    if uploaded_file is not None:
        with st.spinner('Processing data... This may take a moment.'):
            df = load_and_process_data(uploaded_file)
            
        if df is not None:
            # Sidebar filters
            st.sidebar.header("Filters")
            
            # Date range filter
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(df['Reported Date'].min(), df['Reported Date'].max()),
                min_value=df['Reported Date'].min().date(),
                max_value=df['Reported Date'].max().date()
            )
            
            # Crime type filter
            crime_types = st.sidebar.multiselect(
                "Select Crime Types",
                options=df['Offence Level 1 Description'].unique(),
                default=df['Offence Level 1 Description'].unique()
            )
            
            # Filter data
            mask = (df['Reported Date'].dt.date >= date_range[0]) & \
                   (df['Reported Date'].dt.date <= date_range[1]) & \
                   (df['Offence Level 1 Description'].isin(crime_types))
            filtered_df = df[mask]
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Incidents", f"{filtered_df['Offence count'].sum():,}")
            with col2:
                st.metric("Unique Suburbs", f"{filtered_df['Suburb - Incident'].nunique():,}")
            with col3:
                daily_avg = filtered_df.groupby('Reported Date')['Offence count'].sum().mean()
                st.metric("Daily Average", f"{daily_avg:.1f}")
            with col4:
                pct_weekend = (filtered_df[filtered_df['Is_Weekend']]['Offence count'].sum() / 
                             filtered_df['Offence count'].sum() * 100)
                st.metric("Weekend %", f"{pct_weekend:.1f}%")
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Crime Patterns", "🗺️ Geographic Analysis", 
                                            "📈 Trend Analysis", "🔍 Detailed Statistics"])
            
            with tab1:
                st.plotly_chart(create_crime_type_analysis(filtered_df), use_container_width=True)
                st.plotly_chart(create_crime_heatmap(filtered_df), use_container_width=True)
            
            with tab2:
                st.subheader("Geographic Distribution of Crime")
                folium_static(create_geographic_analysis(filtered_df), width=1400)
            
            with tab3:
                st.plotly_chart(create_trend_analysis(filtered_df), use_container_width=True)
            
            with tab4:
                st.subheader("Detailed Crime Statistics")
                
                # Show top suburbs for each crime type
                for crime_type in crime_types:
                    st.write(f"### Top 10 Suburbs - {crime_type}")
                    crime_data = filtered_df[filtered_df['Offence Level 1 Description'] == crime_type]
                    top_suburbs = crime_data.groupby('Suburb - Incident')['Offence count'].sum().nlargest(10)
                    st.bar_chart(top_suburbs)
                
                # Show detailed statistics
                st.write("### Detailed Crime Breakdown")
                detailed_stats = filtered_df.groupby(['Offence Level 2 Description', 'Offence Level 3 Description'])\
                    ['Offence count'].sum().reset_index()
                detailed_stats = detailed_stats.sort_values('Offence count', ascending=False)
                st.dataframe(detailed_stats, use_container_width=True)

if __name__ == "__main__":
    main()
