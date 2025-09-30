# 🌊 ARGO Float Chat - Enhanced Multimodal Version
# AI-Powered Conversational Interface for Ocean Data Discovery with Image Analysis

import os
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import numpy as np
from datetime import datetime, timedelta
import base64
from PIL import Image
import io

# LLM via Ollama (OpenAI-compatible) with multimodal support
from dotenv import load_dotenv
from openai import OpenAI

# Load environment and set Ollama host (default localhost)
load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

# Create an OpenAI-compatible client pointing to Ollama
client = OpenAI(
    base_url=f"{OLLAMA_HOST}/v1",
    api_key="ollama"
)

# Model configuration - now supports multimodal
MULTIMODAL_MODEL = os.getenv("MULTIMODAL_MODEL", "llava-phi3")
TEXT_MODEL = os.getenv("TEXT_MODEL", "llama3")
DB_PATH = "data/floats.db"

# Configure page with dark theme
st.set_page_config(
    page_title="ARGO Float Chat - Multimodal",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_query' not in st.session_state:
    st.session_state.current_query = ''
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_float' not in st.session_state:
    st.session_state.selected_float = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# Enhanced CSS for multimodal interface
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: white;
    }
    
    .main-header {
        background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .metric-card {
        background-color: #2d2d2d;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #404040;
        text-align: center;
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #cccccc;
        margin-top: 0.5rem;
    }
    
    .chat-container {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #404040;
    }
    
    .image-upload-container {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 10px;
        border: 2px dashed #4CAF50;
        text-align: center;
        margin: 1rem 0;
    }
    
    .multimodal-response {
        background-color: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    
    .stSelectbox > div > div {
        background-color: #404040;
        color: white;
    }
    
    .query-button {
        background-color: #2196F3;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Utility functions for image processing
def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def process_uploaded_image(uploaded_file):
    """Process uploaded image file"""
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            # Resize image if too large (for performance)
            max_size = (1024, 1024)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    return None

# Existing database functions (unchanged)
@st.cache_data
def get_summary():
    """Get float summary from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """SELECT float_id, 
               COUNT(DISTINCT profile_date) AS total_profiles,
               MIN(profile_date) as first_profile,
               MAX(profile_date) as last_profile,
               COUNT(*) AS total_measurements
               FROM measurements 
               GROUP BY float_id""",
            conn
        )
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

@st.cache_data
def get_profiles(float_id, max_rows=5000):
    """Get profiles for a specific float"""
    try:
        conn = sqlite3.connect(DB_PATH)
        q = """
        SELECT profile_date, depth, temperature, salinity, mld
        FROM measurements 
        WHERE float_id=? 
        ORDER BY profile_date, depth 
        LIMIT ?
        """
        df = pd.read_sql_query(q, conn, params=(str(float_id), max_rows))
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

@st.cache_data
def get_float_locations():
    """Get float locations for mapping"""
    try:
        # Simulate locations if not in DB (you can modify this)
        float_locations = {
            '2902746': {'lat': -20.0948, 'lon': 70.3452, 'region': 'Central Indian Ocean'},
            '2902747': {'lat': -14.0183, 'lon': 77.8562, 'region': 'East Indian Ocean'},
            '2902748': {'lat': -9.8416, 'lon': 87.139, 'region': 'Bay of Bengal'},
            '2902749': {'lat': -5.1188, 'lon': 93.4933, 'region': 'Andaman Sea'},
            '2902750': {'lat': -0.1789, 'lon': 102.3396, 'region': 'Indonesian Waters'}
        }
        return float_locations
    except Exception:
        return {}

# Enhanced multimodal LLM integration
def call_multimodal_llm(query, image=None, float_data=None):
    """
    Call LLaVA Phi-3 model with optional image input for oceanographic analysis
    """
    try:
        # Prepare system prompt for oceanographic context
        system_prompt = """You are an expert oceanographer and AI assistant specializing in ARGO float data analysis. 
        You can analyze both textual oceanographic data and images related to ocean science.
        
        When analyzing images, look for:
        - Oceanographic instruments and equipment
        - Charts, graphs, and data visualizations
        - Satellite imagery of ocean features
        - Marine life and ocean conditions
        - Scientific diagrams and schematics
        
        Always provide scientifically accurate responses and relate image content to oceanographic concepts when possible."""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Prepare user message
        user_content = query
        
        # Add float data context if available
        if float_data is not None and not float_data.empty:
            user_content += f"\n\nCurrent ARGO float data context:\n{float_data.head().to_string()}"
        
        user_message = {"role": "user", "content": user_content}
        
        # Add image if provided
        if image is not None:
            # Convert PIL Image to base64
            img_base64 = encode_image_to_base64(image)
            # For OpenAI-compatible API, images are typically handled in content
            user_message["content"] = [
                {"type": "text", "text": user_content},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]
        
        messages.append(user_message)
        
        # Choose model based on whether we have an image
        model_to_use = MULTIMODAL_MODEL if image is not None else TEXT_MODEL
        
        # Call the model
        response = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0.1,
            max_tokens=800,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error calling multimodal LLM: {e}\nFalling back to text-only analysis..."

# Enhanced header
def create_main_header():
    """Create the main header with multimodal capabilities notice"""
    st.markdown("""
    <div class="main-header">
        <h1>🌊 ARGO Float Chat - Multimodal AI</h1>
        <p>AI-Powered Conversational Interface for Ocean Data Discovery with Image Analysis</p>
        <p><strong>Now supporting:</strong> Text queries + Image analysis with LLaVA Phi-3</p>
    </div>
    """, unsafe_allow_html=True)

# Enhanced chat interface with image support
def create_enhanced_chat_interface():
    """Create enhanced chat interface with image upload capability"""
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Image upload section
    st.markdown("### 📸 Upload Image for Analysis")
    st.markdown('<div class="image-upload-container">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an oceanographic image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload charts, satellite images, instrument photos, or any ocean-related visuals"
    )
    
    if uploaded_file is not None:
        image = process_uploaded_image(uploaded_file)
        if image is not None:
            st.session_state.uploaded_image = image
            # Display the uploaded image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            st.success("Image uploaded successfully! You can now ask questions about this image.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced query suggestions
    st.markdown("### 💬 Ask Questions (Text + Image)")
    
    # Multimodal query examples
    col1, col2, col3 = st.columns(3)
    
    multimodal_queries = [
        "Analyze this oceanographic chart",
        "What instruments are shown in this image?",
        "Explain the ocean features in this satellite image"
    ]
    
    text_queries = [
        "Show me temperature profiles for selected float",
        "What's the salinity trend over time?",
        "Compare data from different ocean regions"
    ]
    
    with col1:
        st.markdown("**Image Analysis:**")
        for i, query in enumerate(multimodal_queries):
            if st.button(query, key=f"img_q{i}", use_container_width=True):
                st.session_state.current_query = query
    
    with col2:
        st.markdown("**Data Queries:**")
        for i, query in enumerate(text_queries):
            if st.button(query, key=f"text_q{i}", use_container_width=True):
                st.session_state.current_query = query
    
    with col3:
        st.markdown("**Combined Analysis:**")
        combined_queries = [
            "How does this image relate to my float data?",
            "Explain the oceanography shown here",
            "What scientific insights can you draw?"
        ]
        for i, query in enumerate(combined_queries):
            if st.button(query, key=f"comb_q{i}", use_container_width=True):
                st.session_state.current_query = query
    
    # Query input
    st.markdown("### ✍️ Your Question")
    query_input = st.text_input(
        "",
        value=st.session_state.current_query,
        placeholder="Ask about ocean data, uploaded images, or both...",
        label_visibility="collapsed"
    )
    
    # Submit button
    if st.button("🚀 Submit Query", type="primary", use_container_width=True):
        if query_input:
            handle_multimodal_query(query_input)
    
    # Display chat history with image support
    if st.session_state.chat_history:
        st.markdown("### 📜 Recent Conversations")
        for chat in reversed(st.session_state.chat_history[-3:]):  # Show last 3
            with st.expander(f"Q: {chat['query'][:50]}..."):
                st.markdown(f"**Query:** {chat['query']}")
                
                # Show image if it was part of the query
                if 'image' in chat and chat['image'] is not None:
                    st.image(chat['image'], caption="Query Image", width=200)
                
                st.markdown('<div class="multimodal-response">', unsafe_allow_html=True)
                st.markdown(f"**Response:** {chat['response']}")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown(f"**Time:** {chat['timestamp'].strftime('%H:%M:%S')}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def handle_multimodal_query(query):
    """Handle multimodal queries with enhanced context"""
    if not st.session_state.selected_float:
        st.warning("Please select a float in the Data Explorer first.")
        return
    
    chosen_float = st.session_state.selected_float
    
    with st.spinner('🤖 Processing your multimodal query...'):
        try:
            # Get current float data for context
            float_data = get_profiles(chosen_float, max_rows=100)
            
            # Determine if this is a multimodal query
            image = None
            has_image = st.session_state.uploaded_image is not None
            
            if has_image:
                image = st.session_state.uploaded_image
                st.info("🖼️ Including uploaded image in analysis...")
            
            # Call the enhanced multimodal LLM
            response_text = call_multimodal_llm(
                query=query,
                image=image,
                float_data=float_data
            )
            
            # Display response
            st.markdown('<div class="multimodal-response">', unsafe_allow_html=True)
            st.markdown(f"**🤖 AI Response:**\n\n{response_text}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add to chat history
            st.session_state.chat_history.append({
                'query': query,
                'response': response_text,
                'timestamp': datetime.now(),
                'image': image if has_image else None,
                'float_id': chosen_float
            })
            
            # If the query resulted in data analysis, show relevant visualizations
            if any(keyword in query.lower() for keyword in ['temperature', 'salinity', 'profile', 'depth']):
                show_relevant_visualization(chosen_float, query.lower())
                
        except Exception as e:
            st.error(f"Error processing multimodal query: {e}")

def show_relevant_visualization(float_id, query_lower):
    """Show relevant data visualizations based on query content"""
    data = get_profiles(float_id, max_rows=1000)
    
    if data.empty:
        return
    
    st.markdown("### 📊 Relevant Data Visualization")
    
    if 'temperature' in query_lower and 'profile' in query_lower:
        # Temperature profile
        latest_profiles = data.groupby('profile_date').first().tail(5)
        fig = go.Figure()
        
        for date in latest_profiles.index:
            profile_data = data[data['profile_date'] == date]
            fig.add_trace(go.Scatter(
                x=profile_data['temperature'],
                y=profile_data['depth'],
                mode='lines+markers',
                name=f'{date}',
                line=dict(width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title=f"Temperature Profiles - Float {float_id}",
            xaxis_title="Temperature (°C)",
            yaxis_title="Depth (m)",
            yaxis=dict(autorange='reversed'),
            height=500,
            plot_bgcolor='#2d2d2d',
            paper_bgcolor='#2d2d2d',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Main navigation and layout
def main():
    """Main application with enhanced multimodal capabilities"""
    create_main_header()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        nav_options = ["Overview", "Data Explorer", "Multimodal Chat"]
        selected_page = st.selectbox("", nav_options, key="navigation")
        
        st.markdown("## 🔧 Model Configuration")
        st.info(f"**Multimodal Model:** {MULTIMODAL_MODEL}")
        st.info(f"**Text Model:** {TEXT_MODEL}")
        
        # Model status check
        try:
            # Test connection to Ollama
            response = client.chat.completions.create(
                model=TEXT_MODEL,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            st.success("✅ Ollama Connected")
        except:
            st.error("❌ Ollama Not Available")
    
    # Get data
    summary_data = get_summary()
    float_locations = get_float_locations()
    
    # Page routing
    if selected_page == "Overview":
        create_system_overview(summary_data)
        create_float_locations_section(float_locations, summary_data)
        create_float_summary_table(summary_data, float_locations)
        
    elif selected_page == "Data Explorer":
        create_data_explorer_section(summary_data)
        
    elif selected_page == "Multimodal Chat":
        create_enhanced_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("**ARGO Float Chat Multimodal** - Smart India Hackathon 2025 | Team IIT Guwahati")
    st.markdown("*Enhanced with LLaVA Phi-3 for image analysis and ocean science insights*")

# System overview function (unchanged from original)
def create_system_overview(summary_data):
    """Create system overview section with metrics"""
    st.markdown("## 📊 System Overview")
    
    if not summary_data.empty:
        total_floats = len(summary_data)
        total_profiles = summary_data['total_profiles'].sum()
        total_measurements = summary_data['total_measurements'].sum()
    else:
        total_floats, total_profiles, total_measurements = 0, 0, 0
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{total_floats}</div>
            <div class="metric-label">Floats</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{total_profiles}</div>
            <div class="metric-label">Profiles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{total_measurements}</div>
            <div class="metric-label">Measurements</div>
        </div>
        """, unsafe_allow_html=True)

def create_enhanced_floats_map(float_locations, summary_data):
    """Create enhanced interactive map with float locations"""
    # Center map on Indian Ocean
    center_lat, center_lon = -10.0, 80.0
    
    # Create map with dark tiles
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=4,
        tiles=None
    )
    
    # Add dark tile layer
    folium.TileLayer(
        tiles='https://cartodb-basemaps-{s}.global.ssl.fastly.net/dark_all/{z}/{x}/{y}.png',
        attr='CartoDB.DarkMatter',
        name='Dark Theme',
        control=False
    ).add_to(m)
    
    # Add float markers
    for float_id, location in float_locations.items():
        # Get profile count for sizing
        profile_count = 3  # Default
        if not summary_data.empty:
            float_data = summary_data[summary_data['float_id'] == float_id]
            if not float_data.empty:
                profile_count = int(float_data['total_profiles'].iloc[0])
        
        # Size and color based on profile count
        if profile_count > 4:
            radius = 25
            color = '#2196F3'
        elif profile_count > 3:
            radius = 20
            color = '#4CAF50'
        else:
            radius = 15
            color = '#FF9800'
        
        # Create popup content
        popup_content = f"""
        <div style="font-family: Arial; padding: 10px; width: 200px;">
            <h4 style="margin: 0; color: #2196F3;">Float {float_id}</h4>
            <hr style="margin: 5px 0;">
            <p><strong>Region:</strong> {location.get('region', 'Unknown')}</p>
            <p><strong>Location:</strong> {location['lat']:.2f}°, {location['lon']:.2f}°</p>
            <p><strong>Profiles:</strong> {profile_count}</p>
        </div>
        """
        
        folium.CircleMarker(
            location=[location['lat'], location['lon']],
            radius=radius,
            popup=folium.Popup(popup_content, max_width=250),
            tooltip=f"Float {float_id} ({profile_count} profiles)",
            color='white',
            weight=2,
            fillColor=color,
            fillOpacity=0.8
        ).add_to(m)
    
    return m

def create_float_locations_section(float_locations, summary_data):
    """Create the Float Locations section"""
    st.markdown("## 🗺️ Float Locations")
    st.markdown("ARGO Float Locations in Indian Ocean")
    
    if float_locations:
        float_map = create_enhanced_floats_map(float_locations, summary_data)
        map_data = st_folium(float_map, width=800, height=500, key="main_map")
        return map_data
    else:
        st.error("No float location data available")
        return None

def create_float_summary_table(summary_data, float_locations):
    """Create float summary table matching your design"""
    st.markdown("## 📋 Float Summary")
    
    if not summary_data.empty and float_locations:
        # Prepare data for the table
        table_data = []
        for i, row in summary_data.iterrows():
            float_id = str(row['float_id'])
            if float_id in float_locations:
                location = float_locations[float_id]
                table_data.append({
                    '#': i,
                    'float_id': int(float_id),
                    'latitude': location['lat'],
                    'longitude': location['lon'],
                    'total_profiles': row['total_profiles']
                })
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "#": st.column_config.NumberColumn("#", width="small"),
                    "float_id": st.column_config.NumberColumn("float_id", width="medium"),
                    "latitude": st.column_config.NumberColumn("latitude", format="%.4f"),
                    "longitude": st.column_config.NumberColumn("longitude", format="%.4f"),
                    "total_profiles": st.column_config.NumberColumn("total_profiles", width="small")
                }
            )

def create_data_explorer_section(summary_data):
    """Create data explorer section with profile visualization"""
    st.markdown("## 🔍 Data Explorer")
    
    if summary_data.empty:
        st.warning("No float data available")
        return
    
    # Float selection
    float_ids = summary_data['float_id'].dropna().astype(str).tolist()
    if not float_ids:
        st.warning("No floats found in database.")
        return
    
    chosen_float = st.selectbox("Select Float", float_ids)
    st.session_state.selected_float = chosen_float
    
    # Get profiles for selected float
    data = get_profiles(chosen_float)
    st.subheader(f"Profiles for Float {chosen_float}")
    
    if data.empty:
        st.warning("No data for selected float.")
        return
    
    # Profile date selection
    prof_dates = sorted(data['profile_date'].dropna().unique().tolist())
    if not prof_dates:
        st.warning("No profile dates available.")
        return
    
    # Create profile ID format like in your design
    profile_options = [f"{chosen_float}-{i+1:03d}" for i in range(len(prof_dates))]
    selected_profile = st.selectbox("Select Profile", profile_options)
    
    # Extract profile index and get corresponding date
    profile_idx = int(selected_profile.split('-')[-1]) - 1
    if profile_idx < len(prof_dates):
        selected_date = prof_dates[profile_idx]
        prof = data[data['profile_date'] == selected_date].sort_values('depth')
        
        st.markdown(f"### Profile {selected_profile}")
        
        if not prof.empty:
            # Create side-by-side temperature and salinity profiles
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Temperature Profile", "Salinity Profile"),
                shared_yaxes=True
            )
            
            # Temperature profile (red line with dots)
            fig.add_trace(
                go.Scatter(
                    x=prof['temperature'],
                    y=prof['depth'],
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color='red', width=2),
                    marker=dict(size=4, color='red')
                ),
                row=1, col=1
            )
            
            # Salinity profile (blue line with dots)
            fig.add_trace(
                go.Scatter(
                    x=prof['salinity'],
                    y=prof['depth'],
                    mode='lines+markers',
                    name='Salinity',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4, color='blue')
                ),
                row=1, col=2
            )
            
            # Update layout to match your design
            fig.update_layout(
                height=600,
                showlegend=False,
                plot_bgcolor='#2d2d2d',
                paper_bgcolor='#2d2d2d',
                font=dict(color='white')
            )
            
            fig.update_yaxes(title_text="Depth (m)", autorange='reversed', gridcolor='#404040')
            fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1, gridcolor='#404040')
            fig.update_xaxes(title_text="Salinity (PSU)", row=1, col=2, gridcolor='#404040')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mixed layer depth info
            if prof['mld'].notna().any():
                mld_val = prof['mld'].dropna().iloc[0]
                st.info(f"Estimated mixed layer depth: {mld_val:.1f} m")

if __name__ == "__main__":
    main()