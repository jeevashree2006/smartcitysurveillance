import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from collections import Counter

class Dashboard:
    def __init__(self):
        """Initialize the dashboard component"""
        pass
    
    def render(self, detection_logs):
        """
        Render the main dashboard with analytics and visualizations
        
        Args:
            detection_logs: List of detection dictionaries
        """
        st.subheader("📊 Real-time Analytics Dashboard")
        
        if not detection_logs:
            st.info("🔍 No detection data available yet. Start surveillance to see analytics.")
            self._render_empty_dashboard()
            return
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(detection_logs)
        
        # Parse timestamps if available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
        
        # Create dashboard layout
        self._render_summary_metrics(df)
        self._render_detection_charts(df)
        self._render_temporal_analysis(df)
        self._render_detection_table(df)
    
    def _render_empty_dashboard(self):
        """Render dashboard with placeholder content when no data is available"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", "0", "0")
        with col2:
            st.metric("Active Alerts", "0", "0")
        with col3:
            st.metric("Avg Confidence", "0%", "0%")
        with col4:
            st.metric("Peak Hour", "--", "--")
        
        # Empty charts
        st.markdown("### Detection Distribution")
        st.info("No data to display")
        
        st.markdown("### Temporal Trends")
        st.info("No data to display")
    
    def _render_summary_metrics(self, df):
        """Render summary metrics at the top of the dashboard"""
        col1, col2, col3, col4 = st.columns(4)
        
        total_detections = len(df)
        
        # Calculate alerts (high-priority detections)
        alert_categories = ['Graffiti', 'Posters']
        alert_detections = len(df[df['category'].isin(alert_categories)]) if 'category' in df.columns else 0
        
        # Average confidence
        avg_confidence = df['confidence'].mean() if 'confidence' in df.columns and len(df) > 0 else 0
        
        # Peak hour
        peak_hour = df['hour'].value_counts().index[0] if 'hour' in df.columns and len(df) > 0 else None
        
        with col1:
            st.metric(
                "Total Detections", 
                total_detections,
                delta=f"+{min(total_detections, 10)}" if total_detections > 0 else "0"
            )
        
        with col2:
            st.metric(
                "Active Alerts", 
                alert_detections,
                delta=f"+{alert_detections}" if alert_detections > 0 else "0",
                delta_color="inverse"
            )
        
        with col3:
            st.metric(
                "Avg Confidence", 
                f"{avg_confidence:.1%}" if avg_confidence > 0 else "0%",
                delta=f"{avg_confidence:.1%}" if avg_confidence > 0 else "0%"
            )
        
        with col4:
            peak_hour_str = f"{peak_hour:02d}:00" if peak_hour is not None else "--"
            st.metric(
                "Peak Hour", 
                peak_hour_str,
                delta="Most Active" if peak_hour is not None else "--"
            )
    
    def _render_detection_charts(self, df):
        """Render charts showing detection distribution"""
        st.markdown("### 📈 Detection Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution pie chart
            if 'category' in df.columns:
                category_counts = df['category'].value_counts()
                
                fig_pie = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Detections by Category",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No category data available")
        
        with col2:
            # Confidence distribution histogram
            if 'confidence' in df.columns:
                fig_hist = px.histogram(
                    df,
                    x='confidence',
                    nbins=20,
                    title="Confidence Score Distribution",
                    labels={'confidence': 'Confidence Score', 'count': 'Frequency'},
                    color_discrete_sequence=['#1f77b4']
                )
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No confidence data available")
    
    def _render_temporal_analysis(self, df):
        """Render temporal analysis charts"""
        st.markdown("### ⏰ Temporal Analysis")
        
        if 'timestamp' not in df.columns:
            st.info("No temporal data available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly distribution
            if 'hour' in df.columns:
                hourly_counts = df['hour'].value_counts().sort_index()
                
                fig_hourly = go.Figure(data=[
                    go.Bar(
                        x=[f"{h:02d}:00" for h in hourly_counts.index],
                        y=hourly_counts.values,
                        marker_color='lightblue',
                        name='Detections'
                    )
                ])
                fig_hourly.update_layout(
                    title="Detections by Hour",
                    xaxis_title="Hour of Day",
                    yaxis_title="Number of Detections",
                    height=400
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            else:
                st.info("No hourly data available")
        
        with col2:
            # Timeline of detections
            if len(df) > 1:
                # Group by 10-minute intervals for better visualization
                df['time_rounded'] = df['timestamp'].dt.floor('10min')
                timeline_data = df.groupby(['time_rounded', 'category']).size().reset_index(name='count')
                
                if len(timeline_data) > 0:
                    fig_timeline = px.line(
                        timeline_data,
                        x='time_rounded',
                        y='count',
                        color='category',
                        title="Detection Timeline",
                        labels={'time_rounded': 'Time', 'count': 'Detections'}
                    )
                    fig_timeline.update_layout(height=400)
                    st.plotly_chart(fig_timeline, use_container_width=True)
                else:
                    st.info("Insufficient data for timeline")
            else:
                st.info("Need more data points for timeline")
    
    def _render_detection_table(self, df):
        """Render a table showing recent detections"""
        st.markdown("### 📋 Recent Detections")
        
        # Select and format columns for display
        display_columns = ['timestamp', 'category', 'class_name', 'confidence', 'location']
        available_columns = [col for col in display_columns if col in df.columns]
        
        if available_columns:
            # Sort by timestamp (most recent first)
            if 'timestamp' in df.columns:
                df_display = df.sort_values('timestamp', ascending=False)
            else:
                df_display = df
            
            # Format confidence as percentage
            if 'confidence' in df_display.columns:
                df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x:.2%}")
            
            # Format timestamp
            if 'timestamp' in df_display.columns:
                df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Show last 20 detections
            st.dataframe(
                df_display[available_columns].head(20),
                use_container_width=True,
                hide_index=True
            )
            
            # Add filter options
            with st.expander("🔍 Filter Options"):
                col_filter1, col_filter2 = st.columns(2)
                
                with col_filter1:
                    if 'category' in df.columns:
                        selected_categories = st.multiselect(
                            "Filter by Category",
                            options=df['category'].unique(),
                            default=df['category'].unique()
                        )
                        
                        if selected_categories:
                            filtered_df = df[df['category'].isin(selected_categories)]
                        else:
                            filtered_df = df
                    else:
                        filtered_df = df
                
                with col_filter2:
                    if 'confidence' in df.columns:
                        min_confidence = st.slider(
                            "Minimum Confidence",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.0,
                            step=0.05
                        )
                        filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
                
                # Show filtered results
                if len(filtered_df) != len(df):
                    st.markdown(f"**Filtered Results:** {len(filtered_df)} of {len(df)} detections")
                    
                    if len(filtered_df) > 0:
                        if 'confidence' in filtered_df.columns:
                            filtered_df['confidence'] = filtered_df['confidence'].apply(lambda x: f"{x:.2%}")
                        if 'timestamp' in filtered_df.columns:
                            filtered_df['timestamp'] = filtered_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        st.dataframe(
                            filtered_df[available_columns].head(20),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No detections match the selected filters")
        else:
            st.info("No detection data available for display")
    
    def render_live_stats(self, current_frame_detections):
        """
        Render live statistics for the current frame
        
        Args:
            current_frame_detections: List of detections from the current frame
        """
        if not current_frame_detections:
            return
        
        # Create metrics for current frame
        st.markdown("#### 🔴 Live Frame Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Objects Detected", len(current_frame_detections))
        
        with col2:
            if current_frame_detections:
                avg_conf = np.mean([det['confidence'] for det in current_frame_detections])
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
            else:
                st.metric("Avg Confidence", "0%")
        
        with col3:
            alert_objects = sum(1 for det in current_frame_detections 
                              if det.get('category') in ['Graffiti', 'Posters'])
            st.metric("Alert Objects", alert_objects, delta_color="inverse")
        
        # Show current detections in a compact format
        if current_frame_detections:
            categories = [det['category'] for det in current_frame_detections]
            category_counts = Counter(categories)
            
            st.markdown("**Current Frame Detections:**")
            for category, count in category_counts.items():
                st.write(f"• {category}: {count}")
