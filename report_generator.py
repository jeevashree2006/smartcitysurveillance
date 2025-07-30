import pandas as pd
import json
import csv
from datetime import datetime, timedelta
import io
from collections import Counter, defaultdict

class ReportGenerator:
    def __init__(self):
        """Initialize the report generator"""
        pass
    
    def generate_csv_report(self, detection_logs):
        """
        Generate a CSV report from detection logs
        
        Args:
            detection_logs: List of detection dictionaries
            
        Returns:
            CSV data as string
        """
        if not detection_logs:
            return "No detection data available"
        
        # Convert detection logs to DataFrame
        df = pd.DataFrame(detection_logs)
        
        # Clean and format the data
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['time'] = df['timestamp'].dt.time
            df['hour'] = df['timestamp'].dt.hour
        
        # Add additional computed columns
        if 'bbox' in df.columns:
            df['bbox_area'] = df['bbox'].apply(lambda x: (x[2] - x[0]) * (x[3] - x[1]) if isinstance(x, list) and len(x) == 4 else 0)
        
        # Reorder columns for better readability
        column_order = ['timestamp', 'date', 'time', 'category', 'class_name', 'confidence', 'location', 'bbox', 'bbox_area']
        available_columns = [col for col in column_order if col in df.columns]
        remaining_columns = [col for col in df.columns if col not in column_order]
        final_columns = available_columns + remaining_columns
        
        df = df[final_columns]
        
        # Convert to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        return output.getvalue()
    
    def generate_analytics_report(self, detection_logs):
        """
        Generate an analytics report with statistics and insights
        
        Args:
            detection_logs: List of detection dictionaries
            
        Returns:
            JSON data as string
        """
        if not detection_logs:
            return json.dumps({"error": "No detection data available"}, indent=2)
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(detection_logs)
        
        # Parse timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Generate analytics
        analytics = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_detections": len(detection_logs),
                "date_range": {
                    "start": df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                    "end": df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
                }
            },
            "detection_summary": {},
            "category_analysis": {},
            "temporal_analysis": {},
            "location_analysis": {},
            "confidence_analysis": {},
            "alert_analysis": {}
        }
        
        # Detection summary by category
        if 'category' in df.columns:
            category_counts = df['category'].value_counts().to_dict()
            analytics["detection_summary"]["by_category"] = category_counts
            
            # Calculate percentages
            total = len(df)
            category_percentages = {k: round(v/total*100, 2) for k, v in category_counts.items()}
            analytics["detection_summary"]["category_percentages"] = category_percentages
        
        # Category analysis with confidence scores
        if 'category' in df.columns and 'confidence' in df.columns:
            category_analysis = {}
            for category in df['category'].unique():
                cat_data = df[df['category'] == category]
                category_analysis[category] = {
                    "count": len(cat_data),
                    "avg_confidence": round(cat_data['confidence'].mean(), 3),
                    "max_confidence": round(cat_data['confidence'].max(), 3),
                    "min_confidence": round(cat_data['confidence'].min(), 3)
                }
            analytics["category_analysis"] = category_analysis
        
        # Temporal analysis
        if 'timestamp' in df.columns:
            # Hourly distribution
            df['hour'] = df['timestamp'].dt.hour
            hourly_counts = df['hour'].value_counts().sort_index().to_dict()
            analytics["temporal_analysis"]["hourly_distribution"] = hourly_counts
            
            # Daily distribution
            df['date'] = df['timestamp'].dt.date
            daily_counts = df['date'].value_counts().sort_index()
            daily_counts.index = daily_counts.index.astype(str)  # Convert date to string for JSON serialization
            analytics["temporal_analysis"]["daily_distribution"] = daily_counts.to_dict()
            
            # Peak hours
            peak_hour = df['hour'].value_counts().index[0] if len(df) > 0 else None
            analytics["temporal_analysis"]["peak_hour"] = int(peak_hour) if peak_hour is not None else None
        
        # Location analysis
        if 'location' in df.columns:
            location_counts = df['location'].value_counts().to_dict()
            analytics["location_analysis"]["detections_by_location"] = location_counts
        
        # Confidence analysis
        if 'confidence' in df.columns:
            confidence_stats = {
                "average": round(df['confidence'].mean(), 3),
                "median": round(df['confidence'].median(), 3),
                "std_deviation": round(df['confidence'].std(), 3),
                "min": round(df['confidence'].min(), 3),
                "max": round(df['confidence'].max(), 3)
            }
            
            # Confidence distribution
            confidence_bins = pd.cut(df['confidence'], bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0], 
                                   labels=['Very Low (0-0.3)', 'Low (0.3-0.5)', 'Medium (0.5-0.7)', 
                                          'High (0.7-0.9)', 'Very High (0.9-1.0)'])
            confidence_distribution = confidence_bins.value_counts().to_dict()
            confidence_distribution = {str(k): int(v) for k, v in confidence_distribution.items()}
            
            analytics["confidence_analysis"]["statistics"] = confidence_stats
            analytics["confidence_analysis"]["distribution"] = confidence_distribution
        
        # Alert analysis (high-priority detections)
        high_confidence_threshold = 0.7
        if 'confidence' in df.columns and 'category' in df.columns:
            high_confidence_detections = df[df['confidence'] >= high_confidence_threshold]
            
            alert_categories = ['Graffiti', 'Posters']  # Categories that trigger alerts
            alert_detections = df[df['category'].isin(alert_categories)]
            
            analytics["alert_analysis"] = {
                "high_confidence_detections": len(high_confidence_detections),
                "alert_worthy_detections": len(alert_detections),
                "alert_categories": alert_categories,
                "high_confidence_threshold": high_confidence_threshold
            }
            
            if len(alert_detections) > 0:
                alert_by_category = alert_detections['category'].value_counts().to_dict()
                analytics["alert_analysis"]["alerts_by_category"] = alert_by_category
        
        # Trend analysis (if we have data spanning multiple days)
        if 'timestamp' in df.columns and len(df['date'].unique()) > 1:
            daily_trends = df.groupby('date').size()
            trend_direction = "increasing" if daily_trends.iloc[-1] > daily_trends.iloc[0] else "decreasing"
            analytics["temporal_analysis"]["trend_direction"] = trend_direction
            analytics["temporal_analysis"]["trend_change"] = int(daily_trends.iloc[-1] - daily_trends.iloc[0])
        
        # Recommendations based on data
        recommendations = self._generate_recommendations(analytics, df)
        analytics["recommendations"] = recommendations
        
        return json.dumps(analytics, indent=2)
    
    def _generate_recommendations(self, analytics, df):
        """Generate actionable recommendations based on analytics"""
        recommendations = []
        
        try:
            # Recommendation based on peak hours
            if "temporal_analysis" in analytics and "peak_hour" in analytics["temporal_analysis"]:
                peak_hour = analytics["temporal_analysis"]["peak_hour"]
                if peak_hour is not None:
                    recommendations.append({
                        "type": "temporal",
                        "priority": "medium",
                        "message": f"Peak detection activity occurs at hour {peak_hour}:00. Consider increasing surveillance during this time."
                    })
            
            # Recommendation based on high-alert categories
            if "detection_summary" in analytics and "by_category" in analytics["detection_summary"]:
                categories = analytics["detection_summary"]["by_category"]
                
                if "Graffiti" in categories and categories["Graffiti"] > 5:
                    recommendations.append({
                        "type": "security",
                        "priority": "high",
                        "message": f"High graffiti activity detected ({categories['Graffiti']} instances). Consider increased patrols and preventive measures."
                    })
                
                if "Posters" in categories and categories["Posters"] > 3:
                    recommendations.append({
                        "type": "maintenance",
                        "priority": "medium",
                        "message": f"Multiple poster detections ({categories['Posters']} instances). Schedule cleanup and poster removal."
                    })
            
            # Recommendation based on confidence levels
            if "confidence_analysis" in analytics and "statistics" in analytics["confidence_analysis"]:
                avg_confidence = analytics["confidence_analysis"]["statistics"]["average"]
                
                if avg_confidence < 0.6:
                    recommendations.append({
                        "type": "technical",
                        "priority": "medium",
                        "message": f"Average detection confidence is low ({avg_confidence:.2f}). Consider adjusting camera positioning or detection thresholds."
                    })
            
            # Recommendation based on location distribution
            if "location_analysis" in analytics and len(analytics["location_analysis"].get("detections_by_location", {})) == 1:
                recommendations.append({
                    "type": "coverage",
                    "priority": "low",
                    "message": "All detections are from a single location. Consider expanding surveillance coverage to other areas."
                })
            
            # General recommendations
            total_detections = analytics.get("report_metadata", {}).get("total_detections", 0)
            if total_detections > 50:
                recommendations.append({
                    "type": "analysis",
                    "priority": "low",
                    "message": f"High detection volume ({total_detections} total). Consider implementing automated response workflows."
                })
            
        except Exception as e:
            recommendations.append({
                "type": "system",
                "priority": "low",
                "message": f"Unable to generate all recommendations due to data analysis error: {str(e)}"
            })
        
        return recommendations
    
    def export_detection_logs(self, detection_logs, format_type="csv"):
        """
        Export detection logs in the specified format
        
        Args:
            detection_logs: List of detection dictionaries
            format_type: "csv" or "json"
            
        Returns:
            Formatted data as string
        """
        if format_type.lower() == "csv":
            return self.generate_csv_report(detection_logs)
        elif format_type.lower() == "json":
            return json.dumps(detection_logs, indent=2, default=str)
        else:
            raise ValueError("Unsupported format type. Use 'csv' or 'json'")
    
    def save_logs_to_file(self, detection_logs, filename=None):
        """
        Save detection logs to a file
        
        Args:
            detection_logs: List of detection dictionaries
            filename: Optional filename, will be generated if not provided
            
        Returns:
            Filename of the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"surveillance_logs_{timestamp}.csv"
        
        csv_data = self.generate_csv_report(detection_logs)
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_data)
        
        return filename
