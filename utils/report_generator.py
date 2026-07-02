"""
Report Generation Module
Generates PDF and CSV reports for traffic analysis results
"""

import csv
import os
from datetime import datetime
from io import BytesIO
import base64

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class ReportGenerator:
    """Generates PDF and CSV reports for traffic analysis"""
    
    def __init__(self):
        self.report_timestamp = datetime.now()
    
    def generate_csv_report(self, analysis_data, image_path, output_path):
        """
        Generate CSV report
        
        Args:
            analysis_data: Traffic analysis results
            image_path: Path to processed image
            output_path: Where to save the CSV file
            
        Returns:
            output_path if successful
        """
        try:
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header section
                writer.writerow(['TRAFFIC ANALYSIS REPORT'])
                writer.writerow(['Generated:', self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow([])
                
                # Summary section
                writer.writerow(['TRAFFIC SUMMARY'])
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Total Vehicles', analysis_data['total_vehicles']])
                writer.writerow(['Traffic Density (%)', analysis_data['traffic_density_percentage']])
                writer.writerow(['Traffic Level', analysis_data['traffic_level'].upper()])
                writer.writerow(['Suggested Signal Time (sec)', analysis_data['signal_time_recommendation']])
                writer.writerow(['Estimated Waiting Time (sec)', analysis_data['estimated_waiting_time']])
                writer.writerow([])
                
                # Vehicle breakdown
                writer.writerow(['VEHICLE BREAKDOWN'])
                writer.writerow(['Vehicle Type', 'Count'])
                for vehicle_type, count in analysis_data['vehicle_breakdown'].items():
                    writer.writerow([vehicle_type.upper(), count])
                writer.writerow([])
                
                # Detailed metrics
                writer.writerow(['TRAFFIC COMPOSITION'])
                writer.writerow(['Vehicle Type', 'Percentage'])
                for metric, value in analysis_data['detailed_metrics'].items():
                    metric_label = metric.replace('_', ' ').upper()
                    writer.writerow([metric_label, f"{value}%"])
                writer.writerow([])
                
                # AI Decision
                writer.writerow(['AI RECOMMENDATION'])
                writer.writerow([analysis_data['ai_decision']])
                
            return output_path
            
        except Exception as e:
            print(f"Error generating CSV report: {e}")
            raise
    
    def generate_pdf_report(self, analysis_data, image_path, output_path):
        """
        Generate PDF report
        
        Args:
            analysis_data: Traffic analysis results
            image_path: Path to processed image
            output_path: Where to save the PDF file
            
        Returns:
            output_path if successful
        """
        if not HAS_REPORTLAB:
            raise ImportError("reportlab is not installed. Install it with: pip install reportlab")
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#00D4FF'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#00D4FF'),
                spaceAfter=12
            )
            
            # Add title
            story.append(Paragraph("🚗 TRAFFIC ANALYSIS REPORT", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            # Add timestamp
            timestamp_text = f"Generated: {self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            story.append(Paragraph(timestamp_text, styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
            
            # Summary section
            story.append(Paragraph("TRAFFIC SUMMARY", heading_style))
            summary_data = [
                ['Metric', 'Value'],
                ['Total Vehicles', str(analysis_data['total_vehicles'])],
                ['Traffic Density', f"{analysis_data['traffic_density_percentage']}%"],
                ['Traffic Level', analysis_data['traffic_level'].upper()],
                ['Suggested Signal Time', f"{analysis_data['signal_time_recommendation']} sec"],
                ['Estimated Waiting Time', f"{analysis_data['estimated_waiting_time']} sec"],
            ]
            
            summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00D4FF')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Vehicle breakdown
            story.append(Paragraph("VEHICLE BREAKDOWN", heading_style))
            vehicle_data = [['Vehicle Type', 'Count']]
            for vehicle_type, count in analysis_data['vehicle_breakdown'].items():
                vehicle_data.append([vehicle_type.upper(), str(count)])
            
            vehicle_table = Table(vehicle_data, colWidths=[3*inch, 2*inch])
            vehicle_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00D4FF')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(vehicle_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Add processed image if available
            if image_path and os.path.exists(image_path):
                try:
                    story.append(Paragraph("PROCESSED IMAGE", heading_style))
                    img = Image(image_path, width=5*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.2*inch))
                except Exception as e:
                    print(f"Could not add image to PDF: {e}")
            
            # AI Recommendation
            story.append(PageBreak())
            story.append(Paragraph("AI RECOMMENDATION", heading_style))
            story.append(Paragraph(analysis_data['ai_decision'], styles['BodyText']))
            
            # Build PDF
            doc.build(story)
            return output_path
            
        except Exception as e:
            print(f"Error generating PDF report: {e}")
            raise
    
    def generate_json_report(self, analysis_data, image_base64=None):
        """
        Generate JSON report data
        
        Args:
            analysis_data: Traffic analysis results
            image_base64: Base64 encoded image (optional)
            
        Returns:
            dict with report data
        """
        report = {
            'timestamp': self.report_timestamp.isoformat(),
            'summary': {
                'total_vehicles': analysis_data['total_vehicles'],
                'traffic_density_percentage': analysis_data['traffic_density_percentage'],
                'traffic_level': analysis_data['traffic_level'],
                'signal_time_recommendation': analysis_data['signal_time_recommendation'],
                'estimated_waiting_time': analysis_data['estimated_waiting_time'],
            },
            'vehicle_breakdown': analysis_data['vehicle_breakdown'],
            'detailed_metrics': analysis_data['detailed_metrics'],
            'ai_decision': analysis_data['ai_decision'],
            'image': image_base64
        }
        
        return report
