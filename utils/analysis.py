"""
Traffic Analysis Module
Analyzes traffic density, congestion levels, and signal timing recommendations
"""

from config_simple import Config
from datetime import datetime


class TrafficAnalyzer:
    """Analyzes traffic data and provides recommendations"""
    
    def __init__(self):
        self.max_capacity = Config.MAX_ROAD_CAPACITY
        self.low_threshold = Config.LOW_TRAFFIC_THRESHOLD
        self.medium_threshold = Config.MEDIUM_TRAFFIC_THRESHOLD
        self.signal_timings = Config.SIGNAL_TIMINGS
    
    def analyze_traffic(self, vehicle_data):
        """
        Analyze traffic from vehicle detection data
        
        Args:
            vehicle_data: dict with vehicle counts
            
        Returns:
            dict with analysis results
        """
        # Calculate total vehicles
        total_vehicles = sum(vehicle_data.values())
        
        # Calculate traffic density percentage
        traffic_density = (total_vehicles / self.max_capacity) * 100
        traffic_density = min(traffic_density, 100)  # Cap at 100%
        
        # Classify traffic level
        traffic_level = self._classify_traffic(traffic_density)
        
        # Get signal timing recommendation
        signal_time = self.signal_timings.get(traffic_level, 40)
        
        # Estimate waiting time (in seconds)
        estimated_waiting_time = self._estimate_waiting_time(
            traffic_density, 
            total_vehicles
        )
        
        # Generate AI decision
        ai_decision = self._generate_ai_decision(
            traffic_level, 
            traffic_density, 
            total_vehicles,
            signal_time
        )
        
        # Prepare analysis result
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_vehicles': total_vehicles,
            'vehicle_breakdown': vehicle_data,
            'traffic_density_percentage': round(traffic_density, 2),
            'traffic_level': traffic_level,
            'signal_time_recommendation': signal_time,
            'estimated_waiting_time': estimated_waiting_time,
            'ai_decision': ai_decision,
            'detailed_metrics': self._calculate_detailed_metrics(
                vehicle_data, 
                traffic_density
            )
        }
        
        return analysis
    
    def _classify_traffic(self, density):
        """Classify traffic level based on density"""
        if density <= self.low_threshold:
            return 'low'
        elif density <= self.medium_threshold:
            return 'medium'
        else:
            return 'heavy'
    
    def _estimate_waiting_time(self, density, total_vehicles):
        """Estimate average waiting time for vehicles"""
        # Simplified formula: more vehicles = longer wait
        # Base waiting time is proportional to density
        if total_vehicles == 0:
            return 0
        
        base_wait = 10  # seconds
        density_factor = (density / 100) * 50  # Up to 50 additional seconds
        
        estimated_wait = base_wait + density_factor
        return round(estimated_wait, 1)
    
    def _generate_ai_decision(self, level, density, total_vehicles, signal_time):
        """Generate AI recommendation message"""
        message = f"{level.capitalize()} congestion detected. "
        message += f"Total vehicles: {total_vehicles}. "
        message += f"Traffic density: {density:.1f}%. "
        message += f"Recommended green signal duration: {signal_time} seconds."
        
        if level == 'heavy':
            message += " ⚠️ High traffic - Consider alternate routes if possible."
        elif level == 'medium':
            message += " Expected moderate traffic flow."
        else:
            message += " ✓ Traffic flowing smoothly."
        
        return message
    
    def _calculate_detailed_metrics(self, vehicle_data, density):
        """Calculate detailed traffic metrics"""
        total = sum(vehicle_data.values())
        
        metrics = {
            'cars_percentage': round((vehicle_data.get('car', 0) / total * 100) if total > 0 else 0, 2),
            'bikes_percentage': round((vehicle_data.get('bike', 0) / total * 100) if total > 0 else 0, 2),
            'trucks_percentage': round((vehicle_data.get('truck', 0) / total * 100) if total > 0 else 0, 2),
            'buses_percentage': round((vehicle_data.get('bus', 0) / total * 100) if total > 0 else 0, 2),
            'other_percentage': round((vehicle_data.get('other', 0) / total * 100) if total > 0 else 0, 2),
        }
        
        return metrics
    
    def analyze_comparison(self, analysis1, analysis2):
        """Compare two traffic analysis results"""
        density_change = analysis2['traffic_density_percentage'] - analysis1['traffic_density_percentage']
        vehicle_change = analysis2['total_vehicles'] - analysis1['total_vehicles']
        
        return {
            'density_change': round(density_change, 2),
            'vehicle_change': vehicle_change,
            'traffic_level_change': f"{analysis1['traffic_level']} → {analysis2['traffic_level']}"
        }
