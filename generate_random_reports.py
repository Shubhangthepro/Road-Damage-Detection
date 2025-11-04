"""
Script to generate random reports from road_damage_dataset
Only uses images from 'poor' and 'very_poor' folders
"""
import os
import random
import shutil
from datetime import datetime, timedelta
from pymongo import MongoClient
from werkzeug.security import generate_password_hash
import requests
from urllib.parse import urlencode

# MongoDB Configuration
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/roadsight')
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.get_default_database()
reports_col = db.get_collection('reports')
users_col = db.get_collection('users')

# Paths
DATASET_PATH = 'road_damage_dataset'
UPLOAD_FOLDER = 'static/uploads'
POOR_FOLDER = os.path.join(DATASET_PATH, 'poor')
VERY_POOR_FOLDER = os.path.join(DATASET_PATH, 'very_poor')

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Indian cities with coordinates (for realistic locations)
INDIAN_LOCATIONS = [
    {'city': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777, 'address': 'Mumbai, Maharashtra'},
    {'city': 'Delhi', 'lat': 28.6139, 'lon': 77.2090, 'address': 'New Delhi, Delhi'},
    {'city': 'Bangalore', 'lat': 12.9716, 'lon': 77.5946, 'address': 'Bangalore, Karnataka'},
    {'city': 'Hyderabad', 'lat': 17.3850, 'lon': 78.4867, 'address': 'Hyderabad, Telangana'},
    {'city': 'Chennai', 'lat': 13.0827, 'lon': 80.2707, 'address': 'Chennai, Tamil Nadu'},
    {'city': 'Kolkata', 'lat': 22.5726, 'lon': 88.3639, 'address': 'Kolkata, West Bengal'},
    {'city': 'Pune', 'lat': 18.5204, 'lon': 73.8567, 'address': 'Pune, Maharashtra'},
    {'city': 'Ahmedabad', 'lat': 23.0225, 'lon': 72.5714, 'address': 'Ahmedabad, Gujarat'},
    {'city': 'Jaipur', 'lat': 26.9124, 'lon': 75.7873, 'address': 'Jaipur, Rajasthan'},
    {'city': 'Surat', 'lat': 21.1702, 'lon': 72.8311, 'address': 'Surat, Gujarat'},
]

# Sample reporter names
REPORTER_NAMES = [
    'Rajesh Kumar', 'Priya Sharma', 'Amit Patel', 'Sneha Reddy', 'Vikram Singh',
    'Anjali Gupta', 'Rohit Mehta', 'Kavita Nair', 'Suresh Iyer', 'Deepa Joshi',
    'Manoj Desai', 'Sunita Rao', 'Kiran Malhotra', 'Pooja Agarwal', 'Nitin Verma'
]

# Sample descriptions
DESCRIPTIONS = [
    'Large pothole causing vehicle damage. Urgent repair needed.',
    'Severe road damage with multiple cracks. Safety hazard.',
    'Road surface completely deteriorated. Immediate attention required.',
    'Deep potholes causing accidents. Need urgent repair.',
    'Road in very poor condition. Multiple vehicles stuck.',
    'Severe damage to road infrastructure. High priority repair needed.',
    'Road surface broken and dangerous. Public safety concern.',
    'Extensive damage to road. Multiple complaints received.',
    'Road condition critical. Immediate repair required.',
    'Severe deterioration of road surface. Vehicles at risk.'
]

def compute_predictive_risk(latitude, longitude):
    """Compute weather-based predictive risk"""
    try:
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'hourly': 'precipitation,temperature_2m',
            'forecast_days': 1
        }
        url = f"https://api.open-meteo.com/v1/forecast?{urlencode(params)}"
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        data = r.json()
        precip = sum(data.get('hourly', {}).get('precipitation', [])[:12])
        temp_avg = 0.0
        temps = data.get('hourly', {}).get('temperature_2m', [])[:12]
        if temps:
            temp_avg = sum(temps) / len(temps)
        rain_component = min(precip / 10.0, 1.0)
        temp_component = 0.0
        if temp_avg <= 0 or temp_avg >= 38:
            temp_component = 0.2
        return max(0.0, min(rain_component + temp_component, 1.0))
    except Exception:
        return round(random.uniform(0.1, 0.8), 2)  # Random fallback

def compute_density(latitude, longitude, category, report_id=None, radius_m=500):
    """Compute report density"""
    try:
        all_reports = reports_col.find({
            'category': category,
            'location.geo': {'$exists': True}
        })
        count = 0
        for report in all_reports:
            if report_id and str(report['_id']) == str(report_id):
                continue
            loc = report.get('location', {}).get('geo', {}).get('coordinates')
            if loc and len(loc) == 2:
                from math import radians, cos, sin, asin, sqrt
                lat2, lon2 = radians(loc[1]), radians(loc[0])
                lat1, lon1 = radians(latitude), radians(longitude)
                dlon, dlat = lon2 - lon1, lat2 - lat1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                distance = 6371 * c * 1000
                if distance <= radius_m:
                    count += 1
        return count
    except Exception:
        return random.randint(0, 5)  # Random fallback

def recompute_priority(severity_score, predictive_risk, density_count):
    """Recompute priority"""
    if severity_score >= 0.9:  # critical
        return 'High', 0.9
    priority_score = (severity_score * 0.6) + (predictive_risk * 0.25) + (min(density_count, 10) / 10.0 * 0.15)
    if priority_score >= 0.55:
        return 'High', priority_score
    elif priority_score >= 0.25:
        return 'Medium', priority_score
    return 'Low', priority_score

def generate_random_reports(num_reports=20):
    """Generate random reports"""
    
    # Get all images from poor and very_poor folders
    poor_images = []
    very_poor_images = []
    
    if os.path.exists(POOR_FOLDER):
        poor_images = [f for f in os.listdir(POOR_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if os.path.exists(VERY_POOR_FOLDER):
        very_poor_images = [f for f in os.listdir(VERY_POOR_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    all_images = []
    for img in poor_images:
        all_images.append(('poor', os.path.join(POOR_FOLDER, img)))
    for img in very_poor_images:
        all_images.append(('very_poor', os.path.join(VERY_POOR_FOLDER, img)))
    
    if not all_images:
        print("❌ No images found in poor or very_poor folders!")
        return
    
    print(f"✅ Found {len(all_images)} images. Generating {num_reports} reports...\n")
    
    # Create a test user if doesn't exist
    test_email = 'test@roadsight.local'
    if not users_col.find_one({'email': test_email}):
        users_col.insert_one({
            'email': test_email,
            'passwordHash': generate_password_hash('test123'),
            'name': 'Test User',
            'role': 'user',
            'createdAt': datetime.utcnow()
        })
        print(f"✅ Created test user: {test_email}")
    
    generated = 0
    for i in range(num_reports):
        try:
            # Select random image
            condition, image_path = random.choice(all_images)
            
            # Copy image to uploads folder
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"road_{timestamp}_{os.path.basename(image_path)}"
            dest_path = os.path.join(UPLOAD_FOLDER, filename)
            shutil.copy2(image_path, dest_path)
            
            # Random location
            location = random.choice(INDIAN_LOCATIONS)
            lat = location['lat'] + random.uniform(-0.1, 0.1)  # Add some variation
            lon = location['lon'] + random.uniform(-0.1, 0.1)
            address = f"{random.choice(['Main Road', 'Highway', 'Street', 'Avenue'])} near {location['address']}"
            
            # Map condition to severity
            severity_map = {
                'poor': {'level': 'poor', 'score': 0.7},
                'very_poor': {'level': 'critical', 'score': 0.9}
            }
            severity_info = severity_map.get(condition, {'level': 'poor', 'score': 0.7})
            
            # Random data
            reporter_name = random.choice(REPORTER_NAMES)
            reporter_email = f"{reporter_name.lower().replace(' ', '.')}@example.com"
            description = random.choice(DESCRIPTIONS)
            confidence = round(random.uniform(75, 95), 2)
            
            # Random date within last 30 days
            days_ago = random.randint(0, 30)
            created_at = datetime.utcnow() - timedelta(days=days_ago)
            
            # Create report document
            doc = {
                'imageUrl': f'/static/uploads/{filename}',
                'location': {
                    'address': address,
                    'latitude': lat,
                    'longitude': lon,
                    'geo': {'type': 'Point', 'coordinates': [lon, lat]}
                },
                'category': 'RoadDamage',
                'condition': condition.replace('_', ' ').title(),
                'confidence': confidence,
                'severity': severity_info,
                'predictiveRisk': 0.0,
                'reportDensity': 0,
                'priority': 'Medium',
                'status': random.choice(['New', 'Scheduled', 'In Progress', 'Resolved']),
                'reporter': {
                    'name': reporter_name,
                    'email': reporter_email
                },
                'description': description,
                'createdAt': created_at,
                'updatedAt': created_at
            }
            
            # Insert report
            inserted = reports_col.insert_one(doc)
            
            # Compute risk and density
            predictive_risk = compute_predictive_risk(lat, lon)
            density_count = compute_density(lat, lon, doc['category'], inserted.inserted_id)
            priority, score = recompute_priority(severity_info['score'], predictive_risk, density_count)
            
            # Update with computed values
            reports_col.update_one({'_id': inserted.inserted_id}, {'$set': {
                'predictiveRisk': predictive_risk,
                'reportDensity': density_count,
                'priority': priority,
                'priorityScore': score,
                'updatedAt': datetime.utcnow()
            }})
            
            generated += 1
            print(f"✅ [{generated}/{num_reports}] Generated report: {condition} at {address}")
            
        except Exception as e:
            print(f"❌ Error generating report {i+1}: {e}")
            continue
    
    print(f"\n✅ Successfully generated {generated} reports!")
    print(f"📊 Check admin dashboard at http://localhost:5000/admin/login")

if __name__ == '__main__':
    import sys
    num_reports = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    generate_random_reports(num_reports)

