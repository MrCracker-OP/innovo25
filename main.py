# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from datetime import datetime
import uuid
import json
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def load_emissions_data():
    data_file = os.path.join(app.config['DATA_FOLDER'], 'emissions.json')
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            return json.load(f)
    return []


def save_emissions_data(data):
    data_file = os.path.join(app.config['DATA_FOLDER'], 'emissions.json')
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)


def load_emission_factors():
    return {
        'Diesel': 2.9,  # kg CO₂/L
        'Petrol': 2.6,  # kg CO₂/L
        'Natural Gas': 2.2,  # kg CO₂/m³
        'Coal': 2.7,  # kg CO₂/kg
        'LPG': 1.8,  # kg CO₂/L
        'Grid Electricity': 0.5,  # kg CO₂/kWh
        'Heavy Goods Vehicle': 0.15,  # kg CO₂/ton-km
        'Rail Transport': 0.045  # kg CO₂/ton-km
    }


@app.route('/')
def index():
    return "HELLO WORLD"


@app.route('/api/emissions', methods=['GET'])
def get_emissions():
    emissions = load_emissions_data()

    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    business_unit = request.args.get('business_unit')

    filtered_emissions = emissions

    if start_date:
        filtered_emissions = [e for e in filtered_emissions if e['date'] >= start_date]

    if end_date:
        filtered_emissions = [e for e in filtered_emissions if e['date'] <= end_date]

    if business_unit and business_unit != 'All Units':
        filtered_emissions = [e for e in filtered_emissions if e['business_unit'] == business_unit]

    return jsonify(filtered_emissions)


@app.route('/api/emissions', methods=['POST'])
def add_emission():
    data = request.json
    emissions = load_emissions_data()

    data['id'] = str(uuid.uuid4())
    data['created_at'] = datetime.now().isoformat()

    emissions.append(data)
    save_emissions_data(emissions)

    return jsonify(data), 201


@app.route('/api/emissions/<emission_id>', methods=['DELETE'])
def delete_emission(emission_id):
    emissions = load_emissions_data()
    emissions = [e for e in emissions if e['id'] != emission_id]
    save_emissions_data(emissions)
    return jsonify({'success': True})


@app.route('/api/emissions/<emission_id>', methods=['PUT'])
def update_emission(emission_id):
    data = request.json
    emissions = load_emissions_data()

    for i, emission in enumerate(emissions):
        if emission['id'] == emission_id:
            emissions[i].update(data)
            emissions[i]['updated_at'] = datetime.now().isoformat()
            save_emissions_data(emissions)
            return jsonify(emissions[i])

    return jsonify({'error': 'Emission not found'}), 404


@app.route('/api/upload/csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the uploaded file
        try:
            # Determine file type and read accordingly
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:  # Excel files
                df = pd.read_excel(filepath)

            df = clean_emissions_data(df)
            new_emissions = df.to_dict('records')
            emissions = load_emissions_data()

            for emission in new_emissions:
                emission['id'] = str(uuid.uuid4())
                emission['created_at'] = datetime.now().isoformat()
                emissions.append(emission)

            save_emissions_data(emissions)

            return jsonify({
                'success': True,
                'message': f'Successfully processed {len(new_emissions)} records',
                'records_added': len(new_emissions)
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({'error': 'Invalid file type'}), 400


def clean_emissions_data(df):
    df = df.dropna(how='all')
    column_mapping = {
        'Activity Type': 'activity_type',
        'Business Unit': 'business_unit',
        'Date': 'date',
        'Quantity': 'quantity',
        'Unit': 'unit',
        'Source': 'source',
        'Fuel Type': 'fuel_type',
        'Electricity Source': 'electricity_source'
    }
    df = df.rename(columns={col: column_mapping.get(col, col) for col in df.columns})

    required_cols = ['activity_type', 'date']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

    df['business_unit'] = df['business_unit'].fillna('Not Specified')
    df['source'] = df['source'].fillna('Imported Data')

    return df


@app.route('/api/dashboard/summary', methods=['GET'])
def get_dashboard_summary():
    emissions = load_emissions_data()
    if not emissions:
        return jsonify({
            'total_emissions': 0,
            'emissions_by_scope': {'Scope 1': 0, 'Scope 2': 0, 'Scope 3': 0},
            'emissions_by_business_unit': {},
            'emissions_trend': [],
            'business_unit_trend': {}
        })

    scope_filter = request.args.get('scope')
    business_unit_filter = request.args.get('business_unit')

    filtered_emissions = emissions

    if business_unit_filter:
        filtered_emissions = [e for e in filtered_emissions if e.get('business_unit') == business_unit_filter]

    emission_factors = load_emission_factors()
    total_emissions = 0
    emissions_by_scope = {'Scope 1': 0, 'Scope 2': 0, 'Scope 3': 0}
    emissions_by_business_unit = {}
    emissions_by_month = {}
    business_unit_emissions_by_month = {}

    for e in filtered_emissions:
        quantity = float(e.get('quantity', 0))
        unit = e.get('unit', 'unknown').lower()
        activity_type = e.get('activity_type', '').lower()
        fuel_type = e.get('fuel_type', '').strip().title()
        electricity_source = e.get('electricity_source', '').strip().title()
        
        # Determine emission factor
        if fuel_type:
            factor = emission_factors.get(fuel_type, 1.0)
        elif electricity_source:
            factor = emission_factors.get(electricity_source, 1.0)
        else:
            factor = emission_factors.get(activity_type, 1.0)

        if unit in ['liters', 'm3', 'kg', 'kwh', 'ton-km']:
            emission = quantity * factor
        else:
            emission = 0

        # Categorize by scope
        if activity_type in ['vehicle_fleet', 'natural_gas', 'stationary_combustion', 'heating'] or fuel_type:
            scope = 'Scope 1'
        elif activity_type == 'electricity' or electricity_source:
            scope = 'Scope 2'
        else:
            scope = 'Scope 3'

        if scope_filter and scope != scope_filter:
            continue

        emissions_by_scope[scope] += emission
        total_emissions += emission

        # Business unit categorization
        business_unit = e.get('business_unit', 'Unknown')
        emissions_by_business_unit[business_unit] = emissions_by_business_unit.get(business_unit, 0) + emission

        # Trend analysis
        if 'date' in e:
            try:
                date = datetime.fromisoformat(e['date'].replace('Z', '+00:00'))
                month_key = date.strftime('%Y-%m')
                emissions_by_month[month_key] = emissions_by_month.get(month_key, 0) + emission

                if business_unit not in business_unit_emissions_by_month:
                    business_unit_emissions_by_month[business_unit] = {}
                business_unit_emissions_by_month[business_unit][month_key] = business_unit_emissions_by_month[business_unit].get(month_key, 0) + emission
            except (ValueError, AttributeError):
                continue

    emissions_trend = [{'month': k, 'emissions': v} for k, v in sorted(emissions_by_month.items())]
    business_unit_trend = {
        business_unit: [
            {'month': k, 'emissions': v}
            for k, v in sorted(months.items())
        ]
        for business_unit, months in business_unit_emissions_by_month.items()
    }

    return jsonify({
        'total_emissions': round(total_emissions, 2),
        'emissions_by_scope': {k: round(v, 2) for k, v in emissions_by_scope.items()},
        'emissions_by_business_unit': {k: round(v, 2) for k, v in emissions_by_business_unit.items()},
        'emissions_trend': emissions_trend,
        'business_unit_trend': business_unit_trend
    })


@app.route('/api/business-units', methods=['GET'])
def get_business_units():
    emissions = load_emissions_data()
    business_units = set(e.get('business_unit', 'Not Specified') for e in emissions)
    return jsonify(list(business_units))


@app.route('/api/activity-types', methods=['GET'])
def get_activity_types():
    emissions = load_emissions_data()
    activity_types = set(e.get('activity_type', 'Other') for e in emissions)
    return jsonify(list(activity_types))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)