# app.py
from flask import Flask, request, jsonify
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import json
from werkzeug.utils import secure_filename

# Configure Flask app
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable Cross-Origin Resource Sharing
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)

# Helper function to check allowed file extensions


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper function to load existing data


def load_emissions_data():
    data_file = os.path.join(app.config['DATA_FOLDER'], 'emissions.json')
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            return json.load(f)
    return []

# Helper function to save emissions data


def save_emissions_data(data):
    data_file = os.path.join(app.config['DATA_FOLDER'], 'emissions.json')
    with open(data_file, 'w') as f:
        json.dump(data, f, indent=2)

# Route for serving the frontend


@app.route('/')
def index():
    return "HELLO WORLD"

# API Routes


@app.route('/api/emissions', methods=['GET'])
def get_emissions():
    emissions = load_emissions_data()

    # Handle filtering
    scope = request.args.get('scope')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    business_unit = request.args.get('business_unit')

    filtered_emissions = emissions

    if scope and scope != 'All Scopes':
        filtered_emissions = [
            e for e in filtered_emissions if e['scope'] == scope]

    if start_date:
        filtered_emissions = [
            e for e in filtered_emissions if e['date'] >= start_date]

    if end_date:
        filtered_emissions = [
            e for e in filtered_emissions if e['date'] <= end_date]

    if business_unit and business_unit != 'All Units':
        filtered_emissions = [
            e for e in filtered_emissions if e['business_unit'] == business_unit]

    print(f"here with -> {emissions}")

    return jsonify(filtered_emissions)


@app.route('/api/emissions', methods=['POST'])
def add_emission():
    data = request.json
    emissions = load_emissions_data()

    # Generate unique ID
    data['id'] = str(uuid.uuid4())
    data['created_at'] = datetime.now().isoformat()

    # Calculate CO2e based on activity type and quantity
    # This is a simplified calculation - in a real app, you'd use more precise emission factors
    activity_type = data.get('activity_type')
    quantity = float(data.get('quantity', 0))
    unit = data.get('unit', '')

    # Sample emission factors (kg CO2e per unit)
    emission_factors = {
        'electricity': 0.5,  # kg CO2e per kWh
        'natural_gas': 2.1,  # kg CO2e per m3
        'vehicle_fleet': 2.3,  # kg CO2e per liter of fuel
        'purchased_goods': 1.0,  # generic factor per kg
    }

    # Calculate emissions
    factor = emission_factors.get(activity_type, 1.0)
    data['co2e'] = round(quantity * factor, 2)
    data['co2e_unit'] = 'kgCO₂e'

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

    # Find and update the emission entry
    for i, emission in enumerate(emissions):
        if emission['id'] == emission_id:
            # Recalculate CO2e if quantity or activity type changed
            if 'quantity' in data or 'activity_type' in data:
                activity_type = data.get(
                    'activity_type', emission.get('activity_type'))
                quantity = float(
                    data.get('quantity', emission.get('quantity', 0)))

                # Sample emission factors (same as in POST route)
                emission_factors = {
                    'electricity': 0.5,
                    'natural_gas': 2.1,
                    'vehicle_fleet': 2.3,
                    'purchased_goods': 1.0,
                }

                factor = emission_factors.get(activity_type, 1.0)
                data['co2e'] = round(quantity * factor, 2)

            # Update the emission with new data
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

            # Clean the data
            df = clean_emissions_data(df)

            # Convert to list of dicts and add to existing data
            new_emissions = df.to_dict('records')
            emissions = load_emissions_data()

            # Add IDs and timestamps to new records
            for emission in new_emissions:
                emission['id'] = str(uuid.uuid4())
                emission['created_at'] = datetime.now().isoformat()

                # Ensure all numeric fields are properly formatted
                if 'quantity' in emission:
                    emission['quantity'] = float(emission.get('quantity', 0))
                if 'co2e' in emission:
                    emission['co2e'] = float(emission.get('co2e', 0))

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
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

    return jsonify({'error': 'Invalid file type'}), 400


def clean_emissions_data(df):
    # Remove rows with all NaN values
    df = df.dropna(how='all')

    # Rename columns to match our schema if needed
    column_mapping = {
        'Activity Type': 'activity_type',
        'Business Unit': 'business_unit',
        'Date': 'date',
        'Scope': 'scope',
        'Quantity': 'quantity',
        'Unit': 'unit',
        'Source': 'source'
    }

    df = df.rename(columns={col: column_mapping.get(col, col)
                   for col in df.columns})

    # Ensure required columns exist
    required_cols = ['activity_type', 'date', 'scope']
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Convert date strings to ISO format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(
            df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

    # Fill missing values with defaults
    df['scope'] = df['scope'].fillna('Scope 3')
    df['business_unit'] = df['business_unit'].fillna('Not Specified')
    df['source'] = df['source'].fillna('Imported Data')

    # Calculate CO2e if not provided
    if 'co2e' not in df.columns and 'quantity' in df.columns:
        # Sample emission factors (as used elsewhere)
        emission_factors = {
            'electricity': 0.5,
            'natural_gas': 2.1,
            'vehicle_fleet': 2.3,
            'purchased_goods': 1.0,
        }

        def calculate_co2e(row):
            factor = emission_factors.get(row.get('activity_type', ''), 1.0)
            try:
                quantity = float(row.get('quantity', 0))
                return round(quantity * factor, 2)
            except (ValueError, TypeError):
                return 0

        df['co2e'] = df.apply(calculate_co2e, axis=1)
        df['co2e_unit'] = 'kgCO₂e'

    return df


@app.route('/api/dashboard/summary', methods=['GET'])
def get_dashboard_summary():
    # Get query parameters
    scope_filter = request.args.get('scope')
    business_unit_filter = request.args.get('business_unit')

    emissions = load_emissions_data()
    if not emissions:
        return jsonify({
            'total_emissions': 0,
            'emissions_by_scope': {'Scope 1': 0, 'Scope 2': 0, 'Scope 3': 0},
            'emissions_by_business_unit': {},
            'emissions_trend': [],
            'business_unit_trend': {}
        })

    # Filter emissions by scope
    if scope_filter:
        scope_filter = scope_filter.replace(' ', '')
        scope_filter = f"{scope_filter[:5]} {scope_filter[5:]}"
        emissions = [e for e in emissions if e.get('scope') == scope_filter]

    # Filter emissions by business unit
    if business_unit_filter:
        business_unit_filter = business_unit_filter.replace('+', ' ')
        emissions = [e for e in emissions if e.get(
            'business_unit') == business_unit_filter]

    # Calculate total emissions
    total_emissions = sum(e.get('co2e', 0) for e in emissions)

    # Group by scope
    emissions_by_scope = {'Scope 1': 0, 'Scope 2': 0, 'Scope 3': 0}
    for e in emissions:
        scope = e.get('scope', 'Scope 3')
        emissions_by_scope[scope] = emissions_by_scope.get(
            scope, 0) + e.get('co2e', 0)

    # Group by business unit
    emissions_by_business_unit = {}
    for e in emissions:
        business_unit = e.get('business_unit', 'Unknown')
        emissions_by_business_unit[business_unit] = emissions_by_business_unit.get(
            business_unit, 0) + e.get('co2e', 0)

    # Group by month for trend analysis
    emissions_by_month = {}
    business_unit_emissions_by_month = {}
    for e in emissions:
        if 'date' in e:
            try:
                date = datetime.fromisoformat(e['date'].replace('Z', '+00:00'))
                month_key = date.strftime('%Y-%m')
                business_unit = e.get('business_unit', 'Unknown')

                # Overall trend
                emissions_by_month[month_key] = emissions_by_month.get(
                    month_key, 0) + e.get('co2e', 0)

                # Business Unit-specific trend
                if business_unit not in business_unit_emissions_by_month:
                    business_unit_emissions_by_month[business_unit] = {}
                business_unit_emissions_by_month[business_unit][month_key] = business_unit_emissions_by_month[business_unit].get(
                    month_key, 0) + e.get('co2e', 0)
            except (ValueError, AttributeError):
                continue

    # Convert to sorted lists for frontend
    emissions_trend = [
        {'month': k, 'emissions': v}
        for k, v in sorted(emissions_by_month.items())
    ]

    # Format business unit trends
    business_unit_trend = {
        business_unit: [
            {'month': k, 'emissions': v}
            for k, v in sorted(months.items())
        ]
        for business_unit, months in business_unit_emissions_by_month.items()
    }

    return jsonify({
        'total_emissions': round(total_emissions, 2),
        'emissions_by_scope': emissions_by_scope,
        'emissions_by_business_unit': emissions_by_business_unit,
        'emissions_trend': emissions_trend,
        'business_unit_trend': business_unit_trend
    })


@app.route('/api/business-units', methods=['GET'])
def get_business_units():
    emissions = load_emissions_data()
    business_units = set(e.get('business_unit', 'Not Specified')
                         for e in emissions)
    return jsonify(list(business_units))


@app.route('/api/activity-types', methods=['GET'])
def get_activity_types():
    emissions = load_emissions_data()
    activity_types = set(e.get('activity_type', 'Other') for e in emissions)
    return jsonify(list(activity_types))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
