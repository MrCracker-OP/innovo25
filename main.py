# app.py
from flask import Flask, request, jsonify
from flask_caching import Cache
from flask_cors import CORS
import os
import pandas as pd
# import numpy as np
from datetime import datetime
import google.generativeai as genai
import uuid
import json
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder='static')
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")
genai.configure(api_key=GEMINI_API_KEY)


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


@app.route('/')
def index():
    return "HELLO WORLD"


@app.route('/api/emissions', methods=['GET'])
def get_emissions():
    emissions = load_emissions_data()

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

    data['id'] = str(uuid.uuid4())
    data['created_at'] = datetime.now().isoformat()

    activity_type = data.get('activity_type')
    quantity = float(data.get('quantity', 0))

    emission_factors = {
        'electricity': 0.5,  # kg CO2e per kWh
        'natural_gas': 2.1,  # kg CO2e per m3
        'vehicle_fleet': 2.3,  # kg CO2e per liter of fuel
        'purchased_goods': 1.0,  # generic factor per kg
    }

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


@app.route('/api/advice', methods=['POST'])
@cache.cached(timeout=300, query_string=True)  # Cache for 5 mins
def get_carbon_reduction_advice():
    """
    Use the Gemini API to generate carbon emission reduction advice based on summary data.
    """
    try:
        summary_data = request.json
        # Create a prompt for the Gemini API
        prompt = f"""
        You are an expert in sustainability and carbon emission reduction. Based on the following emissions data, provide specific, actionable advice on how to reduce carbon emissions. Focus on practical strategies tailored to the data provided, including which areas (scopes or business units) to prioritize.

        Emissions Data:
        - Total Emissions: {summary_data['total_emissions']} tCO2e
        - Emissions by Scope: {summary_data['emissions_by_scope']}
        - Emissions by Business Unit: {summary_data['emissions_by_business_unit']}
        - Emissions Trend (by month): {summary_data['emissions_trend']}
        - Business Unit Trends: {summary_data['business_unit_trend']}

        Provide your advice in a concise, bullet-point format, prioritizing the areas with the highest emissions.
        """

        # Initialize the Gemini model (use 'gemini-1.5-pro' or 'gemini-1.5-flash' based on your needs)
        model = genai.GenerativeModel('gemini-1.5-pro')

        # Generate the advice
        response = model.generate_content(prompt)

        # Extract the text from the response
        advice = response.text if response.text else "Unable to generate advice at this time."

        return jsonify({"advice": advice})  # Ensure JSON response

    except Exception as e:
        # Handle errors gracefully (e.g., API key issues, network errors)
        return jsonify({"error": str(e)}), 500  # Return error as JSON


def determine_scope(activity_type):
    """
    Determine the emissions scope based on activity type.
    """
    scope1_activities = {
        'stationary_combustion', 'mobile_combustion', 'vehicle_fleet',
        'refrigerants', 'natural_gas', 'diesel', 'propane', 'process_emissions'
    }

    scope2_activities = {
        'electricity', 'steam', 'heating', 'cooling', 'purchased_energy'
    }

    activity = activity_type.lower().replace(' ', '_')

    if activity in scope1_activities:
        return 'Scope 1'
    elif activity in scope2_activities:
        return 'Scope 2'
    return 'Scope 3'  # Default to Scope 3 for all other activities


def get_emission_factor(activity_type, unit):
    """
    Get the emission factor for converting to CO2e based on activity type and unit.
    Values are simplified examples - you should replace with actual emission factors.
    """
    # Normalize inputs
    activity = activity_type.lower().replace(' ', '_')
    unit = unit.lower().replace(' ', '_')

    # Emission factors mapping (activity_type, unit) to kgCO2e per unit
    # emission_factors = {
    #     # Scope 1
    #     ('natural_gas', 'therms'): 5.3,  # kg CO2e per therm
    #     ('natural_gas', 'mmbtu'): 53.0,  # kg CO2e per MMBtu
    #     ('vehicle_fleet', 'gallons'): 8.78,  # kg CO2e per gallon of gasoline
    #     ('diesel', 'gallons'): 10.21,  # kg CO2e per gallon of diesel
    #     ('propane', 'gallons'): 5.72,  # kg CO2e per gallon
    #     ('refrigerants', 'kg'): 1430,  # Example for R134a (varies by type)
    #
    #     # Scope 2
    #     ('electricity', 'kwh'): 0.385,  # kg CO2e per kWh (varies by region)
    #     ('electricity', 'mwh'): 385,  # kg CO2e per MWh
    #     ('steam', 'mmbtu'): 66.33,  # kg CO2e per MMBtu
    #     ('cooling', 'ton_hour'): 0.735,  # kg CO2e per ton-hour
    #
    #     # Scope 3
    #     ('business_travel', 'miles'): 0.404,  # kg CO2e per mile
    #     ('employee_commuting', 'miles'): 0.404,  # kg CO2e per mile
    #     ('waste_disposal', 'kg'): 2.53,  # kg CO2e per kg waste
    #     # Example factor - varies widely by good
    #     ('purchased_goods', 'kg'): 3.0,
    # }

    emission_factors = {
        # Scope 1
        ('natural_gas', 'scm'): 2.15,  # kg CO2e per Standard Cubic Meter
        ('natural_gas', 'mmbtu'): 56.1,  # kg CO2e per MMBtu
        ('vehicle_fleet', 'liters'): 2.31,  # kg CO2e per liter of petrol
        ('diesel', 'liters'): 2.68,  # kg CO2e per liter of diesel
        ('lpg', 'kg'): 2.99,  # kg CO2e per kg of LPG
        ('refrigerants', 'kg'): 1430,  # Example for R134a (varies by type)

        # Scope 2
        # kg CO2e per kWh (CEA 2023 grid emission factor)
        ('electricity', 'kwh'): 0.79,
        ('electricity', 'mwh'): 790,  # kg CO2e per MWh
        ('steam', 'mmbtu'): 75.2,  # kg CO2e per MMBtu
        # kg CO2e per TR-hour (based on Indian HVAC systems)
        ('cooling', 'tr_hour'): 0.948,

        # Scope 3
        # kg CO2e per km (average for Indian vehicles)
        ('business_travel', 'km'): 0.14,
        ('employee_commuting', 'km'): 0.14,  # kg CO2e per km
        # kg CO2e per kg waste (Indian landfill conditions)
        ('waste_disposal', 'kg'): 2.94,
        ('rail_travel', 'km'): 0.029,  # kg CO2e per km (Indian Railways)
        ('domestic_flight', 'km'): 0.17,  # kg CO2e per km
        ('international_flight', 'km'): 0.115,  # kg CO2e per km

        # India-specific categories
        ('autorickshaw', 'km'): 0.08,  # kg CO2e per km (CNG auto)
        ('two_wheeler', 'km'): 0.045,  # kg CO2e per km
        ('bus_travel', 'km'): 0.015,  # kg CO2e per passenger-km
    }

    default_factor = 1.0

    return emission_factors.get((activity, unit), default_factor)


def calculate_co2e(row):
    """
    Calculate CO2e based on quantity, activity type, and unit
    """
    try:
        quantity = float(row.get('quantity', 0))
        activity_type = row.get('activity_type', '')
        unit = row.get('unit', 'metric_tons').lower()

        # Get appropriate emission factor
        factor = get_emission_factor(activity_type, unit)

        # Calculate CO2e in kg
        co2e = quantity * factor

        # Convert to metric tons if over 1000 kg
        if co2e > 1000:
            co2e = co2e / 1000
            co2e_unit = 'metric_tons'
        else:
            co2e_unit = 'kg'

        return round(co2e, 3), co2e_unit

    except Exception as e:
        print(f"Error calculating CO2e: {str(e)}")
        return 0, 'kg'


def clean_emissions_data(df):
    """
    Clean and process emissions data from uploaded CSV.
    """
    try:
        # Previous cleaning code remains the same...
        df = df.dropna(how='all')

        # Convert column names to lowercase and replace spaces with underscores
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]

        # Standardize column names
        column_mapping = {
            'activity_type': 'activity_type',
            'activity': 'activity_type',
            'business_unit': 'business_unit',
            'business': 'business_unit',
            'date': 'date',
            'quantity': 'quantity',
            'units': 'unit',
            'unit': 'unit',
            'source': 'source',
            'scope': 'scope'
        }

        # Rename columns based on mapping
        df = df.rename(columns={col: column_mapping.get(col, col)
                       for col in df.columns})

        # Ensure required columns exist
        required_columns = ['activity_type', 'quantity', 'date', 'unit']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        # Clean and standardize dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(
                df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

        # Fill missing values
        df['business_unit'] = df['business_unit'].fillna('Not Specified')
        df['source'] = df['source'].fillna('CSV Import')
        df['unit'] = df['unit'].fillna('metric_tons')

        # Determine scope based on activity type
        if 'scope' not in df.columns:
            df['scope'] = df['activity_type'].apply(determine_scope)

        # Convert quantity to float
        df['quantity'] = pd.to_numeric(
            df['quantity'], errors='coerce').fillna(0)

        # Calculate CO2e for each row
        co2e_data = df.apply(calculate_co2e, axis=1)
        df['co2e'] = [x[0] for x in co2e_data]
        df['co2e_unit'] = [x[1] for x in co2e_data]

        return df

    except Exception as e:
        print(f"Error in clean_emissions_data: {str(e)}")
        print(f"Current dataframe columns: {df.columns.tolist()}")
        raise


@app.route('/api/upload/csv', methods=['POST'])
def upload_csv():
    # Previous upload code remains the same...
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read file
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 500

        # Process data
        try:
            df = clean_emissions_data(df)
        except Exception as e:
            return jsonify({'error': f'Error cleaning data: {str(e)}'}), 500

        # Convert to records and add metadata
        try:
            new_emissions = df.to_dict('records')
            current_emissions = load_emissions_data()

            # Add IDs and timestamps
            for emission in new_emissions:
                emission['id'] = str(uuid.uuid4())
                emission['created_at'] = datetime.now().isoformat()
                emission['quantity'] = float(emission['quantity'])
                emission['co2e'] = float(emission['co2e'])
                current_emissions.append(emission)

            # Save updated data
            save_emissions_data(current_emissions)
        except Exception as e:
            return jsonify({'error': f'Error processing records: {str(e)}'}), 500

        # Clean up
        os.remove(filepath)

        return jsonify({
            'success': True,
            'message': f'Successfully processed {len(new_emissions)} records',
            'records_added': len(new_emissions),
            # Return first record for verification
            'sample_record': new_emissions[0] if new_emissions else None
        })

    except Exception as e:
        # Clean up on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


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
