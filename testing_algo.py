def determine_emission_scope(activity_type, business_unit=None, source=None):
    """
    Automatically determines the emission scope based on activity type and other parameters.

    Args:
        activity_type (str): The type of activity (e.g., 'natural_gas', 'electricity', etc.)
        business_unit (str, optional): The business unit involved
        source (str, optional): The source of the emission data

    Returns:
        str: The determined scope ('Scope 1', 'Scope 2', or 'Scope 3')
    """

    activity_type = activity_type.lower().strip()

    scope_1_activities = {
        'natural_gas',
        'diesel',
        'petrol',
        'gasoline',
        'propane',
        'coal',
        'fuel_oil',
        'refrigerants',
        'process_emissions',
        'fugitive_emissions',
        'company_vehicles',
        'on-site_combustion'
    }

    scope_2_activities = {
        'electricity',
        'purchased_heat',
        'purchased_steam',
        'purchased_cooling'
    }

    scope_3_activities = {
        'business_travel',
        'employee_commuting',
        'waste_disposal',
        'purchased_goods',
        'capital_goods',
        'upstream_transport',
        'downstream_transport',
        'third_party_logistics',
        'franchises',
        'investments'
    }

    if activity_type in scope_1_activities:
        return "Scope 1"
    elif activity_type in scope_2_activities:
        return "Scope 2"
    elif activity_type in scope_3_activities:
        return "Scope 3"

    if source and business_unit:
        if 'purchased' in source.lower() and 'electricity' in activity_type:
            return "Scope 2"
        if 'third_party' in source.lower() or 'vendor' in source.lower():
            return "Scope 3"
        if 'manufacturing' in business_unit.lower() and 'gas' in activity_type:
            return "Scope 1"

    return "Scope 3"


def process_emission_data(emission_entry):
    """
    Processes an emission entry and auto-determines the scope.

    Args:
        emission_entry (dict): The emission data dictionary

    Returns:
        dict: Updated emission data with auto-determined scope
    """
    updated_entry = emission_entry.copy()

    scope = determine_emission_scope(
        activity_type=emission_entry['activity_type'],
        business_unit=emission_entry.get('business_unit'),
        source=emission_entry.get('source')
    )

    updated_entry['scope'] = scope

    return updated_entry


emission_data = {
    "id": "550e8400-e29b-41d4-a716-446655440020",
    "date": "2024-02-23",
    "business_unit": "Manufacturing",
    "activity_type": "natural_gas",
    "quantity": 2100,
    "unit": "m3",
    "source": "Gas Bills",
    "co2e": 4410,
    "co2e_unit": "kgCO₂e",
    "created_at": "2024-02-23T13:15:00.000Z"
}

updated_data = process_emission_data(emission_data)
print(updated_data)
