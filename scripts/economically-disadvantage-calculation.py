import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import logging
import numpy as np
import re
import math
from io import StringIO
from datetime import datetime
import os
import requests

warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress yfinance INFO/ERROR logs
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("pandas_datareader").setLevel(logging.CRITICAL)

# Load the csv files
file_path = "data/Economically Disadvantaged calculation C3 - World Bank Country Groups.csv"
poverty_rates_file = "data/Economically Disadvantaged calculation C3 - Poverty Rates.csv"
country_groups_df = pd.read_csv(file_path)
poverty_rates_df = pd.read_csv(poverty_rates_file)

# ---- Constant poverty lines in USD ----

POVERTY_LINE_USD = {
    "High income": 21.70,
    "Upper middle income": 6.85,
    "Lower middle income": 3.65,
    "Low income": 2.15
}


def fetch_exchange_rate(base: str = "USD", target_date: str | None = None) -> float | None:
    ticker = f"EUR{base}=X"

    try:
        df = yf.download(
            ticker,
            start="2025-09-17",
            end="2025-09-18",
            auto_adjust=False,
            progress=False
        )

        if not df.empty:
            return float(df["Close"].iloc[0])
        else:
            return 0.0

    except Exception as e:
        print(f"Error fetching exchange rate for {ticker}: {e}")
        return None


def get_poverty_line_eur(country: str | None, base_currency: str = "EUR") -> float:
    """
    Get Poverty line (P) in EUR for a given country.
    Returns np.nan if country is NaN or not found.
    """
    # Handle NaN/None input
    if country is None or (isinstance(country, float) and np.isnan(country)):
        print("Warning: country is NaN, returning np.nan")
        return np.nan

    # Clean columns
    country_groups_df.columns = country_groups_df.columns.str.replace(
        '\n', '').str.strip()

    # Step 1: Lookup country to get income group
    country_row = country_groups_df[country_groups_df['Country'] == country]
    if country_row.empty:
        print(
            f"Warning: Country '{country}' not found in World Bank Country Groups sheet, returning np.nan")
        return np.nan

    income_group = country_row['Income group'].values[0]
    country = country_row['Country'].values[0]

    # Step 2: Get poverty line in USD from constant
    P_usd = POVERTY_LINE_USD.get(income_group)
    if P_usd is None:
        print(
            f"Warning: Poverty line USD not found for income group '{income_group}', returning np.nan")
        return np.nan

    # Step 3: Fetch exchange rate USD→EUR for historical date
    eur_to_usd = fetch_exchange_rate("USD", "Sep 17 2025")
    P_eur = P_usd / eur_to_usd
    return P_eur


def get_country_with_country_code(country: str) -> float:
    # Clean columns
    country_groups_df.columns = country_groups_df.columns.str.replace(
        '\n', '').str.strip()

    # Step 1: Lookup country to get income group
    country_row = country_groups_df[country_groups_df['Country'].str.contains(
        country, case=False, na=False)]
    if country_row.empty:
        raise ValueError(
            f"Country '{country}' not found in World Bank Country Groups sheet")

    country = country_row['Country'].values[0]
    return country


def normalize_currency(currency):
    if currency is None:
        return None
    if isinstance(currency, float):   # catches NaN (float)
        return None
    return str(currency).strip()


def calculate_benchmark_pa_in_local_currency_model_2(
    country: str,
    currency: str,
    household_over_16_years_old: float,
    household_under_16_years_old: float,
    base_currency: str = "EUR"
) -> float:
    country = get_country_with_country_code(country)

    # Normalize currency
    currency = normalize_currency(currency)
    if currency is None:
        raise ValueError("Currency is missing or NaN")

    # Step 1: Get poverty line in EUR
    model_2_poverty_line_in_euro = get_poverty_line_eur(country, base_currency)

    # Step 2: Exchange rate

    if currency.upper() == base_currency.upper():
        exchange_rate_to_euro = 1.0
    else:
        exchange_rate_to_euro = fetch_exchange_rate(currency, "Sep 17 2025")

    if np.isnan(exchange_rate_to_euro):
        raise ValueError(
            f"Exchange rate for currency '{currency}' not available")

    # Step 3: Compute benchmark
    model_2_poverty_line_in_euro_yearly = model_2_poverty_line_in_euro * 365
    part = household_over_16_years_old + \
        household_under_16_years_old * exchange_rate_to_euro

    if model_2_poverty_line_in_euro_yearly * part == 0:
        benchmark = model_2_poverty_line_in_euro_yearly * exchange_rate_to_euro
    else:
        benchmark = model_2_poverty_line_in_euro_yearly * part * exchange_rate_to_euro

    return benchmark

# ---- Lookup functions ----


def get_ab2(country: str) -> int:
    """
    Equivalent to AB2: get the maximum reporting_year for a given country
    """
    rows = poverty_rates_df[poverty_rates_df["country"] == country]
    if rows.empty:
        return None
    return int(rows["reporting_year"].max())


def get_s2(country: str, ab2: str) -> float:
    """
    Equivalent to S2: Lookup poverty rate based on multiple keys
    """
    # Create a combined key
    key = str(country) + str(ab2)

    # Try exact match in Poverty Rates
    poverty_rates_df["combined"] = (
        poverty_rates_df["country"].astype(str) +
        poverty_rates_df["reporting_year"].astype(str)
    )
    match = poverty_rates_df.loc[poverty_rates_df["combined"] == key, "median"]
    if not match.empty and not pd.isna(match.iloc[0]):
        return float(match.iloc[0])

    # Try urban fallback
    poverty_rates_df["combined_urban"] = poverty_rates_df["combined"] + \
        poverty_rates_df["reporting_level"].astype(str)
    match_urban = poverty_rates_df.loc[poverty_rates_df["combined_urban"]
                                       == key + "urban", "median"]
    if not match_urban.empty:
        return float(match_urban.iloc[0])

    return None


def get_t2(s2: float, historical_date: str) -> float:
    """
    Equivalent to T2: Convert USD → EUR
    """
    usd_per_eur = fetch_exchange_rate("USD", historical_date)
    return s2 / usd_per_eur

# ---- Final benchmark calculation ----


def calculate_benchmark_pa_in_local_currency_model_4(
    country: str | None,
    currency: str,
    household_over_16: float,
    household_under_16: float,
    historical_date: str = "Sep 17 2025",
    base_currency: str = "EUR"
) -> float:
    # Handle NaN or None country input
    if country is None or (isinstance(country, float) and np.isnan(country)):
        print("Warning: country is NaN, returning np.nan")
        return np.nan

    # Clean and standardize country name (optional: partial match can be added)
    country = get_country_with_country_code(country)

    ab2 = get_ab2(country)
    if ab2 is None:
        print(
            f"Warning: Country '{country}' not found in Poverty Rates sheet, returning np.nan")
        return np.nan

    s2 = get_s2(country, ab2)
    if s2 is None:
        print(
            f"Warning: Poverty rate not found for '{country}' with AB2='{ab2}', returning np.nan")
        return np.nan

    t2 = get_t2(s2, historical_date)

    # Total household members
    household_total = household_over_16 + household_under_16

    # Calculate benchmark per person per year
    yearly_poverty_per_person = t2 * 365

    # Exchange rate
    exchange_rate_to_euro = 1.0 if currency.upper() == base_currency.upper(
    ) else fetch_exchange_rate(currency, historical_date)
    if np.isnan(exchange_rate_to_euro):
        print(
            f"Warning: Exchange rate for currency '{currency}' not available, returning np.nan")
        return np.nan

    # Total benchmark
    benchmark = yearly_poverty_per_person * exchange_rate_to_euro
    if benchmark * household_total == 0:
        return benchmark
    return benchmark * household_total

# ----------------------------
# CONFIG
# ----------------------------


DREAMAPPLY_API_KEY = os.getenv('DREAM_APPLY_API_KEY')
DREAMAPPLY_TABLE_ID = os.getenv('DREAM_APPLY_TABLE_ID')
HUBSPOT_API_KEY = os.getenv('HUBSPOT_API_KEY')
HUBSPOT_OBJECT_TYPE = os.getenv('HUBSPOT_OBJECT_TYPE')

# Map CSV countries to HubSpot allowed dropdown values
ALLOWED_COUNTRIES = {
    "AF Afghanistan": "Afghanistan",
    "AL Albania": "Albania",
    "DZ Algeria": "Algeria",
    "AS American Samoa": "American Samo",
    "AD Andorra": "Andorra",
    "AO Angola": "Angola",
    "AI Anguilla": "Anguilla",
    "AQ Antarctica": "Antartica",
    "AG Antigua & Barbuda": "Antigua & Barbuda",
    "AR Argentina": "Argentina",
    "AM Armenia": "Armenia",
    "AW Aruba": "Aruba",
    "AC Ascension Island": "Ascension Island",
    "AU Australia": "Australia",
    "AT Austria": "Austria",
    "AZ Azerbaijan": "Azerbaijan",
    "BS Bahamas": "Bahamas",
    "BH Bahrain": "Bahrain",
    "BD Bangladesh": "Bangladesh",
    "BB Barbados": "Barbados",
    "BY Belarus": "Belarus",
    "BE Belgium": "Belgium",
    "BZ Belize": "Belize",
    "BJ Benin": "Benin",
    "BM Bermuda": "Bermuda",
    "BT Bhutan": "Bhutan",
    "BO Bolivia": "Bolivia",
    "BA Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "BA Bosnia & Herzegovina": "Bosnia & Herzegovina",
    "BW Botswana": "Botswana",
    "BV Bouvet Island": "Bouvet Island",
    "BR Brazil": "Brazil",
    "IO British Indian Ocean Territory": "British Indian Ocean Territory",
    "VG British Virgin Islands": "British Virgin Islands",
    "BN Brunei": "Brunei",
    "BG Bulgaria": "Bulgaria",
    "BF Burkina Faso": "Burkina Faso",
    "BI Burundi": "Burundi",
    "KH Cambodia": "Cambodia",
    "CM Cameroon": "Cameroon",
    "CA Canada": "Canada",
    "IC Canary Islands": "Canary Islands",
    "CV Cape Verde": "Cape Verde",
    "BQ Caribbean Netherlands": "Caribbean Netherlands",
    "KY Cayman Islands": "Cayman Islands",
    "CF Central African Republic": "Central African Republic",
    "EA Ceuta & Melilla": "Ceuta & Melilla",
    "TD Chad": "Chad",
    "CL Chile": "Chile",
    "CN China": "China",
    "CX Christmas Island": "Christmas Island",
    "CP Clipperton Island": "Clipperton Island",
    "CC Cocos (Keeling) Islands": "Cocos (Keeling) Islands",
    "CO Colombia": "Colombia",
    "KM Comoros": "Comoros",
    "CG Congo - Brazzaville": "Congo - Brazzaville",
    "CD Congo - Kinshasa": "Congo - Kinshasa",
    "CK Cook Islands": "Cook Islands",
    "CR Costa Rica": "Costa Rica",
    "HR Croatia": "Croatia",
    "CU Cuba": "Cuba",
    "CW Curaçao": "Curaçao",
    "CY Cyprus": "Cyprus",
    "CY-N Cyprus (North)": "Cyprus (North)",
    "CZ Czechia": "Czechia",
    "CI Côte d'Ivoire": "Côte d'Ivoire",
    "DK Denmark": "Denmark",
    "DG Diego Garcia": "Diego Garcia",
    "DJ Djibouti": "Djibouti",
    "DM Dominica": "Dominica",
    "DO Dominican Republic": "Dominican Republic",
    "EC Ecuador": "Ecuador",
    "EG Egypt": "Egypt",
    "SV El Salvador": "El Salvador",
    "GQ Equatorial Guinea": "Equatorial Guinea",
    "ER Eritrea": "Eritrea",
    "EE Estonia": "Estonia",
    "SZ Eswatini": "Eswatini",
    "ET Ethiopia": "Ethiopia",
    "FK Falkland Islands": "Farkland Islands",
    "FO Faroe Islands": "Faroe Islands",
    "FJ Fiji": "Fiji",
    "FI Finland": "Finland",
    "FR France": "France",
    "GF French Guiana": "French Guiana",
    "PF French Polynesia": "French Poynesia",
    "TF French Southern Territories": "French Southern .terşr",
    "GA Gabon": "Gabon",
    "GM Gambia": "Gambia",
    "GE Georgia": "Georgia",
    "DE Germany": "Germany",
    "GH Ghana": "Ghana",
    "GI Gibraltar": "Gibraltar",
    "GR Greece": "Greece",
    "GL Greenland": "Greenland",
    "GD Grenada": "Grenada",
    "GP Guadeloupe": "Guadeloupe",
    "GU Guam": "Guam",
    "GT Guatemala": "Guatemala",
    "GG Guernsey": "Guernsey",
    "GN Guinea": "Guinea",
    "GW Guinea-Bissau": "Guinea-Bissau",
    "GY Guyana": "Guyana",
    "HT Haiti": "Haiti",
    "HM Heard & McDonald Islands": "Heard & McDonald Islands",
    "HN Honduras": "Honduras",
    "HK Hong Kong SAR China": "Hong Kong SAR China",
    "HU Hungary": "Hungary",
    "IS Iceland": "Iceland",
    "IN India": "India",
    "ID Indonesia": "Indonesia",
    "IR Iran": "Iran",
    "IQ Iraq": "Iraq",
    "IE Ireland": "Ireland",
    "IM Isle of Man": "Isle of Man",
    "IL Israel": "Israel",
    "IT Italy": "Italy",
    "JM Jamaica": "Jamaica",
    "JP Japan": "Japan",
    "JE Jersey": "Jersey",
    "JO Jordan": "Jordan",
    "KZ Kazakhstan": "Kazakhstan",
    "KE Kenya": "Kenya",
    "KI Kiribati": "Kiribati",
    "XK Kosovo": "Kosovo",
    "KW Kuwait": "Kuwait",
    "KG Kyrgyzstan": "Kyrgyzstan",
    "LA Laos": "Laos",
    "LV Latvia": "Latvia",
    "LB Lebanon": "Lebanon",
    "LS Lesotho": "Lesotho",
    "LR Liberia": "Liberia",
    "LY Libya": "Libya",
    "LI Liechtenstein": "Liechtenstein",
    "LT Lithuania": "Lithuania",
    "LU Luxembourg": "Luxembourg",
    "MO Macao SAR China": "Macao SAR China",
    "MG Madagascar": "Madagascar",
    "MW Malawi": "Malawi",
    "MY Malaysia": "Malaysia",
    "MV Maldives": "Maldives",
    "ML Mali": "Mali",
    "MT Malta": "Malta",
    "MH Marshall Islands": "Marshall Islands",
    "MQ Martinique": "Martinique",
    "MR Mauritania": "Mauritania",
    "MU Mauritius": "Mauritius",
    "YT Mayotte": "Mayotte",
    "MX Mexico": "Mexico",
    "FM Micronesia": "Micronesia",
    "MD Moldova": "Moldova",
    "MC Monaco": "Monaco",
    "MN Mongolia": "Mongolia",
    "ME Montenegro": "Montenegro",
    "MS Montserrat": "Montserrat",
    "MA Morocco": "Morocco",
    "MZ Mozambique": "Mozambique",
    "MM Myanmar (Burma)": "Myanmar (Burma)",
    "NA Namibia": "Namibia",
    "NR Nauru": "Nauru",
    "NP Nepal": "Nepal",
    "NL Netherlands": "Netherlands",
    "NC New Caledonia": "New Caledonia",
    "NZ New Zealand": "New Zealand",
    "NI Nicaragua": "Nicaragua",
    "NE Niger": "Niger",
    "NG Nigeria": "Nigeria",
    "NU Niue": "Niue",
    "NF Norfolk Island": "Norfolk Island",
    "KP North Korea": "North Korea",
    "MK North Macedonia": "North Macedonia",
    "MP Northern Mariana Islands": "Northern Mariana Islands",
    "NO Norway": "Norway",
    "OM Oman": "Oman",
    "PK Pakistan": "Pakistan",
    "PS Palestine": "State of Palestine",
    "PW Palau": "Palau",
    "PA Panama": "Panama",
    "PG Papua New Guinea": "Papua New Guinea",
    "PY Paraguay": "Paraguay",
    "PE Peru": "Peru",
    "PH Philippines": "Philippines",
    "PN Pitcairn Islands": "Pitcairn Islands",
    "PL Poland": "Poland",
    "PT Portugal": "Portugal",
    "PR Puerto Rico": "Puerto Rico",
    "QA Qatar": "Qatar",
    "RO Romania": "Romania",
    "RU Russia": "Russia",
    "RW Rwanda": "Rwanda",
    "RE Réunion": "Réunion",
    "WS Samoa": "Samoa",
    "SM San Marino": "San Marino",
    "SA Saudi Arabia": "Saudi Arabia",
    "SN Senegal": "Senegal",
    "RS Serbia": "Serbia",
    "SC Seychelles": "Seychelles",
    "SL Sierra Leone": "Sierra Leone",
    "SG Singapore": "Singapore",
    "SX Sint Maarten": "Sint Maarten",
    "SK Slovakia": "Slovakia",
    "SI Slovenia": "Slovenia",
    "SB Solomon Islands": "Solomon Islands",
    "SO Somalia": "Somalia",
    "ZA South Africa": "South Africa",
    "GS South Georgia & South Sandwich Islands": "South Georgia & South Sandwich Islands",
    "KR South Korea": "South Korea",
    "SS South Sudan": "South Sudan",
    "ES Spain": "Spain",
    "LK Sri Lanka": "Sri Lanka",
    "BL St. Barthélemy": "St. Barthélemy",
    "SH St. Helena": "St. Helena",
    "KN St. Kitts & Nevis": "St. Kitts & Nevis",
    "LC St. Lucia": "St. Lucia",
    "MF St. Martin": "St. Martin",
    "PM St. Pierre & Miquelon": "St. Pierre & Miquelon",
    "VC St. Vincent & Grenadines": "St. Vincent & Grenadines",
    "PS State of Palestine": "State of Palestine",
    "SD Sudan": "Sudan",
    "SR Suriname": "Suriname",
    "SJ Svalbard & Jan Mayen": "Svalbard & Jan Mayen",
    "SE Sweden": "Sweden",
    "CH Switzerland": "Switzerland",
    "SY Syria": "Syria",
    "ST São Tomé & Príncipe": "São Tomé & Príncipe",
    "TW Taiwan": "Taiwan",
    "TJ Tajikistan": "Tajikistan",
    "TZ Tanzania": "Tanzania",
    "TH Thailand": "Thailand",
    "TL Timor-Leste": "Timor-Leste",
    "TG Togo": "Togo",
    "TK Tokelau": "Tokelau",
    "TO Tonga": "Tonga",
    "TT Trinidad & Tobago": "Trinidad & Tobago",
    "TA Tristan da Cunha": "Tristan da Cunha",
    "TN Tunisia": "Tunisia",
    "TR Turkey": "Türkiye",
    "TR Türkiye": "Türkiye",
    "TM Turkmenistan": "Turkmenistan",
    "TC Turks & Caicos Islands": "Turks & Caicos Islands",
    "TV Tuvalu": "Tuvalu",
    "UM U.S. Outlying Islands": "U.S. Outlying Islands",
    "VI U.S. Virgin Islands": "U.S. Virgin Islands",
    "UG Uganda": "Uganda",
    "UA Ukraine": "Ukraine",
    "AE United Arab Emirates": "United Arab Emirates",
    "GB United Kingdom": "United Kingdom",
    "US United States": "United States",
    "UY Uruguay": "Uruguay",
    "UG Uganda": "Uganda",
    "UZ Uzbekistan": "Uzbekistan",
    "VU Vanuatu": "Vanuatu",
    "VA Vatican City": "Vatican City",
    "VE Venezuela": "Venezuela",
    "VN Vietnam": "Vietnam",
    "WF Wallis & Futuna": "Wallis & Futuna",
    "EH Western Sahara": "Western Sahara",
    "YE Yemen": "Yemen",
    "ZM Zambia": "Zambia",
    "ZW Zimbabwe": "Zimbabwe",
    "AX Åland Islands": "Åland Islands"
}

ALLOWED_STATUS = {
    "Unreplied": "unreplied",
    "Missing Documents": "missing_documents",
    "Not Eligible": "not_eligible",
    "Eligible": "eligible",
    "Ready to Evaluate": "ready_to_evaluate",
    "Evaluated": "evaluated",
    "Waitlisted": "waitlisted",
    "Disqualified": "disqualified",
    "Fellowship Candidate": "fellowship_candidate",
    "Disqualification": "disqualification",
    "Shortlisted": "shortlisted",
    "FS - Missing Document": "fs___missing_document",
    "FS - Evaluated": "fs___evaluated",
    "Enrolled": "enrolled",
    "Not eligible - country": "not_eligible___country",
    "Disqualification pre interviews": "disqualification_pre_interviews",
    "Not eligible - age group": "not_eligible___age_group",
    "Eligible (Mentor)": "Eligible (Mentor)",
    "Missing Document (Mentor)": "Missing Document (Mentor)",
    "Not Eligible (Mentor)": "Not Eligible (Mentor)",
    "Global Summit - Uninvited": "Global Summit - Uninvited",
    "Grant Application Invitation": "Grant Application Invitation",
    "Project Scholarship - Evaluated": "Project Scholarship - Evaluated",
    "Project Scholarship - Awarded": "Project Scholarship - Awarded"
}

MAPPING = {
    "email": "E-mail",
    "application__with_link_": "Application",
    "applicant__with_link_": "Applicant",
    "application_status": "Status",
    "first_name": "Given name(s)",
    "last_name": "Family name(s)",
    "citizenship": "Citizenship",
    "application_offer_type": "Offer type",
    "application_offer_type_confirmed": "Offer type (confirmed)",
    "address__country": "Address: Country",
    "application_created_date": "Created date",
    "application_submitted_date": "Submitted date",
    "gender": "Gender",
    "date_of_birth__dob_": "Date of birth",
    "yearly_household_income": "Extra question: What was your yearly household income in the year 2024, in your local currency?",
    "household_over_16_years_old": "Extra question: How many people in your household are over 16 years old?",
    "household_under_16_years_old": "Extra question: How many people in your household are under 16 years old?",
    "financial_support_type": "Extra question: For which type of financial support would you like to apply?",
    "dependency_on_your_income": "Extra question: How many people including yourself, depend on your income?",
    "income_reduction": "Extra question: Do you anticipate a reduction in your income due to your participation in the fellowship? Please specify why",
    "income_reduction__estimated__c4_": "Extra question: What is your estimated monthly income reduction as a result of your participation in the fellowship?",
}

EXCLUDED_COUNTRIES = {"egypt", "yemen"}

# ----------------------------
# HELPERS
# ----------------------------


def to_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_country_from_c3(addr: str):
    if pd.isna(addr):
        return np.nan
    parts = str(addr).strip().split()
    return " ".join(parts[1:]) if len(parts) > 1 else parts[0]


def parse_amount_with_currency(text):
    if pd.isna(text):
        return np.nan
    s = str(text).strip()
    if not s:
        return np.nan
    parts = s.split()
    num_str = re.sub(r"[^\d.\-]", "", parts[-1])
    try:
        return float(num_str)
    except:
        return np.nan


def fetch_dreamapply_data(table_id):
    url = f'https://applications.bevisioneers.world/api/tableviews/{table_id}/tabledata'
    headers = {'Authorization': f'DREAM apikey="{DREAMAPPLY_API_KEY}"'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    csv_text = response.text
    df = pd.read_csv(StringIO(csv_text))
    df.columns = df.columns.str.strip()
    return df


def sanitize_value(val):
    if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
        return None
    return val


def to_unix_ms(date_str):
    if pd.isna(date_str) or date_str == "":
        return None
    try:
        dt = pd.to_datetime(date_str).normalize()
        return int(dt.timestamp() * 1000)
    except:
        return None


def parse_numeric(val):
    if pd.isna(val) or val is None:
        return None
    try:
        return float(''.join(c for c in str(val) if c.isdigit() or c == '.'))
    except:
        return None


def normalize_enum(value, allowed_dict):
    if pd.isna(value) or value == "":
        return None
    key = value.strip().lower()
    return allowed_dict.get(key, None)


def map_country_to_region(country):
    # Define regions
    south_asia = {"India"}
    east_asia = {"Hong Kong SAR China", "Japan", "South Korea"}
    southeast_asia = {"Indonesia", "Malaysia",
                      "Philippines", "Singapore", "Thailand", "Vietnam"}
    north_america = {"Canada", "United States", "Mexico"}
    central_america = {"Costa Rica", "Panama"}
    south_america = {"Argentina", "Chile"}
    southern_africa = {"South Africa", "Zimbabwe"}
    east_africa = {"Kenya", "Rwanda", "Tanzania", "Uganda"}
    west_africa = {"Nigeria"}
    europe = {"Albania", "Andorra", "Åland Islands", "Austria", "Belgium", "Bosnia and Herzegovina", "Bulgaria",
              "Croatia", "Cyprus", "Czechia", "Denmark", "Estonia", "Faroe Islands", "Finland", "France", "Germany",
              "Gibraltar", "Greece", "Guernsey", "Hungary", "Iceland", "Ireland", "Italy", "Jersey", "Kosovo", "Latvia",
              "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Monaco", "Montenegro", "Netherlands", "North Macedonia",
              "Norway", "Poland", "Portugal", "Romania", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain",
              "Svalbard and Jan Mayen", "Sweden", "Switzerland", "Türkiye", "United Kingdom"}
    oceania = {"Australia"}

    if country in south_asia:
        return "South Asia"
    elif country in east_asia:
        return "East Asia"
    elif country in southeast_asia:
        return "South East Asia"
    elif country in north_america:
        return "North America"
    elif country in central_america:
        return "Central America"
    elif country in south_america:
        return "South America"
    elif country in southern_africa:
        return "Southern Africa"
    elif country in east_africa:
        return "East Africa"
    elif country in west_africa:
        return "West Africa"
    elif country in europe:
        return "Europe"
    elif country in oceania:
        return "Oceania"
    else:
        return "Out of Target Countries!"


def map_dob_to_age_group(dob_str):
    if not dob_str or pd.isna(dob_str):
        return None
    try:
        dob = datetime.strptime(dob_str, "%Y-%m-%d")
        today = datetime.today()
        age = today.year - dob.year - \
            ((today.month, today.day) < (dob.month, dob.day))
        if age < 16:
            return "under 16"
        elif age <= 18:
            return "16-18"
        elif age <= 23:
            return "19-23"
        elif age <= 28:
            return "23-28"
        else:
            return "over 28"
    except:
        return None


def econ_flag(income, benchmark):
    if pd.isna(income) or pd.isna(benchmark):
        return "no income stated!"
    return "yes" if income < benchmark else "no"


def econ_flag_after_reduction(income, reduction, benchmark):
    if pd.isna(income) or pd.isna(benchmark):
        return "no income stated!"
    return "yes" if (income - reduction) < benchmark else "no"

# Sanitize all objects first


def sanitize_dict(d):
    if isinstance(d, dict):
        return {k: sanitize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [sanitize_dict(v) for v in d]
    elif isinstance(d, float):
        if math.isnan(d) or math.isinf(d):
            return None
        return d
    else:
        return d


def safe_benchmark(func, *args):
    result = func(*args)
    if result is None:
        return 0  # or np.nan if you prefer
    return result


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    # --- Fetch C4 DreamApply data ---
    print("Fetching DreamApply data...")
    c4 = fetch_dreamapply_data(DREAMAPPLY_TABLE_ID)

    # Make a working copy of c4 as df
    df = c4.copy()

    # --- Clean / extract columns ---
    df["country"] = df["Address: Country"].astype(str).str.strip()
    df["household_over_16"] = df["Extra question: How many people in your household are over 16 years old?"].fillna(
        0)
    df["household_under_16"] = df["Extra question: How many people in your household are under 16 years old?"].fillna(
        0)
    df["currency"] = df["Extra question: What was your yearly household income in the year 2024, in your local currency?"].str.split().str[0]

    # --- Parse income and reductions ---
    df["Annual_Income_Local"] = df["Extra question: What was your yearly household income in the year 2024, in your local currency?"].apply(
        parse_amount_with_currency)
    df["Monthly_Reduction_Local"] = df["Extra question: What is your estimated monthly income reduction as a result of your participation in the fellowship?"].apply(
        parse_amount_with_currency).fillna(0)
    df["Monthly_Reduction_Local"] = np.where(
        df["Monthly_Reduction_Local"] < 0, 0, df["Monthly_Reduction_Local"])
    df["Annual_Reduction_Local"] = df["Monthly_Reduction_Local"] * 12

    # Drop excluded countries before any benchmark calculations
    df = df[~df["country"].str.lower().isin(EXCLUDED_COUNTRIES)]

    # --- Model5 benchmark per household ---
    # --- Filter eligible rows for calculations ---
    eligible_mask = df["Offer type"] == "Eligible"

    # --- Model5 benchmark per household (only for eligible rows) ---
    df.loc[eligible_mask, "economically_disadvantaged_benchmark"] = df.loc[eligible_mask].apply(
        lambda x: max(
            safe_benchmark(
                calculate_benchmark_pa_in_local_currency_model_2,
                x["country"], x["currency"], x["household_over_16"], x["household_under_16"]
            ),
            safe_benchmark(
                calculate_benchmark_pa_in_local_currency_model_4,
                x["country"], x["currency"], x["household_over_16"], x["household_under_16"]
            )
        ),
        axis=1
    )

    # --- Flags (only for eligible rows) ---
    df.loc[eligible_mask, "economically_disadvantaged"] = df.loc[eligible_mask].apply(
        lambda x: econ_flag(x["Annual_Income_Local"], x["economically_disadvantaged_benchmark"]), axis=1
    )

    df.loc[eligible_mask, "economically_disadvantaged___after_income_reduction"] = df.loc[eligible_mask].apply(
        lambda x: econ_flag_after_reduction(
            x["Annual_Income_Local"],
            x["Annual_Reduction_Local"],
            x["economically_disadvantaged_benchmark"]
        ),
        axis=1
    )

    def format_for_hubspot(df):
        """
        Convert DataFrame rows into HubSpot batch create/update objects.
        """
        hubspot_data = []

        for _, row in df.iterrows():
            obj = {"properties": {"application_term": "Cohort 4"}}

            # ---- Base mapped properties ----
            for hs_prop, csv_col in MAPPING.items():
                val = row.get(csv_col)

                if hs_prop in [
                    "application_created_date",
                    "application_submitted_date"
                ]:
                    obj["properties"][hs_prop] = to_unix_ms(val)

                elif hs_prop in [
                    "yearly_household_income",
                    "household_over_16_years_old",
                    "household_under_16_years_old",
                    "income_reduction__estimated__c4_",
                    "income_reduction"
                ]:
                    obj["properties"][hs_prop] = sanitize_value(
                        parse_numeric(val))

                elif hs_prop == "address__country":
                    rev_map = {v: v for v in ALLOWED_COUNTRIES.values()}
                    obj["properties"][hs_prop] = rev_map.get(val)

                elif hs_prop == "application_offer_type_confirmed":
                    obj["properties"][hs_prop] = normalize_enum(
                        val, ALLOWED_STATUS)

                else:
                    obj["properties"][hs_prop] = sanitize_value(val)

            # ---- Additional computed fields ----
            obj["properties"]["region"] = map_country_to_region(
                row.get("Address: Country")
            )
            obj["properties"]["age_bracket"] = map_dob_to_age_group(
                row.get("Date of birth")
            )

            # ---- New properties (clean and consistent) ----
            obj["properties"]["economically_disadvantaged"] = sanitize_value(
                row.get("economically_disadvantaged")
            )
            obj["properties"]["economically_disadvantaged_benchmark"] = sanitize_value(
                row.get("economically_disadvantaged_benchmark")
            )
            obj["properties"]["economically_disadvantaged___after_income_reduction"] = sanitize_value(
                row.get("economically_disadvantaged___after_income_reduction")
            )

            # ---- Email is required ----
            email = row.get("E-mail")
            if email and pd.notna(email):
                obj["properties"]["email"] = email
                hubspot_data.append(obj)
            else:
                print(f"⚠️ Skipped row without email: {row}")

        return hubspot_data

    def fetch_existing_hubspot_emails(emails):
        """Return HUBSPOT ID for each email that already exists."""

        url = f"https://api.hubapi.com/crm/objects/v3/{HUBSPOT_OBJECT_TYPE}/search"
        headers = {
            "Authorization": f"Bearer {HUBSPOT_API_KEY}",
            "Content-Type": "application/json"
        }

        existing = {}

        for email in emails:
            payload = {
                "filterGroups": [
                    {
                        "filters": [
                            {"propertyName": "email",
                                "operator": "EQ", "value": email},
                            {"propertyName": "application_term",
                                "operator": "EQ", "value": "Cohort 4"}
                        ]
                    }
                ],
                "properties": ["email"]
            }

            resp = requests.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            if data.get("results"):
                obj = data["results"][0]
                existing[email] = obj["id"]
        return existing

    def chunk_list(lst, chunk_size=100):
        """Yield successive chunks of size chunk_size from lst."""
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    def upload_to_hubspot(objects):
        """Create or update HubSpot objects safely."""

        headers = {
            'Authorization': f'Bearer {HUBSPOT_API_KEY}',
            'Content-Type': 'application/json'
        }

        # Step 1: Collect emails
        emails = [obj["properties"].get(
            "email") for obj in objects if obj["properties"].get("email")]

        # Step 2: Fetch existing matching records
        existing = fetch_existing_hubspot_emails(emails)

        to_create = []
        to_update = []

        for obj in objects:
            email = obj["properties"].get("email")

            if email in existing:
                # Existing record → update
                to_update.append({
                    "id": existing[email],
                    "properties": obj["properties"]
                })
            else:
                # New record → create
                to_create.append({
                    "properties": obj["properties"]
                })

        # Step 3: Batch CREATE (in chunks of 50)
        for chunk in chunk_list(to_create, 50):
            url = f"https://api.hubapi.com/crm/objects/v3/{HUBSPOT_OBJECT_TYPE}/batch/create"
            resp = requests.post(url, headers=headers, json={"inputs": chunk})
            if resp.status_code not in (200, 201):
                print("Error creating:", resp.status_code, resp.text)

        # Step 4: Batch UPDATE (in chunks of 50)
        for chunk in chunk_list(to_update, 50):
            url = f"https://api.hubapi.com/crm/objects/v3/{HUBSPOT_OBJECT_TYPE}/batch/update"
            resp = requests.post(url, headers=headers, json={"inputs": chunk})
            if resp.status_code not in (200, 201):
                print("Error updating:", resp.status_code, resp.text)

        print(
            f"Created {len(to_create)} and updated {len(to_update)} HubSpot objects.")

    print("Formatting for HubSpot...")
    hubspot_objects = format_for_hubspot(df)
    print(f"Prepared {len(hubspot_objects)} objects for HubSpot.")
    upload_to_hubspot(hubspot_objects)
    print("Upload completed successfully!")

# ---------------------------------------------
# NEW: Push Economically Disadvantaged flags to DreamApply scoresheets
# ---------------------------------------------

DREAMAPPLY_SCORESHEET_ECON_MAP = {
    "economically_disadvantaged": 164,
    "economically_disadvantaged___after_income_reduction": 165,
}

# column from the DreamApply tableview
DREAMAPPLY_APPLICATION_ID_COL = "Application"


def push_econ_scores_to_dreamapply(df, api_key):
    """
    Push 'economically_disadvantaged' and
    'economically_disadvantaged___after_income_reduction'
    to DreamApply scoresheets 164 and 165.

    Data in df:
      - 'yes', 'no', 'no income stated!'

    Mapping to scoresheet points:
      - yes  -> 2
      - no   -> 1
      - no income stated! -> 0
    """
    headers = {
        "Authorization": f'DREAM apikey="{api_key}"',
        "Accept": "application/json",
    }

    # Safety: make sure Application column exists
    if DREAMAPPLY_APPLICATION_ID_COL not in df.columns:
        print(
            f"❌ Application ID column '{DREAMAPPLY_APPLICATION_ID_COL}' not found in dataframe. Skipping DreamApply econ push.")
        return

    for col_name, scoresheet_id in DREAMAPPLY_SCORESHEET_ECON_MAP.items():
        if col_name not in df.columns:
            print(
                f"⚠️ Skipping scoresheet {scoresheet_id}: column '{col_name}' not found in dataframe.")
            continue

        url = f"https://applications.bevisioneers.world/api/scoresheets/{scoresheet_id}/scores"
        print(
            f"\nPushing column '{col_name}' to DreamApply scoresheet {scoresheet_id}...")

        consider = df[df[col_name].notna()].copy()
        print(
            f"Found {len(consider)} rows with non-empty '{col_name}' to consider.")

        sent = 0

        for idx, row in consider.iterrows():
            app_val = row[DREAMAPPLY_APPLICATION_ID_COL]
            if pd.isna(app_val):
                continue

            try:
                app_id = int(app_val)
            except Exception:
                print(
                    f"⚠️ Row {idx}: invalid Application ID '{app_val}', skipping.")
                continue

            raw_flag = str(row[col_name]).strip()
            if not raw_flag:
                # truly empty, nothing to push
                continue

            low = raw_flag.lower()

            # Map our three states to numeric points
            if low == "yes":
                points = "2"
            elif low == "no":
                points = "1"
            elif low == "no income stated!":
                points = "0"
            else:
                # Fallback: treat unknown as 0 but log it
                points = "0"
                print(
                    f"ℹ️ Row {idx}: unexpected flag value '{raw_flag}' for application {app_id} (sending 0).")

            payload = {
                "application": app_id,
                "points": points,
            }

            resp = requests.post(url, headers=headers, data=payload)
            if resp.status_code not in (200, 201, 204):
                print(
                    f"❌ Error for application {app_id} on scoresheet {scoresheet_id}: "
                    f"{resp.status_code} {resp.text[:400]}"
                )
            else:
                sent += 1

        print(
            f"✅ Done scoresheet {scoresheet_id} from column '{col_name}': sent={sent} updates to DreamApply.")


# ---- CALL IT (after HubSpot upload has finished) ----
print("\nNow pushing economically disadvantaged flags to DreamApply scoresheets 164 & 165...")
push_econ_scores_to_dreamapply(df, DREAMAPPLY_API_KEY)
print("DreamApply econ scores push completed.")
