import fire

import os
import glob
import json
import re

import pandas as pd
from geotext import GeoText

from configs import *

base_dir = BASE_DIR

profile_dir = PROFILE_DIR

feature_dir = os.path.join(base_dir, "user_features")

states = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
    "United States",
    "USA",
]

major_us_cities_100 = [
    "New York",
    "Los Angeles",
    "NY",
    "LA",
    "NYC",
    "Chicago",
    "CHI",
    "Houston",
    "Phoenix",
    "Philadelphia",
    "San Antonio",
    "San Diego",
    "Dallas",
    "San Jose",
    "Austin",
    "Jacksonville",
    "Fort Worth",
    "Columbus",
    "Charlotte",
    "Francisco",
    "Indianapolis",
    "Seattle",
    "Denver",
    "Washington",
    "Boston",
    "El Paso",
    "Nashville",
    "Detroit",
    "Oklahoma",
    "Portland",
    "Las Vegas",
    "Memphis",
    "Louisville",
    "Baltimore",
    "Milwaukee",
    "Minneapolis",
    "Albuquerque",
    "Tucson",
    "Fresno",
    "Mesa",
    "Sacramento",
    "Atlanta",
    "Kansas",
    "Colorado",
    "Omaha",
    "Raleigh",
    "Miami",
    "Long Beach",
    "Virginia",
    "Oakland",
    "Tulsa",
    "Arlington",
    "Tampa",
    "Orleans",
    "Wichita",
    "Cleveland",
    "Bakersfield",
    "Aurora",
    "Anaheim",
    "Honolulu",
    "Santa Ana",
    "Riverside",
    "Corpus Christi",
    "Lexington",
    "Stockton",
    "Henderson",
    "Saint Paul",
    "St. Louis",
    "Cincinnati",
    "Pittsburgh",
    "Greensboro",
    "Anchorage",
    "Plano",
    "Lincoln",
    "Orlando",
    "Irvine",
    "Newark",
    "Toledo",
    "Durham",
    "Chula Vista",
    "Fort Wayne",
    "Jersey City",
    "Petersburg",
    "Pittsburgh",
    "Princeton",
    "Laredo",
    "Madison",
    "Chandler",
    "Buffalo",
    "Lubbock",
    "Scottsdale",
    "Reno",
    "Glendale",
    "Gilbert",
    "Winston–Salem",
    "North Las Vegas",
    "Norfolk",
    "Chesapeake",
    "Garland",
    "Irving",
    "Hialeah",
    "Fremont",
    "Boise",
    "Richmond",
    "Baton Rouge",
]

country_names = [
    "afghanistan",
    "albania",
    "algeria",
    "andorra",
    "angola",
    "antigua and barbuda",
    "argentina",
    "armenia",
    "australia",
    "austria",
    "azerbaijan",
    "bahamas",
    "bahrain",
    "bangladesh",
    "barbados",
    "belarus",
    "belgium",
    "belize",
    "benin",
    "bhutan",
    "bolivia",
    "bosnia and herzegovina",
    "botswana",
    "brazil",
    "brunei",
    "bulgaria",
    "burkina faso",
    "burundi",
    "cabo verde",
    "cambodia",
    "cameroon",
    "canada",
    "central african republic",
    "chad",
    "chile",
    "china",
    "colombia",
    "comoros",
    "congo",
    "costa rica",
    "croatia",
    "cuba",
    "cyprus",
    "czechia",
    "democratic republic of the congo",
    "denmark",
    "djibouti",
    "dominica",
    "dominican republic",
    "ecuador",
    "egypt",
    "el salvador",
    "equatorial guinea",
    "eritrea",
    "estonia",
    "eswatini",
    "ethiopia",
    "fiji",
    "finland",
    "france",
    "gabon",
    "gambia",
    "georgia",
    "germany",
    "ghana",
    "greece",
    "grenada",
    "guatemala",
    "guinea",
    "guinea-bissau",
    "guyana",
    "haiti",
    "honduras",
    "hungary",
    "iceland",
    "india",
    "indonesia",
    "iran",
    "iraq",
    "ireland",
    "israel",
    "italy",
    "jamaica",
    "japan",
    "jordan",
    "kazakhstan",
    "kenya",
    "kiribati",
    "kuwait",
    "kyrgyzstan",
    "laos",
    "latvia",
    "lebanon",
    "lesotho",
    "liberia",
    "libya",
    "liechtenstein",
    "lithuania",
    "luxembourg",
    "madagascar",
    "malawi",
    "malaysia",
    "maldives",
    "mali",
    "malta",
    "marshall islands",
    "mauritania",
    "mauritius",
    "mexico",
    "micronesia",
    "moldova",
    "monaco",
    "mongolia",
    "montenegro",
    "morocco",
    "mozambique",
    "myanmar",
    "namibia",
    "nauru",
    "nepal",
    "netherlands",
    "new zealand",
    "nicaragua",
    "niger",
    "nigeria",
    "north korea",
    "north macedonia",
    "norway",
    "oman",
    "pakistan",
    "palau",
    "palestine",
    "panama",
    "papua new guinea",
    "paraguay",
    "peru",
    "philippines",
    "poland",
    "portugal",
    "qatar",
    "romania",
    "russia",
    "rwanda",
    "saint kitts and nevis",
    "saint lucia",
    "saint vincent and the grenadines",
    "samoa",
    "san marino",
    "sao tome and principe",
    "saudi arabia",
    "senegal",
    "serbia",
    "seychelles",
    "sierra leone",
    "singapore",
    "slovakia",
    "slovenia",
    "solomon islands",
    "somalia",
    "south africa",
    "south korea",
    "south sudan",
    "spain",
    "sri lanka",
    "sudan",
    "suriname",
    "sweden",
    "switzerland",
    "syria",
    "taiwan",
    "tajikistan",
    "tanzania",
    "thailand",
    "timor-leste",
    "togo",
    "tonga",
    "trinidad and tobago",
    "tunisia",
    "turkey",
    "turkmenistan",
    "tuvalu",
    "uganda",
    "ukraine",
    "united arab emirates",
    "united kingdom",
    "england",
    "uruguay",
    "uzbekistan",
    "vanuatu",
    "vatican city",
    "venezuela",
    "vietnam",
    "yemen",
    "zambia",
    "zimbabwe",
    "africa",
    "asia",
    "europe",
    "oceania",
    "antarctica",
    "world",
    "deutschland",
]

country_abbrs = [
    "AF",
    "AL",
    "DZ",
    "AD",
    "AO",
    "AG",
    "AR",
    "AM",
    "AU",
    "AT",
    "AZ",
    "BS",
    "BH",
    "BD",
    "BB",
    "BY",
    "BE",
    "BZ",
    "BJ",
    "BT",
    "BO",
    "BA",
    "BW",
    "BR",
    "BN",
    "BG",
    "BF",
    "BI",
    "CV",
    "KH",
    "CM",
    "CA",
    "CF",
    "TD",
    "CL",
    "CN",
    "CO",
    "KM",
    "CG",
    "CR",
    "HR",
    "CU",
    "CY",
    "CZ",
    "CD",
    "DK",
    "DJ",
    "DM",
    "DO",
    "EC",
    "EG",
    "SV",
    "GQ",
    "ER",
    "EE",
    "SZ",
    "ET",
    "FJ",
    "FI",
    "FR",
    "GA",
    "GM",
    "GE",
    "DE",
    "GH",
    "GR",
    "GD",
    "GT",
    "GN",
    "GW",
    "GY",
    "HT",
    "HN",
    "HU",
    "IS",
    "IN",
    "ID",
    "IR",
    "IQ",
    "IE",
    "IL",
    "IT",
    "JM",
    "JP",
    "JO",
    "KZ",
    "KE",
    "KI",
    "KW",
    "KG",
    "LA",
    "LV",
    "LB",
    "LS",
    "LR",
    "LY",
    "LI",
    "LT",
    "LU",
    "MG",
    "MW",
    "MY",
    "MV",
    "ML",
    "MT",
    "MH",
    "MR",
    "MU",
    "MX",
    "FM",
    "MD",
    "MC",
    "MN",
    "ME",
    "MA",
    "MZ",
    "MM",
    "NA",
    "NR",
    "NP",
    "NL",
    "NZ",
    "NI",
    "NE",
    "NG",
    "KP",
    "MK",
    "NO",
    "OM",
    "PK",
    "PW",
    "PS",
    "PA",
    "PG",
    "PY",
    "PE",
    "PH",
    "PL",
    "PT",
    "QA",
    "RO",
    "RU",
    "RW",
    "KN",
    "LC",
    "VC",
    "WS",
    "SM",
    "ST",
    "SA",
    "SN",
    "RS",
    "SC",
    "SL",
    "SG",
    "SK",
    "SI",
    "SB",
    "SO",
    "ZA",
    "KR",
    "SS",
    "ES",
    "LK",
    "SD",
    "SR",
    "SE",
    "CH",
    "SY",
    "TW",
    "TJ",
    "TZ",
    "TH",
    "TL",
    "TG",
    "TO",
    "TT",
    "TN",
    "TR",
    "TM",
    "TV",
    "UG",
    "UA",
    "UK",
    "AE",
    "GB",
    "UY",
    "UZ",
    "VU",
    "VA",
    "VE",
    "VN",
    "YE",
    "ZM",
    "ZW",
]

country_emojis = [
    "🇦🇫",  # Afghanistan
    "🇦🇱",  # Albania
    "🇩🇿",  # Algeria
    "🇦🇩",  # Andorra
    "🇦🇴",  # Angola
    "🇦🇬",  # Antigua and Barbuda
    "🇦🇷",  # Argentina
    "🇦🇲",  # Armenia
    "🇦🇺",  # Australia
    "🇦🇹",  # Austria
    "🇦🇿",  # Azerbaijan
    "🇧🇸",  # Bahamas
    "🇧🇭",  # Bahrain
    "🇧🇩",  # Bangladesh
    "🇧🇧",  # Barbados
    "🇧🇾",  # Belarus
    "🇧🇪",  # Belgium
    "🇧🇿",  # Belize
    "🇧🇯",  # Benin
    "🇧🇹",  # Bhutan
    "🇧🇴",  # Bolivia
    "🇧🇦",  # Bosnia and Herzegovina
    "🇧🇼",  # Botswana
    "🇧🇷",  # Brazil
    "🇧🇳",  # Brunei
    "🇧🇬",  # Bulgaria
    "🇧🇫",  # Burkina Faso
    "🇧🇮",  # Burundi
    "🇨🇻",  # Cabo Verde
    "🇰🇭",  # Cambodia
    "🇨🇲",  # Cameroon
    "🇨🇦",  # Canada
    "🇨🇫",  # Central African Republic
    "🇹🇩",  # Chad
    "🇨🇱",  # Chile
    "🇨🇳",  # China
    "🇨🇴",  # Colombia
    "🇰🇲",  # Comoros
    "🇨🇬",  # Congo
    "🇨🇷",  # Costa Rica
    "🇭🇷",  # Croatia
    "🇨🇺",  # Cuba
    "🇨🇾",  # Cyprus
    "🇨🇿",  # Czechia
    "🇨🇩",  # Democratic Republic of the Congo
    "🇩🇰",  # Denmark
    "🇩🇯",  # Djibouti
    "🇩🇲",  # Dominica
    "🇩🇴",  # Dominican Republic
    "🇪🇨",  # Ecuador
    "🇪🇬",  # Egypt
    "🇸🇻",  # El Salvador
    "🇬🇶",  # Equatorial Guinea
    "🇪🇷",  # Eritrea
    "🇪🇪",  # Estonia
    "🇸🇿",  # Eswatini
    "🇪🇹",  # Ethiopia
    "🇫🇯",  # Fiji
    "🇫🇮",  # Finland
    "🇫🇷",  # France
    "🇬🇦",  # Gabon
    "🇬🇲",  # Gambia
    "🇬🇪",  # Georgia
    "🇩🇪",  # Germany
    "🇬🇭",  # Ghana
    "🇬🇷",  # Greece
    "🇬🇩",  # Grenada
    "🇬🇹",  # Guatemala
    "🇬🇳",  # Guinea
    "🇬🇼",  # Guinea-Bissau
    "🇬🇾",  # Guyana
    "🇭🇹",  # Haiti
    "🇭🇳",  # Honduras
    "🇭🇺",  # Hungary
    "🇮🇸",  # Iceland
    "🇮🇳",  # India
    "🇮🇩",  # Indonesia
    "🇮🇷",  # Iran
    "🇮🇶",  # Iraq
    "🇮🇪",  # Ireland
    "🇮🇱",  # Israel
    "🇮🇹",  # Italy
    "🇯🇲",  # Jamaica
    "🇯🇵",  # Japan
    "🇯🇴",  # Jordan
    "🇰🇿",  # Kazakhstan
    "🇰🇪",  # Kenya
    "🇰🇮",  # Kiribati
    "🇰🇼",  # Kuwait
    "🇰🇬",  # Kyrgyzstan
    "🇱🇦",  # Laos
    "🇱🇻",  # Latvia
    "🇱🇧",  # Lebanon
    "🇱🇸",  # Lesotho
    "🇱🇷",  # Liberia
    "🇱🇾",  # Libya
    "🇱🇮",  # Liechtenstein
    "🇱🇹",  # Lithuania
    "🇱🇺",  # Luxembourg
    "🇲🇬",  # Madagascar
    "🇲🇼",  # Malawi
    "🇲🇾",  # Malaysia
    "🇲🇻",  # Maldives
    "🇲🇱",  # Mali
    "🇲🇹",  # Malta
    "🇲🇭",  # Marshall Islands
    "🇲🇷",  # Mauritania
    "🇲🇺",  # Mauritius
    "🇲🇽",  # Mexico
    "🇫🇲",  # Micronesia
    "🇲🇩",  # Moldova
    "🇲🇨",  # Monaco
    "🇲🇳",  # Mongolia
    "🇲🇪",  # Montenegro
    "🇲🇦",  # Morocco
    "🇲🇿",  # Mozambique
    "🇲🇲",  # Myanmar
    "🇳🇦",  # Namibia
    "🇳🇷",  # Nauru
    "🇳🇵",  # Nepal
    "🇳🇱",  # Netherlands
    "🇳🇿",  # New Zealand
    "🇳🇮",  # Nicaragua
    "🇳🇪",  # Niger
    "🇳🇬",  # Nigeria
    "🇰🇵",  # North Korea
    "🇲🇰",  # North Macedonia
    "🇳🇴",  # Norway
    "🇴🇲",  # Oman
    "🇵🇰",  # Pakistan
    "🇵🇼",  # Palau
    "🇵🇸",  # Palestine
    "🇵🇦",  # Panama
    "🇵🇬",  # Papua New Guinea
    "🇵🇾",  # Paraguay
    "🇵🇪",  # Peru
    "🇵🇭",  # Philippines
    "🇵🇱",  # Poland
    "🇵🇹",  # Portugal
    "🇶🇦",  # Qatar
    "🇷🇴",  # Romania
    "🇷🇺",  # Russia
    "🇷🇼",  # Rwanda
    "🇰🇳",  # Saint Kitts and Nevis
    "🇱🇨",  # Saint Lucia
    "🇻🇨",  # Saint Vincent and the Grenadines
    "🇼🇸",  # Samoa
    "🇸🇲",  # San Marino
    "🇸🇹",  # Sao Tome and Principe
    "🇸🇦",  # Saudi Arabia
    "🇸🇳",  # Senegal
    "🇷🇸",  # Serbia
    "🇸🇨",  # Seychelles
    "🇸🇱",  # Sierra Leone
    "🇸🇬",  # Singapore
    "🇸🇰",  # Slovakia
    "🇸🇮",  # Slovenia
    "🇸🇧",  # Solomon Islands
    "🇸🇴",  # Somalia
    "🇿🇦",  # South Africa
    "🇰🇷",  # South Korea
    "🇸🇸",  # South Sudan
    "🇪🇸",  # Spain
    "🇱🇰",  # Sri Lanka
    "🇸🇩",  # Sudan
    "🇸🇷",  # Suriname
    "🇸🇪",  # Sweden
    "🇨🇭",  # Switzerland
    "🇸🇾",  # Syria
    "🇹🇼",  # Taiwan
    "🇹🇯",  # Tajikistan
    "🇹🇿",  # Tanzania
    "🇹🇭",  # Thailand
    "🇹🇱",  # Timor-Leste
    "🇹🇬",  # Togo
    "🇹🇴",  # Tonga
    "🇹🇹",  # Trinidad and Tobago
    "🇹🇳",  # Tunisia
    "🇹🇷",  # Turkey
    "🇹🇲",  # Turkmenistan
    "🇹🇻",  # Tuvalu
    "🇺🇬",  # Uganda
    "🇺🇦",  # Ukraine
    "🇦🇪",  # United Arab Emirates
    "🇬🇧",  # United Kingdom
    "🇺🇾",  # Uruguay
    "🇺🇿",  # Uzbekistan
    "🇻🇺",  # Vanuatu
    "🇻🇦",  # Vatican City
    "🇻🇪",  # Venezuela
    "🇻🇳",  # Vietnam
    "🇾🇪",  # Yemen
    "🇿🇲",  # Zambia
    "🇿🇼",  # Zimbabwe
]

state_abbrs = [
    "US",
    "USA",
    "usa",
    "us",
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NYC",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]

prons = ["her", "she", "he", "him", "his", "they", "them"]

strange = [
    "hell",
    "heaven",
    "twitter",
    "tiktok",
    "instagram",
    "facebook",
    "fuck",
    "planet",
    "alien",
    "aliens",
    "earth",
    "emotion",
    "mastodon",
    "ig",
    "tweet",
    "idk",
    "stardew",
    ".com",
    "podcast",
    "mcdonalds",
    "kfc",
]


def gather_all_locations():
    if os.path.exists(os.path.join(base_dir, "all_locations.json")):
        with open(os.path.join(base_dir, "all_locations.json"), "r") as f:
            all_locations = json.load(f)
        return all_locations

    all_locations = set()
    all_feature_files = glob.glob(os.path.join(feature_dir, "user-*.parquet"))

    for feature_file in all_feature_files:
        df = pd.read_parquet(feature_file)
        df = df[df["location"].notna()]
        all_locations.update(df["location"].unique())

    all_locations = list(all_locations)
    print(len(all_locations))

    with open(os.path.join(base_dir, "all_locations.json"), "w") as f:
        json.dump(all_locations, f)

    return all_locations


def test_usaddress():
    all_locations = gather_all_locations()
    for i in range(100):
        location = all_locations[i + 2000]
        if location == "":
            continue
        print(location)

        print(GeoText(location).country_mentions)


def word_freq_analysis(texts):
    """
    找到高频词
    """
    word_freq = {}
    for text in texts:
        for word in text.split(" "):
            if len(word) < 2:
                continue
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    print(word_freq[:100])


def location_identification(location_file: str = None, output_file: str = None):
    if location_file is None:
        all_locations = gather_all_locations()
    else:
        if os.path.exists(location_file):
            with open(location_file, "r") as f:
                all_locations = json.load(f)
        else:
            raise FileNotFoundError(f"Location file {location_file} not found")

    non_us = set()
    us = set()
    not_sure = set()
    undecided = set()
    for location in all_locations:
        if location == "":
            not_sure.add(location)
            continue
        # 如果没有英文字母
        # if not re.search(r"[a-zA-Z]", location):
        #     not_sure.add(location)
        #     continue
        for word in strange: # + prons:
            continue_flag = False
            if word in location.lower():
                not_sure.add(location)
                continue_flag = True
                break
        if continue_flag:
            continue
        for state in states:
            if state.lower() in location.lower():
                us.add(location)
                continue_flag = True
                break
        if continue_flag:
            continue

        for city in major_us_cities_100:
            if city.lower() in location.lower():
                us.add(location)
                continue_flag = True
                break
        if continue_flag:
            continue

        continue_flag = False
        for country in country_names:
            if country.lower() in location.lower():
                non_us.add(location)
                continue_flag = True
                break
        if continue_flag:
            continue

        for emoji in country_emojis:
            if emoji in location:
                non_us.add(location)
                continue_flag = True
                break
        if continue_flag:
            continue

        splitted = (
            location.strip()
            .replace(",", " ")
            .replace(".", " ")
            .replace("/", " ")
            .split(" ")
        )
        for word in splitted:
            if word in state_abbrs:
                us.add(location)
                continue_flag = True
                break
            if word in country_abbrs:
                non_us.add(location)
                continue_flag = True
                break
        if continue_flag:
            continue

        country_result = GeoText(location).country_mentions
        if not country_result:
            undecided.add(location)
        else:
            if "US" in country_result:
                us.add(location)
            else:
                non_us.add(location)

    # 随机打印 undecided中的一百个
    # print(list(undecided)[:1000])
    # word_freq_analysis(list(undecided))

    print(
        f"non_us: {len(non_us)}, us: {len(us)}, not_sure: {len(not_sure)}, undecided: {len(undecided)}"
    )

    default_output_file = os.path.join(base_dir, "non_us_user_analysis.json")
    if output_file is None:
        output_file = default_output_file
    
    if os.path.exists(default_output_file):
        with open(default_output_file, "r") as f:
            existing_data = json.load(f)
        us_locations = set(existing_data["us"])
        
        # 使用集合交集操作找到需要移动的位置，避免在遍历时修改集合
        to_move_from_not_sure = not_sure & us_locations
        not_sure -= to_move_from_not_sure
        us |= to_move_from_not_sure
        
        to_move_from_undecided = undecided & us_locations
        undecided -= to_move_from_undecided
        us |= to_move_from_undecided

    with open(output_file, "w") as f:
        json.dump(
            {
                "non_us": list(non_us),
                "us": list(us),
                "not_sure": list(not_sure),
                "undecided": list(undecided),
            },
            f,
        )


def merge_and_report():
    base_dir = "ai_atti"
    gpt_dir = "ai_atti/llm_analysis"
    with open(os.path.join(base_dir, "non_us_user_analysis.json"), "r") as f:
        non_us_user_analysis = json.load(f)

    llm_result = pd.read_parquet(os.path.join(gpt_dir, "llm_result.parquet"))

    print(llm_result["result"].value_counts())
    # 0-not_sure, 1-us, 2-non_us, add the locations to the set

    llm_not_sure_locations = set(llm_result[llm_result["result"] == 0]["location"])
    llm_us_locations = set(llm_result[llm_result["result"] == 1]["location"])
    llm_non_us_locations = set(llm_result[llm_result["result"] == 2]["location"])

    print(
        f"llm_not_sure_locations: {len(llm_not_sure_locations)}, llm_us_locations: {len(llm_us_locations)}, llm_non_us_locations: {len(llm_non_us_locations)}"
    )

    non_us_user_analysis["not_sure"].extend(llm_not_sure_locations)
    non_us_user_analysis["us"].extend(llm_us_locations)
    non_us_user_analysis["non_us"].extend(llm_non_us_locations)

    print(
        f"non_us_user_analysis: {len(non_us_user_analysis['non_us'])}, us: {len(non_us_user_analysis['us'])}, not_sure: {len(non_us_user_analysis['not_sure'])}, undecided: {len(non_us_user_analysis['undecided'])}"
    )

    with open(os.path.join(base_dir, "non_us_user_analysis.json"), "w") as f:
        json.dump(non_us_user_analysis, f)


def get_non_us_user_ids():
    with open(os.path.join(base_dir, "non_us_user_analysis.json"), "r") as f:
        non_us_user_analysis = json.load(f)
        non_us_locations = non_us_user_analysis["non_us"]
        us_locations = non_us_user_analysis["us"]

    all_feature_files = glob.glob(os.path.join(feature_dir, "user-*.parquet"))

    us_user_ids = set()
    non_us_user_ids = set()
    for feature_file in all_feature_files:
        df = pd.read_parquet(feature_file)
        df = df[df["location"].notna()]
        non_us_df = df[df["location"].isin(non_us_locations)]
        non_us_user_ids.update(non_us_df["id"].unique())

        us_df = df[df["location"].isin(us_locations)]
        us_user_ids.update(us_df["id"].unique())

    print(f"non_us_user_ids: {len(non_us_user_ids)}")
    print(f"us_user_ids: {len(us_user_ids)}")

    with open(os.path.join(base_dir, "non_us_user_ids.json"), "w") as f:
        json.dump(list(non_us_user_ids), f)

    with open(os.path.join(base_dir, "us_user_ids.json"), "w") as f:
        json.dump(list(us_user_ids), f)


if __name__ == "__main__":
    fire.Fire(
        {
            "gather": gather_all_locations,
            "test": test_usaddress,
            "identify": location_identification,
            "merge": merge_and_report,
            "user": get_non_us_user_ids,
        }
    )
