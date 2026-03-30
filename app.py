import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk

# --- CONFIG & MODELS ---
st.set_page_config(page_title="Carbon & Cost Predictor", layout="wide")

@st.cache_resource
def load_resources():
    return (
        joblib.load('carbon_predictor_model.pkl'),
        joblib.load('cost_predictor_model.pkl'),
        joblib.load('elec_predictor_model.pkl'),
        joblib.load('gas_predictor_model.pkl'), 
        pd.read_csv('Master_City_Database_Geocoded.csv'),
        pd.read_csv('ML_Master_Data_Final.csv')
    )

carbon_model, cost_model, elec_model, gas_model, city_db, ground_truth_df = load_resources()

FIXED_AREA = 20000 

# --- SIDEBAR & GLOBAL LOGIC ---
st.sidebar.title("Thesis Navigation")
page = st.sidebar.radio("Go to:", [
    "🏠 Introduction", 
    "⚙️ System Descriptions", 
    "🧱 Building Templates", 
    "📊 Results Dashboard", 
    "📋 Recommendations & Export"
])

st.sidebar.markdown("---")
st.sidebar.header("Building Settings")
city_db['City_State'] = city_db['City'] + ", " + city_db['State']
selected_city = st.sidebar.selectbox("Select Location", sorted(city_db['City_State'].tolist()))
building_type = st.sidebar.selectbox("Building Type", ["Office", "Retail", "Hospital", "School"])

# Get location data
city_data = city_db[city_db['City_State'] == selected_city].iloc[0]
raw_city_name = city_data['City']

# --- SCENARIO OVERRIDE LOGIC ---
st.sidebar.markdown("---")
st.sidebar.header("🔮 Scenario Modeling")
override_grid = st.sidebar.checkbox("Override Utility Grid Data")

if override_grid:
    grid_fossil_pct = st.sidebar.slider(
        "Fossil Fuel Percentage (%)", 
        min_value=0, max_value=100, 
        value=int(city_data['Grid_Fossil_Pct'] * 100)
    ) / 100.0
    grid_renew_pct = 1.0 - grid_fossil_pct
else:
    grid_fossil_pct = city_data['Grid_Fossil_Pct']
    grid_renew_pct = city_data['Grid_Renew_Pct']

bldg_map = {"Office": 0, "Retail": 1, "Hospital": 2, "School": 3}
sys_map = {"Elec": 0, "Gas": 1}

# Check if city is Ground Truth
is_ground_truth = (raw_city_name in ground_truth_df['City'].values) and (building_type == "Office")

# Shared Prediction Function
def get_hybrid_preds(sys_type):
    # ONLY use ground truth if they haven't overridden the grid
    if is_ground_truth and not override_grid:
        match = ground_truth_df[(ground_truth_df['City'] == raw_city_name) & 
                                (ground_truth_df['Simulation Name'].str.endswith(sys_type))]
        if not match.empty:
            carb = match['Total_Carbon_Lbs'].values[0]
            cost = match['Total_Cost_$'].values[0]
            elec_mwh = match['Site_Elec_MWh'].values[0]
            gas_therms = match['Site_Gas_Therms'].values[0]
            site_kbtu = (elec_mwh * 3412.14) + (gas_therms * 100)
            source_kbtu = (elec_mwh * 3412.14 * 2.8) + (gas_therms * 100 * 1.05)
            return carb, cost, site_kbtu / FIXED_AREA, source_kbtu / FIXED_AREA

    # AI Fallback
    input_df_str = pd.DataFrame({'HDD': [city_data['HDD']], 'CDD': [city_data['CDD']], 
                                 'Grid_Fossil_Pct': [grid_fossil_pct], 'Grid_Renew_Pct': [grid_renew_pct],
                                 'Building_Type': [building_type], 'System_Type': [sys_type]})
    
    input_df_int = pd.DataFrame({'HDD': [city_data['HDD']], 'CDD': [city_data['CDD']], 
                                 'Grid_Fossil_Pct': [grid_fossil_pct], 'Grid_Renew_Pct': [grid_renew_pct],
                                 'Building_Type': [bldg_map[building_type]], 'System_Type': [sys_map[sys_type]]})
    
    carb = carbon_model.predict(input_df_str)[0]
    cost = cost_model.predict(input_df_str)[0]
    elec_mwh = elec_model.predict(input_df_int)[0]
    gas_therms = gas_model.predict(input_df_int)[0]
    
    site_kbtu = (elec_mwh * 3412.14) + (gas_therms * 100)
    source_kbtu = (elec_mwh * 3412.14 * 2.8) + (gas_therms * 100 * 1.05)
    return carb, cost, site_kbtu / FIXED_AREA, source_kbtu / FIXED_AREA

# Run predictions for ALL pages
gas_c, gas_p, gas_site, gas_source = get_hybrid_preds("Gas")
elec_c, elec_p, elec_site, elec_source = get_hybrid_preds("Elec")

# Calculate Dynamic Percentages for the Final Verdict
if elec_c < gas_c:
    carb_diff_pct = ((gas_c - elec_c) / gas_c) * 100
    carb_diff_text = f"a **{carb_diff_pct:.1f}% reduction** in carbon footprint"
else:
    carb_diff_pct = ((elec_c - gas_c) / gas_c) * 100
    carb_diff_text = f"a **{carb_diff_pct:.1f}% larger** carbon footprint"


# --- 1. INTRODUCTION PAGE ---
if page == "🏠 Introduction":
    st.header("Predictive Building Analytics Tool")
    st.write("""
    This tool was developed as part of a Master's Thesis to bridge the gap between 
    **High-Fidelity Physics Simulations (IES VE)** and **Rapid Concept Design**.
    
    By utilizing a Random Forest Machine Learning algorithm trained on ASHRAE-compliant 
    simulations across 20 climate zones, this tool allows architects to instantly 
    compare Carbon and Cost implications for any city in the US.
    """)
    st.info(f"**Note:** All calculations are currently based on a standardized Office Building prototype modeled at **{FIXED_AREA:,} SF**.")

# --- 2. SYSTEM DESCRIPTIONS ---
elif page == "⚙️ System Descriptions":
    st.header("Mechanical System Configurations")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mixed-Fuel (Gas)")
        st.write("- **Heating:** Natural Gas Boiler / RTU")
        st.write("- **Cooling:** DX Cooling")
        st.write("- **Logic:** Traditional baseline for most US jurisdictions.")
    with col2:
        st.subheader("All-Electric")
        st.write("- **Heating:** Air-Source Heat Pump / Electric Resistance")
        st.write("- **Cooling:** High-Efficiency DX")
        st.write("- **Logic:** Decarbonization pathway leveraging grid cleaning.")

# --- 3. BUILDING TEMPLATES ---
elif page == "🧱 Building Templates":
    st.header("ASHRAE Prototype & IES VE Model")
    st.write("The model follows ASHRAE 90.1-2019 standards for envelope and internal loads.")
    st.table({
        "Component": ["Roof", "Walls", "Windows (WWR)"],
        "Value": ["R-30 c.i.", "R-13 + R-7.5 c.i.", "40% Window-to-Wall Ratio"]
    })

# --- 4. RESULTS DASHBOARD ---
elif page == "📊 Results Dashboard":
    st.subheader(f"Location Analysis: {selected_city}")
    
    # Dynamic Banner
    if is_ground_truth and not override_grid:
        st.success(f"✅ **Ground Truth Data:** Showing precise IES VE physics simulation results for {raw_city_name}.")
    elif override_grid:
        st.warning(f"🔮 **Future Scenario Mode:** AI predicting performance for {raw_city_name} with a hypothetical {grid_fossil_pct*100:.1f}% fossil grid.")
    else:
        st.info(f"🤖 **AI Prediction Mode:** Estimated via Machine Learning based on {city_data['HDD']} HDD and {grid_fossil_pct*100:.1f}% Fossil Grid.")

    map_view = st.radio("Select Map Layer:", ["🌍 Grid Carbon Intensity", "🌡️ ASHRAE Climate Zones"], horizontal=True)

    if "Grid" in map_view:
        st.markdown("**Legend:** &nbsp;&nbsp; <span style='color:#FF0000; font-size:20px;'>■</span> High Fossil Fuels &nbsp;&nbsp;&nbsp;&nbsp; <span style='color:#00FF00; font-size:20px;'>■</span> High Renewables", unsafe_allow_html=True)
    else:
        st.markdown("**Legend:** &nbsp;&nbsp; <span style='color:#FF0000; font-size:20px;'>■</span> 1 (Very Hot) &nbsp;&nbsp; <span style='color:#FF8C00; font-size:20px;'>■</span> 2 (Hot) &nbsp;&nbsp; <span style='color:#FFFF00; font-size:20px;'>■</span> 3 (Warm) &nbsp;&nbsp; <span style='color:#00FF00; font-size:20px;'>■</span> 4 (Mixed) &nbsp;&nbsp; <span style='color:#0000FF; font-size:20px;'>■</span> 5 (Cool) &nbsp;&nbsp; <span style='color:#4B0082; font-size:20px;'>■</span> 6 (Cold) &nbsp;&nbsp; <span style='color:#8B00FF; font-size:20px;'>■</span> 7/8 (Very Cold)", unsafe_allow_html=True)

    def get_map_color(row):
        if "Grid" in map_view:
            fossil = row['Grid_Fossil_Pct']
            return [int(255 * fossil), int(255 * (1 - fossil)), 0, 160]
        else:
            zone = str(row['ASHRAE_Zone'])
            if "1" in zone: return [255, 0, 0, 160]
            if "2" in zone: return [255, 140, 0, 160]
            if "3" in zone: return [255, 255, 0, 160]
            if "4" in zone: return [0, 255, 0, 160]
            if "5" in zone: return [0, 0, 255, 160]
            if "6" in zone: return [75, 0, 130, 160]
            return [139, 0, 255, 160]

    city_db['color'] = city_db.apply(get_map_color, axis=1)

    # --- UPDATED MAP LAYER ---
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=city_db,
        get_position="[lon, lat]",
        get_color="color",
        get_radius=100000, # Increased base physical radius so they are visible from high up
        radius_min_pixels=4, # DOTS WILL NEVER BE SMALLER THAN THIS
        radius_max_pixels=12, # DOTS WILL NEVER BE BIGGER THAN THIS (Fixes the zooming issue!)
        pickable=True
    )

    st.pydeck_chart(pdk.Deck(
        map_style='light',
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=39.8, longitude=-98.5, zoom=3.5, pitch=0),
        tooltip={"text": "{City_State}\nZone: {ASHRAE_Zone}\nFossil Fuels: {Grid_Fossil_Pct}"}
    ))
    # --- END MAP LAYER ---
    
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("### 🌍 Carbon (lbs)")
        st.bar_chart(pd.DataFrame({"System": ["Gas", "Elec"], "Carbon": [gas_c, elec_c]}).set_index("System"))
    with col2:
        st.write("### 💰 Cost ($)")
        st.bar_chart(pd.DataFrame({"System": ["Gas", "Elec"], "Cost": [gas_p, elec_p]}).set_index("System"))
    with col3:
        st.write("### 🔌 Site EUI")
        st.bar_chart(pd.DataFrame({"System": ["Gas", "Elec"], "Site EUI": [gas_site, elec_site]}).set_index("System"))
    with col4:
        st.write("### ⚡ Source EUI")
        st.bar_chart(pd.DataFrame({"System": ["Gas", "Elec"], "Source EUI": [gas_source, elec_source]}).set_index("System"))

# --- 5. RECOMMENDATIONS & EXPORT ---
elif page == "📋 Recommendations & Export":
    st.header(f"Automated Design Recommendations: {raw_city_name}")
    
    st.subheader("🌍 Environmental Impact")
    if elec_c < gas_c:
        st.success(f"**Go All-Electric:** The grid in this region is clean enough that an all-electric system will yield {carb_diff_text} annually compared to burning natural gas on-site.")
    else:
        st.warning(f"**Grid Warning:** The utility grid in this region relies heavily on fossil fuels ({grid_fossil_pct*100:.1f}%). Currently, an all-electric system yields {carb_diff_text} than a gas system. **Recommendation:** Consider a hybrid system, or design all-electric but anticipate higher emissions until the local grid decarbonizes.")

    st.subheader("💰 Economic Viability")
    if elec_p < gas_p:
        st.success(f"**Operational Savings:** Electricity utility rates and heat-pump efficiencies in this climate make the all-electric system cheaper to operate by **${(gas_p - elec_p):,.2f}** per year.")
    else:
        st.error(f"**Utility Cost Penalty:** Natural gas is currently cheaper per BTU in this market. An all-electric building will carry an operating cost premium of **${(elec_p - gas_p):,.2f}** annually.")

    st.subheader("🏛️ Final System Recommendation")
    if (elec_c < gas_c) and (elec_p < gas_p):
        st.info(f"⭐ **Strong Recommendation:** The All-Electric system is the clear winner for both the environment and the client's budget, offering {carb_diff_text}.")
    elif (elec_c < gas_c) and (elec_p > gas_p):
        st.info(f"⚖️ **Trade-off Required:** The All-Electric system offers {carb_diff_text}, but the client must be willing to pay a 'Green Premium' on their utility bills.")
    elif (elec_c > gas_c) and (elec_p < gas_p):
        st.info(f"⚖️ **Trade-off Required:** The All-Electric system saves money, but the local utility grid's heavy reliance on fossil fuels ({grid_fossil_pct*100:.1f}%) means the building will technically have {carb_diff_text} than a gas building.")
    else:
        st.info(f"🛑 **Tough Market:** The utility grid's reliance on fossil fuels ({grid_fossil_pct*100:.1f}%) results in {carb_diff_text} for All-Electric systems. Combined with higher utility costs, we recommend sticking with high-efficiency Natural Gas until grid infrastructure improves.")

    st.markdown("---")
    st.write("### 📥 Client Export")
    results_df = pd.DataFrame({
        "System": ["All-Electric", "Mixed-Fuel (Gas)"],
        "Total Carbon (lbs CO2)": [round(elec_c, 2), round(gas_c, 2)],
        "Total Annual Cost ($)": [round(elec_p, 2), round(gas_p, 2)],
        "Site EUI (kBtu/sf)": [round(elec_site, 2), round(gas_site, 2)],
        "Source EUI (kBtu/sf)": [round(elec_source, 2), round(gas_source, 2)]
    })
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Scenario Results as CSV", data=csv, file_name=f"{selected_city}_Energy_Results.csv", mime="text/csv")
