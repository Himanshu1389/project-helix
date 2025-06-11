import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.compensation_metrics import (
    calculate_comp_ratio, 
    calculate_salary_range,
    calculate_equity_metrics,
    generate_recommendations
)
import shap
import joblib
import os
import streamlit.components.v1 as components
import numpy as np

# --- Login Screen Function ---
def login_screen():
    st.sidebar.empty() # Clear sidebar for login screen
    st.title("Login to Project Helix")
    st.write("Please enter your credentials to access the application.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

        if login_button:
            # Hardcoded credentials for demonstration purposes
            if username == "admin" and password == "admin@123":
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
                st.rerun() # Rerun to switch to main app
            else:
                st.error("Invalid username or password. Please try again.")

# Custom SHAP component for Streamlit
def st_shap(plot, height=None):
    # This should return the full HTML snippet including <script> tags
    shap_html = plot.html()
    # Provide a default height if not specified, and ensure width is 100%
    if height is None:
        height = 500 # A reasonable default height for SHAP plots
    components.html(shap_html, height=height, width="100%", scrolling=True)

# Page config
st.set_page_config(
    page_title="Project Helix - Compensation Planning",
    page_icon="💰",
    layout="wide"
)

# Initialize logged_in state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Main application logic controlled by login state
if st.session_state.logged_in:
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .employee-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .employee-card h3 {
            color: #1E88E5;
            margin-bottom: 10px;
        }
        .employee-card p {
            margin: 5px 0;
            color: #424242;
        }
        .metric-value {
            font-weight: bold;
            color: #1E88E5;
        }
        .comp-ratio-meter-container {
            width: 100%;
            background-color: #f0f2f6;
            border-radius: 5px;
            height: 10px;
            margin-top: 5px;
            overflow: hidden; /* Ensures the gradient bar stays within bounds */
        }
        .comp-ratio-meter-bar {
            height: 100%;
            border-radius: 5px;
            transition: width 0.3s ease-in-out, background-color 0.3s ease-in-out;
        }
        .comp-ratio {
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 15px;
            display: inline-block;
            margin-top: 5px;
        }
        .comp-ratio-high {
            background-color: #E8F5E9;
            color: #2E7D32;
        }
        .comp-ratio-medium {
            background-color: #FFF3E0;
            color: #E65100;
        }
        .comp-ratio-low {
            background-color: #FFEBEE;
            color: #C62828;
        }
        .stButton>button {
            width: 100%;
            margin-top: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title + Logo
    image_path = os.path.join(os.path.dirname(__file__), "assets", "helix_logo.png")
    st.sidebar.image(image_path, width=100)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Employee Data", "Compensation Analysis", "Market Insights", "AI Insights", "Data Management", "Settings"])

    # Logout Button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.experimental_rerun()

    # Load data
    # @st.cache_data # Commented out to ensure data is always reloaded
    def load_data():
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        employee_df = pd.read_csv(os.path.join(data_dir, "employee_data.csv"))
        benchmark_df = pd.read_csv(os.path.join(data_dir, "benchmark_data.csv"))

        # Load settings from settings.csv or initialize defaults
        settings_path = os.path.join(data_dir, "settings.csv")
        settings = {
            'company_name': "Your Organization Name",
            'currency': "₹",
            'target_comp_budget': 0.0
        }
        if os.path.exists(settings_path):
            try:
                settings_df = pd.read_csv(settings_path)
                if not settings_df.empty:
                    # Assuming settings.csv has one row with these columns
                    settings['company_name'] = settings_df['company_name'].iloc[0]
                    settings['currency'] = settings_df['currency'].iloc[0]
                    settings['target_comp_budget'] = float(settings_df['target_comp_budget'].iloc[0])
            except Exception as e:
                st.warning(f"Could not load settings from settings.csv. Using defaults. Error: {e}")
        else:
            # Create settings.csv with default values if it doesn't exist
            pd.DataFrame([settings]).to_csv(settings_path, index=False)

        # Initialize/update session state with loaded or default settings
        st.session_state.target_comp_budget = settings['target_comp_budget']
        st.session_state.currency_symbol = settings['currency']
        st.session_state.organization_name = settings['company_name']

        # Aggressively convert numerical columns to float64 to prevent type issues with SHAP
        for col in ['salary', 'experience', 'budget', 'target_salary']:
            if col in employee_df.columns:
                employee_df[col] = pd.to_numeric(employee_df[col], errors='coerce').astype(float)
        employee_df.dropna(subset=['salary', 'experience'], inplace=True) # Drop rows with NaN in critical columns after conversion

        try:
            explainer = joblib.load(os.path.join(os.path.dirname(__file__), "models", "shap_explainers.pkl"))
        except:
            explainer = None
        return employee_df, benchmark_df, explainer

    try:
        employee_df, benchmark_df, explainer = load_data()
    except:
        st.error("Please ensure data files are present in the data directory")
        st.stop()

    # Dashboard Page
    if page == "Dashboard":
        st.title("Project Helix – CHRO Compensation Intelligence Dashboard")
        st.markdown(f"#### Quarterly Report | {st.session_state.organization_name} | April–June 2025")
        st.markdown("Built with 💡 by Helix Viewpoints™")
        
        # Executive Summary
        st.header("🧾 Executive Summary")

        # Dynamic Pay Compression Metric
        mid_management_roles = ["Senior Software Engineer", "Product Manager", "Tech Lead", "UI Designer", "UX Designer", "Software Engineer", "Associate Product Manager"]
        mid_management_df = employee_df[employee_df['role'].isin(mid_management_roles)]
        
        pay_compression_count = 0
        if not mid_management_df.empty and 'comp_ratio' in mid_management_df.columns:
            # Assuming pay compression if comp ratio is below 0.9 for mid-management
            pay_compression_count = mid_management_df[mid_management_df['comp_ratio'] < 0.9].shape[0]
            
        total_mid_management = mid_management_df.shape[0]
        pay_compression_pct = (pay_compression_count / total_mid_management * 100) if total_mid_management > 0 else 0
        
        # Dynamic Gender Pay Gap in Engineering
        engineering_df = employee_df[employee_df['department'] == 'Engineering']
        engineering_gender_gap_pct = 0
        if not engineering_df.empty and 'gender' in engineering_df.columns and 'salary' in engineering_df.columns:
            engineering_gender_medians = engineering_df.groupby('gender')['salary'].median()
            if len(engineering_gender_medians) > 1:
                engineering_gender_ratio = engineering_gender_medians.min() / engineering_gender_medians.max()
                engineering_gender_gap_pct = (1 - engineering_gender_ratio) * 100

        # Dynamic AI Optimization Equity (Best Achievable)
        # Simulate a very large budget to find the theoretical best equity score
        large_budget_cap = employee_df['salary'].sum() * 1000 # Assume an unconstrained budget for optimal equity
        optimized_df_for_equity = employee_df.copy()
        
        # Generate recommendations with a large budget to see ideal equity
        ideal_recommendations = generate_recommendations(optimized_df_for_equity, large_budget_cap / 10000000, employee_df['budget'].sum()) # Pass budget cap in crores and total organizational budget
        
        # Convert target salaries from formatted string to numeric for equity calculation
        optimized_df_for_equity['salary'] = [float(x.replace('₹', '').replace(',', '')) for x in ideal_recommendations['Target Salary']]
        
        # Calculate equity metrics on the ideally optimized dataframe
        overall_equity_metrics_optimal = calculate_equity_metrics(optimized_df_for_equity)
        optimal_ai_equity_score = overall_equity_metrics_optimal.get('equity_alignment_score', 0)

        st.success(f"🔍 {pay_compression_pct:.1f}% of mid-management shows pay compression.\n\n⚠️ Gender pay gap of {engineering_gender_gap_pct:.1f}% in Engineering.\n\n✅ With AI optimisation, your organisation can achieve up to {optimal_ai_equity_score:.1f}% pay equity.")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Employees", len(employee_df))
            st.metric("Current Salary Budget", f"{st.session_state.currency_symbol}{employee_df['salary'].sum():,.0f}")
        with col2:
            avg_salary = employee_df['salary'].mean()
            st.metric("Average Salary", f"{st.session_state.currency_symbol}{avg_salary/100000:.2f} L")
            st.metric("Target Salary Budget", f"{st.session_state.currency_symbol}{employee_df['budget'].sum():,.0f}")
        with col3:
            comp_ratio = calculate_comp_ratio(employee_df)
            st.metric("Average Comp Ratio", f"{comp_ratio:.2f}")
            # Budget Utilisation from Target
            total_allocated_budget = employee_df['budget'].sum()
            budget_utilization_from_target = (employee_df['salary'].sum() / total_allocated_budget) * 100 if total_allocated_budget > 0 else 0
            st.metric("Budget Utilisation (from Target)", f"{budget_utilization_from_target:.1f}%")

        # Salary Distribution
        st.subheader("Salary Distribution by Department")
        fig = px.box(employee_df, x='department', y='salary', 
                     title="Salary Distribution by Department",
                     color='department')
        st.plotly_chart(fig, use_container_width=True)

    # Employee Data Page
    elif page == "Employee Data":
        st.title("Employee Data Management")

        # Add summary of employee data
        total_employee_count = len(employee_df)
        current_total_salary_budget = employee_df['salary'].sum()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Employees", total_employee_count)
        with col2:
            st.metric("Current Total Salary Budget", f"₹{current_total_salary_budget/100000:.2f} L")
        
        # Department filter - Changed from dropdown to clickable cards
        st.subheader("Filter by Department")
        department_counts = employee_df['department'].value_counts().sort_index()
        
        # Add "All" option
        all_employees_count = len(employee_df)
        department_options = {"All": all_employees_count}
        department_options.update(department_counts.to_dict())

        # Initialize session state for selected department if not already set
        if "selected_dept" not in st.session_state:
            st.session_state.selected_dept = "All"

        # Create columns for the department cards, with a maximum of 4 cards per row
        num_cols = min(len(department_options), 4) # Limit to 4 columns per row for better layout
        cols_cards = st.columns(num_cols)

        card_index = 0
        for dept_name, count in department_options.items():
            with cols_cards[card_index % num_cols]:
                if st.button(f"**{dept_name}** ({count})", key=f"dept_card_{dept_name}"):
                    st.session_state.selected_dept = dept_name
                    st.session_state.page_number = 1 # Reset pagination on filter change
                    st.experimental_rerun() # Rerun to apply new filter and page number

            card_index += 1

        # Filter data based on department
        selected_dept = st.session_state.selected_dept # Use the selected department from session state
        filtered_df = employee_df if selected_dept == "All" else employee_df[employee_df['department'] == selected_dept]
        
        # Calculate comp ratio for each employee (ensure target_salary exists to prevent division by zero)
        if 'target_salary' in filtered_df.columns and not filtered_df['target_salary'].isnull().all():
            filtered_df['comp_ratio'] = filtered_df['salary'] / filtered_df['target_salary']
        else:
            filtered_df['comp_ratio'] = 1.0 # Default or handle as appropriate

        # Pagination settings
        records_per_page = 9
        total_records = len(filtered_df)
        total_pages = (total_records + records_per_page - 1) // records_per_page

        if "page_number" not in st.session_state:
            st.session_state.page_number = 1

        # Navigation buttons
        col_prev, col_status, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("Previous", disabled=st.session_state.page_number <= 1):
                st.session_state.page_number -= 1
                st.experimental_rerun()
        with col_status:
            st.markdown(f"<div style='text-align: center;'>Page {st.session_state.page_number} of {total_pages}</div>", unsafe_allow_html=True)
        with col_next:
            if st.button("Next", disabled=st.session_state.page_number >= total_pages):
                st.session_state.page_number += 1
                st.experimental_rerun()

        # Calculate start and end index for current page
        start_idx = (st.session_state.page_number - 1) * records_per_page
        end_idx = start_idx + records_per_page
        
        # Slice the dataframe for the current page
        display_df = filtered_df.iloc[start_idx:end_idx]

        # Create columns for the grid layout
        cols = st.columns(3)
        
        # Display employee cards
        for idx, employee in display_df.iterrows():
            # Determine comp ratio class
            comp_ratio = employee['comp_ratio']
            if comp_ratio >= 0.95:
                ratio_class = "comp-ratio-high"
                ratio_status = "Optimal"
            elif comp_ratio >= 0.85:
                ratio_class = "comp-ratio-medium"
                ratio_status = "Moderate"
            else:
                ratio_class = "comp-ratio-low"
                ratio_status = "Low"

            # Calculate meter style
            # Map comp_ratio from 0.7 to 1.2 to 0-100% width and red-green gradient
            normalized_ratio = max(0, min(1, (comp_ratio - 0.7) / (1.2 - 0.7))) # Normalize to 0-1 range
            red_val = int(255 * (1 - normalized_ratio))
            green_val = int(255 * normalized_ratio)
            meter_color = f"rgb({red_val}, {green_val}, 0)"
            meter_width = int(normalized_ratio * 100) # Percentage width
            
            with cols[idx % 3]:
                st.markdown(f"""
                    <div class="employee-card">
                        <h3>{employee['name']}</h3>
                        <div class="comp-ratio-meter-container">
                            <div class="comp-ratio-meter-bar" style="width: {meter_width}%; background-color: {meter_color};"></div>
                        </div>
                        <p><strong>Department:</strong> {employee['department']}</p>
                        <p><strong>Role:</strong> {employee['role']}</p>
                        <p><strong>Experience:</strong> {employee['experience']} years</p>
                        <p><strong>Current Salary:</strong> <span class="metric-value">₹{employee['salary']/100000:.2f} L</span></p>
                        <p><strong>Target Salary:</strong> <span class="metric-value">₹{employee['target_salary']/100000:.2f} L</span></p>
                        <p><strong>Gender:</strong> {employee['gender']}</p>
                        <p><strong>Comp Ratio:</strong> <span class="metric-value">{comp_ratio:.2f}</span></p>
                        <div class="comp-ratio {ratio_class}">{ratio_status}</div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Add spacing
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Export Button
        if st.button("Export Updated Data"):
            # Export the original filtered_df, not just the displayed page
            filtered_df.to_csv("data/employee_data.csv", index=False)
            st.success("Data exported successfully!")

    # Data Management Page
    elif page == "Data Management":
        st.title("Data Management")
        st.header("🔄 Update Your Data")
        st.markdown("Upload your latest employee and benchmark data. Use the provided templates for correct formatting.")

        data_dir = os.path.join(os.path.dirname(__file__), "data")
        employee_data_path = os.path.join(data_dir, "employee_data.csv")
        benchmark_data_path = os.path.join(data_dir, "benchmark_data.csv")

        col_emp, col_bench = st.columns(2)

        with col_emp:
            st.subheader("Employee Data (employee_data.csv)")
            uploaded_employee_file = st.file_uploader("Upload new employee_data.csv", type=["csv"], key="employee_upload")
            if uploaded_employee_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_employee_file)
                    uploaded_df.to_csv(employee_data_path, index=False)
                    st.success("Employee data updated successfully! Please refresh the page to see changes.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error processing employee data: {e}")
            
            with open(employee_data_path, "rb") as file:
                st.download_button(
                    label="Download Employee Data Template",
                    data=file,
                    file_name="employee_data_template.csv",
                    mime="text/csv",
                    key="download_employee_template"
                )

        with col_bench:
            st.subheader("Benchmark Data (benchmark_data.csv)")
            uploaded_benchmark_file = st.file_uploader("Upload new benchmark_data.csv", type=["csv"], key="benchmark_upload")
            if uploaded_benchmark_file is not None:
                try:
                    uploaded_df = pd.read_csv(uploaded_benchmark_file)
                    uploaded_df.to_csv(benchmark_data_path, index=False)
                    st.success("Benchmark data updated successfully! Please refresh the page to see changes.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error processing benchmark data: {e}")

            with open(benchmark_data_path, "rb") as file:
                st.download_button(
                    label="Download Benchmark Data Template",
                    data=file,
                    file_name="benchmark_data_template.csv",
                    mime="text/csv",
                    key="download_benchmark_template"
                )

    # Compensation Analysis Page
    elif page == "Compensation Analysis":
        st.title("Compensation Analysis")
        
        # Department filter - Changed from dropdown to clickable cards (copied from Employee Data, with separate session state key)
        st.subheader("Filter by Department")
        department_counts = employee_df['department'].value_counts().sort_index()
        all_employees_count = len(employee_df)
        department_options = {"All": all_employees_count}
        department_options.update(department_counts.to_dict())

        # Use a separate session state key for Compensation Analysis
        if "selected_dept_analysis" not in st.session_state:
            st.session_state.selected_dept_analysis = "All"

        num_cols = min(len(department_options), 4)
        cols_cards = st.columns(num_cols)
        card_index = 0
        for dept_name, count in department_options.items():
            with cols_cards[card_index % num_cols]:
                if st.button(f"**{dept_name}** ({count})", key=f"dept_card_analysis_{dept_name}"):
                    st.session_state.selected_dept_analysis = dept_name
                    st.experimental_rerun()
            card_index += 1

        selected_dept = st.session_state.selected_dept_analysis
        dept_df = employee_df if selected_dept == "All" else employee_df[employee_df['department'] == selected_dept]
        
        # Salary Range Analysis
        st.subheader("Salary Range Analysis")

        # Display key metrics for the selected department
        if not dept_df.empty:
            total_current_dept_salary = dept_df['salary'].sum()
            total_average_salary = dept_df['salary'].mean()
            total_target_dept_salary = dept_df['budget'].sum()
            current_average_comp_ratio = calculate_comp_ratio(dept_df)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Current Department Salary", f"{st.session_state.currency_symbol}{total_current_dept_salary:,.0f}")
            with col2:
                st.metric("Total Average Salary", f"{st.session_state.currency_symbol}{total_average_salary:,.0f}")
            with col3:
                st.metric("Total Target Department Salary", f"{st.session_state.currency_symbol}{total_target_dept_salary:,.0f}")
            with col4:
                st.metric("Current Average Comp Ratio", f"{current_average_comp_ratio:.2f}")
        else:
            st.info(f"No data available for {selected_dept} department.")

        salary_range = calculate_salary_range(dept_df)
        
        fig = go.Figure()
        fig.add_trace(go.Box(y=dept_df['salary'], name="Current Distribution"))
        fig.add_trace(go.Box(y=[salary_range['min'], salary_range['max']], name="Target Range", boxpoints=False))
        fig.update_layout(title=f"{selected_dept} Salary Range Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Compensation Recommendations
        st.subheader("Compensation Recommendations")
        
        # Calculate role-specific salary ranges and quartiles
        role_salary_ranges = {}
        for role in dept_df['role'].unique():
            role_df = dept_df[dept_df['role'] == role]
            if not role_df.empty:
                s_range = calculate_salary_range(role_df)
                min_s_internal = s_range['min']
                max_s_internal = s_range['max']

                # Incorporate benchmark data
                market_data = benchmark_df[benchmark_df['role'] == role]['market_median']
                
                if not market_data.empty:
                    market_median = market_data.values[0]
                    
                    # Define a band around the market median for recommended range
                    # This ensures the recommended range is competitive but also flexible
                    market_min = market_median * 0.85 # 85% of market median
                    market_max = market_median * 1.15 # 115% of market median

                    # Adjust min_s and max_s based on internal and market data
                    # Take the higher of internal min or market min for the lower bound
                    min_s = max(min_s_internal, market_min)
                    # Take the lower of internal max or market max for the upper bound
                    max_s = min(max_s_internal, market_max)
                else:
                    # If no market data, use internal percentiles
                    min_s = min_s_internal
                    max_s = max_s_internal
                
                # Ensure min_s is not greater than max_s after adjustments
                if min_s > max_s:
                    min_s = max_s * 0.9 # Set min to 90% of max if it somehow exceeds it

                q1_s = min_s + 0.25 * (max_s - min_s)
                q2_s = min_s + 0.50 * (max_s - min_s)
                q3_s = min_s + 0.75 * (max_s - min_s)
                role_salary_ranges[role] = {
                    'min': min_s,
                    'max': max_s,
                    'q1': q1_s,
                    'q2': q2_s,
                    'q3': q3_s
                }

        # Now, generate recommendations with role-specific ranges
        employees = []
        current_salaries = []
        recommended_ranges = []
        q1_values = []
        q2_values = []
        q3_values = []
        q4_values = []

        for idx, employee in dept_df.iterrows():
            employees.append(employee['name'])
            current_salaries.append(employee['salary'])
            
            role = employee['role']
            if role in role_salary_ranges:
                r_range = role_salary_ranges[role]
                recommended_ranges.append(f"₹{r_range['min']:,.2f} - ₹{r_range['max']:,.2f}")
                q1_values.append(f"₹{r_range['q1']:,.2f}")
                q2_values.append(f"₹{r_range['q2']:,.2f}")
                q3_values.append(f"₹{r_range['q3']:,.2f}")
                q4_values.append(f"₹{r_range['max']:,.2f}") # Q4 is max
            else:
                # Handle cases where a role might not have a calculated range (e.g., no data)
                recommended_ranges.append("N/A")
                q1_values.append("N/A")
                q2_values.append("N/A")
                q3_values.append("N/A")
                q4_values.append("N/A")

        recommendations = pd.DataFrame({
            'Employee': employees,
            'Current Salary': current_salaries,
            'Recommended Range': recommended_ranges,
            'Q1': q1_values,
            'Q2 (Median)': q2_values,
            'Q3': q3_values,
            'Q4 (Max)': q4_values
        })
        st.dataframe(recommendations)

        st.subheader("Individual Compensation Explanation")
        
        # Dropdown to select an employee for detailed explanation
        selected_employee_name = st.selectbox(
            "Select an employee to see their compensation explanation:",
            options=recommendations['Employee'].tolist(),
            key="employee_explanation_selection"
        )

        if selected_employee_name:
            employee_rec = recommendations[recommendations['Employee'] == selected_employee_name].iloc[0]
            current_salary = employee_rec['Current Salary']
            recommended_range = employee_rec['Recommended Range']
            
            # Assuming target_salary is within the recommended_range or derived from it
            # For this explanation, we will parse the recommended range and use a placeholder for adjustment
            try:
                min_salary_str = recommended_range.split(' - ')[0].replace('₹', '').replace(',', '')
                max_salary_str = recommended_range.split(' - ')[1].replace('₹', '').replace(',', '')
                min_salary = float(min_salary_str)
                max_salary = float(max_salary_str)
                # For simplicity, let's assume the target salary is the average of the recommended range for explanation
                target_salary = (min_salary + max_salary) / 2
                adjustment = target_salary - current_salary
                adjustment_percent = (adjustment / current_salary) * 100 if current_salary != 0 else 0
            except (ValueError, IndexError):
                target_salary = current_salary # Default if parsing fails
                adjustment = 0
                adjustment_percent = 0

            # Additional explanation variables
            selected_employee_data = employee_df[employee_df['name'] == selected_employee_name].iloc[0]
            employee_role = selected_employee_data['role']

            median_experience_for_role = employee_df[employee_df['role'] == employee_role]['experience'].median()
            median_experience_for_role_display = f"{median_experience_for_role:.1f}" if pd.notna(median_experience_for_role) else "N/A"

            market_median_role_series = benchmark_df[benchmark_df['role'] == employee_role]['market_median']
            market_median_role = market_median_role_series.values[0] if not market_median_role_series.empty else 0
            market_median_role_display = f"₹{market_median_role:,.0f}" if market_median_role > 0 else "N/A"

            # Dynamic Gender Equity Explanation with actual data
            gender_equity_explanation = "" # Renamed for consistency
            relevant_df_for_gender_gap = dept_df[dept_df['role'] == employee_role]

            if 'gender' in relevant_df_for_gender_gap.columns and len(relevant_df_for_gender_gap['gender'].unique()) > 1:
                gender_medians_role = relevant_df_for_gender_gap.groupby('gender')['salary'].median()

                male_median_salary = gender_medians_role.get('M', 0)
                female_median_salary = gender_medians_role.get('F', 0)

                if male_median_salary > 0 and female_median_salary > 0:
                    gender_ratio_role = min(male_median_salary, female_median_salary) / max(male_median_salary, female_median_salary)
                    gender_pay_gap_pct_role = (1 - gender_ratio_role) * 100

                    if selected_employee_data['gender'] == 'F':
                        if female_median_salary < male_median_salary:
                            gender_equity_explanation = f"As a female employee, this recommendation considers the {gender_pay_gap_pct_role:.1f}% gender pay gap observed in the {employee_role} role (median male salary: ₹{male_median_salary:,.0f}, median female salary: ₹{female_median_salary:,.0f}), aiming to reduce disparities and ensure fair compensation."
                        else:
                            gender_equity_explanation = "This recommendation reinforces fair compensation, as no significant gender pay gap was identified for this role, aligning with gender equity principles."
                    elif selected_employee_data['gender'] == 'M':
                        if male_median_salary < female_median_salary:
                             gender_equity_explanation = f"As a male employee, this recommendation is made within a framework that also addresses the {gender_pay_gap_pct_role:.1f}% gender pay gap observed in the {employee_role} role (median male salary: ₹{male_median_salary:,.0f}, median female salary: ₹{female_median_salary:,.0f}), ensuring overall fairness across genders."
                        else:
                            gender_equity_explanation = "This recommendation reinforces fair compensation, as no significant gender pay gap was identified for this role, aligning with gender equity principles."
                else:
                    gender_equity_explanation = "Gender equity considerations are applied across the organization to ensure fair pay practices, but specific gender pay gap data for this role is not fully available due to limited representation of one or both genders."
            else:
                gender_equity_explanation = "Gender equity considerations are applied across the organization to ensure fair pay practices, but specific gender pay gap data for this role is not available due to limited diversity."

            # Budget Constraints Explanation (remains general as specific budget impact per employee is from simulator)
            current_org_budget = employee_df['salary'].sum()
            budget_constraint_explanation = f"All recommendations are made with consideration for the overall organizational budget (Current total: ₹{current_org_budget:,.0f}). This ensures that while individual salaries are adjusted fairly, the total payroll remains sustainable and aligned with financial planning."

            st.markdown(f"""
            --- 
            **Employee:** {selected_employee_name}  
            **Current Salary:** ₹{current_salary:,.0f}  
            **Target Salary:** ₹{target_salary:,.0f}  
            **Adjustment:** {f"₹{adjustment:,.0f} ({adjustment_percent:.1f}%)" if adjustment_percent >= 0 else f"-₹{-adjustment:,.0f} ({adjustment_percent:.1f}%)"} 

            **Explanation:**  
            {selected_employee_name}’s recommended target salary is determined by several factors:
            - **Experience Alignment:** With {selected_employee_data['experience']:.1f} years of experience, compared to the median of {median_experience_for_role_display} years for a {employee_role} in your department, their compensation is adjusted to align with their tenure and expertise.
            - **Gender Equity:** {gender_equity_explanation}
            - **Market Alignment:** Their compensation is benchmarked against current market data for similar roles (e.g., {employee_role} market median: {market_median_role_display}). The recommended range aims to ensure their salary is competitive and aligns with industry standards.
            - **Budget Constraints:** {budget_constraint_explanation}

            As a result, {selected_employee_name} is recommended an adjustment, moving them closer to both internal equity and market standards, while respecting organizational budget constraints.
            """)

    # Market Insights Page
    elif page == "Market Insights":
        st.title("Market Compensation Insights")
        
        # Market Comparison
        st.subheader("Market Comparison")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(benchmark_df, x='role', y=['market_median', 'company_median'],
                         title="Market vs Company Compensation",
                         barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(benchmark_df[['role', 'market_median', 'company_median']], use_container_width=True)
        
        # Market Trends
        st.subheader("Market Trends")
        trend_data = benchmark_df.melt(id_vars=['role'], 
                                     value_vars=['market_median', 'company_median'],
                                     var_name='Source', value_name='Salary')
        col3, col4 = st.columns(2)
        with col3:
            fig = px.line(trend_data, x='role', y='Salary', color='Source',
                          title="Compensation Trends by Role")
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.dataframe(trend_data, use_container_width=True)

    # AI Insights Page
    elif page == "AI Insights":
        st.title("🧠 AI Insights")
        st.write("DEBUG: AI Insights page code is executing.")

        # Initialize slider_moved state if not already set
        if 'slider_moved' not in st.session_state:
            st.session_state.slider_moved = False

        def on_slider_change():
            st.session_state.slider_moved = True

        # Display employee data
        st.subheader("Employee Data")

        # Display overall equity alignment score
        overall_equity_metrics = calculate_equity_metrics(employee_df)
        overall_equity_score = overall_equity_metrics.get('equity_alignment_score', 0)
        st.metric("Overall Equity Alignment Score", f"{overall_equity_score:.1f}%")

        # Role-Specific Analysis
        st.header("Role-Specific Analysis")

        # Get unique roles from the DataFrame
        unique_roles = employee_df['role'].unique().tolist()
        # st.write("DEBUG: Unique roles in employee_df:")
        # st.dataframe(pd.DataFrame(unique_roles, columns=['value']))

        # Get unique genders from the DataFrame
        unique_genders = employee_df['gender'].unique().tolist()
        # st.write("DEBUG: Unique genders in employee_df:")
        # st.dataframe(pd.DataFrame(unique_genders, columns=['value']))

        selected_role = st.selectbox("Select a Role for Detailed Analysis", unique_roles)

        role_df = employee_df[employee_df['role'] == selected_role]

        if not role_df.empty:
            role_equity_metrics = calculate_equity_metrics(role_df)
            role_equity_score = role_equity_metrics.get('equity_alignment_score', 0)

            col1, col2 = st.columns(2)
            with col1:
                # Gender Pay Gap
                gender_medians = role_df.groupby('gender')['salary'].median()
                if len(gender_medians) > 1:
                    gender_ratio = gender_medians.min() / gender_medians.max()
                    gender_pay_gap_pct = (1 - gender_ratio) * 100
                    st.metric(f"Gender Pay Gap – {selected_role}", f"{gender_pay_gap_pct:.1f}%")
                else:
                    st.info(f"Not enough gender diversity in {selected_role} to calculate gender pay gap.")
            with col2:
                st.metric(f"Role Equity Score – {selected_role}", f"{role_equity_score:.1f}%")


            # Create gender-based salary distribution pie chart
            gender_salary = role_df.groupby('gender')['salary'].agg(['sum', 'count']).reset_index()
            gender_salary['avg_salary'] = gender_salary['sum'] / gender_salary['count']

            # Create pie chart for salary distribution
            fig = px.pie(
                gender_salary,
                values='sum',
                names='gender',
                title=f"Total Salary Distribution by Gender - {selected_role}",
                color='gender',
                color_discrete_map={'M': '#1E88E5', 'F': '#E91E63'}, # Changed to M/F
                labels={'sum': 'Total Salary', 'gender': 'Gender'},
                hole=0.4
            )

            # Update layout
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>" +\
                             "Total Salary: ₹%{value:,.0f}<br>" +\
                             "Average Salary: ₹%{customdata[0]:,.0f}<br>" +\
                             "Count: %{customdata[1]}<extra></extra>",
                customdata=[[row['avg_salary'], row['count']] for _, row in gender_salary.iterrows()],
                textfont_size=14,
                textfont_color='white'
            )

            fig.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                title=dict(
                    text=f"Total Salary Distribution by Gender - {selected_role}",
                    x=0.5,
                    xanchor='center',
                    yanchor='top',
                    font=dict(
                        size=14
                    )
                ),
                height=400,
                width=500,
                margin=dict(l=0, r=0, t=80, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            # Display the pie chart in a centered column
            col_left, col_chart, col_right = st.columns([1, 3, 1])
            with col_chart:
                st.plotly_chart(fig, use_container_width=False)

            # Add detailed metrics below the chart
            st.markdown("### Detailed Gender Metrics")

            # Check if both genders are present for the selected role
            genders_in_role = role_df['gender'].unique()
            if len(genders_in_role) < 2:
                st.warning(f"⚠️ Only {genders_in_role[0] if len(genders_in_role) > 0 else 'no'} employees found for the selected role ({selected_role}). Gender equity metrics may not be meaningful.")

            # Convert gender_salary to a dictionary for easier and safer lookup
            gender_metrics_dict = {}
            for index, row in gender_salary.iterrows():
                gender_metrics_dict[row['gender']] = {
                    'sum': row['sum'],
                    'count': row['count'],
                    'avg_salary': row['avg_salary']
                }

            # Safely get values, defaulting to 0 if gender not present
            male_data = gender_metrics_dict.get('M', {'sum': 0, 'count': 0, 'avg_salary': 0})
            female_data = gender_metrics_dict.get('F', {'sum': 0, 'count': 0, 'avg_salary': 0})

            male_avg = male_data['avg_salary']
            female_avg = female_data['avg_salary']
            male_count = male_data['count']
            female_count = female_data['count']
            male_total = male_data['sum']
            female_total = female_data['sum']

            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Average Salary (Male)",
                    f"₹{male_avg/100000:.2f} L"
                )
                st.metric(
                    "Average Salary (Female)",
                    f"₹{female_avg/100000:.2f} L"
                )
            
            with col2:
                st.metric(
                    "Male Employees",
                    male_count
                )
                st.metric(
                    "Female Employees",
                    female_count
                )
            
            with col3:
                st.metric(
                    "Total Salary (Male)",
                    f"₹{male_total/100000:.2f} L"
                )
                st.metric(
                    "Total Salary (Female)",
                    f"₹{female_total/100000:.2f} L"
                )

        # Compensation Adjustment Simulator
        st.header("🚀 Compensation Adjustment Simulator")
        st.write("Adjust salaries based on various factors and see the impact on your budget and equity.")
        
        # Calculate current total salary to set dynamic budget slider range
        current_total_salary = employee_df['salary'].sum()
        total_target_budget_from_data = employee_df['target_salary'].sum() if 'target_salary' in employee_df.columns else current_total_salary

        min_budget_cap = (current_total_salary * 0.85) / 10000000 # Limit to -15% of current total salary
        max_budget_cap = (current_total_salary * 1.25) / 10000000 # Limit to 25% above current total salary
        default_budget_cap = total_target_budget_from_data / 10000000 # Default to current total budget (from 'target_salary' column if available)

        budget_cap_crores = st.slider(
            "Budget Cap for Adjustments (Crores)",
            min_value=float(min_budget_cap),
            max_value=float(max_budget_cap),
            value=float(default_budget_cap),
            step=0.01,
            format="₹%.2f Cr"
        )

        # Generate recommendations automatically when slider changes
        with st.spinner('Generating recommendations...'):
            # Pass budget cap in actual amount, not crores for internal function
            epsilon = 0.1 # Small tolerance for floating point comparisons

            if abs(budget_cap_crores - default_budget_cap) < epsilon:
                # If slider is at default, use original target_salary values from employee_df
                recommendations_df = pd.DataFrame({
                    'Employee': employee_df['name'],
                    'Current Salary': employee_df['salary'].apply(lambda x: f'{st.session_state.currency_symbol}{x:,.2f}'),
                    'Target Salary': employee_df['target_salary'].apply(lambda x: f'{st.session_state.currency_symbol}{x:,.2f}'),
                    'Adjustment': (employee_df['target_salary'] - employee_df['salary']).apply(lambda x: f'{st.session_state.currency_symbol}{x:,.2f}'),
                    'Adjustment %': ((employee_df['target_salary'] - employee_df['salary']) / employee_df['salary'].replace(0, 1) * 100).round(1).apply(lambda x: f'{x:+.1f}%')
                })
                # Calculate final total and utilization for the default case
                final_total = employee_df['target_salary'].sum()
                total_budget_from_employee_data = employee_df['budget'].sum()
                utilization = (final_total / total_budget_from_employee_data) * 100 if total_budget_from_employee_data != 0 else 0
                recommendations_df.attrs['utilization'] = utilization
                recommendations_df.attrs['final_total'] = final_total
            else:
                # If slider is moved, use the generate_recommendations function with the adjusted budget cap
                recommendations_df = generate_recommendations(employee_df.copy(), budget_cap_crores * 10000000, employee_df['budget'].sum()) 
            
            if recommendations_df is not None:
                # Display total budget impact
                current_total_salary_sim = employee_df['salary'].sum() # Recalculate current total salary to ensure it's always fresh
                total_budget_impact = recommendations_df.attrs.get('final_total', 0) - current_total_salary_sim
                budget_utilization_sim = recommendations_df.attrs.get('utilization', 0)
                
                col_impact1, col_impact2, col_impact3, col_impact4 = st.columns(4)
                with col_impact1:
                    # st.write(f"DEBUG: Current Total Budget value = {current_total_salary_sim}")
                    st.metric("Current Total Budget", f"₹{current_total_salary_sim/100000:.2f} L")
                with col_impact2:
                    # st.write(f"DEBUG: Total Budget Impact value = {total_budget_impact}")
                    st.metric("Total Budget Impact", f"₹{total_budget_impact/100000:.2f} L")
                with col_impact3:
                    # st.write(f"DEBUG: Budget Utilization (Simulated) value = {budget_utilization_sim}")
                    st.metric("Budget Utilization (Simulated)", f"{budget_utilization_sim:.1f}%")

                # Updated Equity Alignment Score below the simulator
                # Calculate equity based on target_salary from the simulation
                simulated_df_for_equity = employee_df.copy()
                # Ensure 'Target Salary' column exists in recommendations_df and is numeric before assignment
                if 'Target Salary' in recommendations_df.columns:
                    # Clean 'Target Salary' string for conversion to numeric
                    simulated_df_for_equity['salary'] = recommendations_df['Target Salary'].replace(r'[₹,]', '', regex=True).astype(float)
                
                updated_equity_metrics = calculate_equity_metrics(simulated_df_for_equity)
                updated_equity_score = updated_equity_metrics.get('equity_alignment_score', 0)
                with col_impact4:
                    # st.write(f"DEBUG: Updated Equity Alignment Score value = {updated_equity_score}")
                    st.metric("Updated Equity Alignment Score", f"{updated_equity_score:.1f}%")

                st.subheader("Compensation Recommendations")
                st.dataframe(recommendations_df, use_container_width=True)

                # CSV Download Button
                csv = recommendations_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Recommendations as CSV",
                    data=csv,
                    file_name="compensation_recommendations.csv",
                    mime="text/csv",
                )

                st.subheader("\nIndividual Adjustment Explanation (SHAP)")
                
                # Removed SHAP explanation for now as per user request
                st.info("SHAP explanation feature is temporarily disabled.")

            else:
                st.warning("No recommendations generated. Please check the budget cap and data.")

    # Settings Page
    elif page == "Settings":
        st.title("⚙️ Settings")
        st.markdown("Manage application settings, including organization details and currency.")

        # Display current settings
        st.subheader("Current Settings")
        st.write(f"**Organization Name:** {st.session_state.organization_name}")
        st.write(f"**Currency Symbol:** {st.session_state.currency_symbol}")
        st.write(f"**Target Compensation Budget:** {st.session_state.currency_symbol}{st.session_state.target_comp_budget:,.0f}")

        st.subheader("Update Settings")
        with st.form("settings_form"):
            new_company_name = st.text_input("Organization Name", value=st.session_state.organization_name)
            new_currency_symbol = st.text_input("Currency Symbol", value=st.session_state.currency_symbol)
            new_target_comp_budget = st.number_input("Target Compensation Budget (Absolute Value)", 
                                                    value=st.session_state.target_comp_budget, 
                                                    min_value=0.0, format="%.0f")
            
            settings_submitted = st.form_submit_button("Save Settings")

            if settings_submitted:
                updated_settings = {
                    'company_name': new_company_name,
                    'currency': new_currency_symbol,
                    'target_comp_budget': new_target_comp_budget
                }
                settings_df = pd.DataFrame([updated_settings])
                settings_path = os.path.join(os.path.dirname(__file__), "data", "settings.csv")
                settings_df.to_csv(settings_path, index=False)
                st.success("Settings updated successfully! Please refresh the page to see changes.")
                st.rerun()

# Fallback for when not logged in
else:
    login_screen() 