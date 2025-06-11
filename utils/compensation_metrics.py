import numpy as np
import pandas as pd

def calculate_comp_ratio(df):
    """
    Calculate the compensation ratio (actual salary / target salary)
    """
    if 'salary' not in df.columns or 'target_salary' not in df.columns:
        return 1.0
    return (df['salary'] / df['target_salary']).mean()

def calculate_salary_range(df, percentile_low=25, percentile_high=75):
    """
    Calculate the salary range for a department or role
    """
    if 'salary' not in df.columns:
        return {'min': 0, 'max': 0}
    
    min_salary = df['salary'].quantile(percentile_low/100)
    max_salary = df['salary'].quantile(percentile_high/100)
    
    return {
        'min': min_salary,
        'max': max_salary
    }

def calculate_market_position(df, benchmark_df):
    """
    Calculate the company's market position relative to market data
    """
    if 'role' not in df.columns or 'salary' not in df.columns:
        return 0
    
    role_medians = df.groupby('role')['salary'].median()
    market_medians = benchmark_df.set_index('role')['market_median']
    
    market_position = (role_medians / market_medians).mean()
    return market_position

def calculate_budget_impact(df, target_comp_ratio=1.0):
    """
    Calculate the budget impact of adjusting salaries to target comp ratio
    """
    if 'salary' not in df.columns or 'target_salary' not in df.columns:
        return 0
    
    current_budget = df['salary'].sum()
    target_budget = (df['target_salary'] * target_comp_ratio).sum()
    
    return {
        'current_budget': current_budget,
        'target_budget': target_budget,
        'difference': target_budget - current_budget,
        'percentage_change': ((target_budget - current_budget) / current_budget) * 100
    }

def calculate_equity_metrics(df):
    """
    Calculate equity metrics including experience-based and gender-based equity.
    Returns a dynamic equity alignment score based on actual metrics.
    
    Args:
        df (pd.DataFrame): Employee data with experience and gender columns
    
    Returns:
        dict: Dictionary containing equity metrics and alignment score
    """
    try:
        metrics = {}
        
        # Experience-based equity
        if 'experience' in df.columns:
            # Calculate experience-based salary correlation
            exp_corr = df['experience'].corr(df['salary'])
            metrics['experience_correlation'] = exp_corr
            
            # Calculate experience-based equity score (0-100)
            # Higher correlation means better alignment
            exp_score = max(0, min(100, (exp_corr + 1) * 50))  # Convert from [-1,1] to [0,100]
            metrics['experience_score'] = exp_score
            
            # Calculate experience quartiles
            df['exp_quartile'] = pd.qcut(df['experience'], 
                                       q=4, 
                                       labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                       duplicates='drop')
            
            # Calculate median salary by experience quartile
            exp_quartile_medians = df.groupby('exp_quartile')['salary'].median()
            metrics['experience_quartile_medians'] = exp_quartile_medians.to_dict()
        
        # Gender-based equity
        if 'gender' in df.columns:
            # Calculate gender pay gap
            gender_medians = df.groupby('gender')['salary'].median()
            gender_ratio = gender_medians.min() / gender_medians.max()
            metrics['gender_pay_ratio'] = gender_ratio
            
            # Calculate gender equity score (0-100)
            # Higher ratio means better equity
            gender_score = max(0, min(100, gender_ratio * 100))
            metrics['gender_score'] = gender_score
            
            # Calculate gender distribution
            gender_dist = df['gender'].value_counts(normalize=True)
            metrics['gender_distribution'] = gender_dist.to_dict()
        
        # Calculate overall equity alignment score
        if 'experience_score' in metrics and 'gender_score' in metrics:
            # Weight experience and gender scores equally
            overall_score = (metrics['experience_score'] + metrics['gender_score']) / 2
        elif 'experience_score' in metrics:
            overall_score = metrics['experience_score']
        elif 'gender_score' in metrics:
            overall_score = metrics['gender_score']
        else:
            overall_score = 0
            
        metrics['equity_alignment_score'] = round(overall_score, 1)
        
        return metrics
        
    except Exception as e:
        return {
            'error': str(e),
            'equity_alignment_score': 0
        }

def generate_recommendations(df, budget_cap, total_organizational_budget):
    """
    Generate compensation recommendations based on budget cap, equity metrics, and market position.
    Ensures no negative adjustments while maintaining budget constraints.
    
    Args:
        df (pd.DataFrame): Employee data
        budget_cap (float): Budget cap in crores (1 crore = 10 million)
        total_organizational_budget (float): The total budget from employee_data.csv (sum of 'budget' column).
    
    Returns:
        pd.DataFrame: Recommendations with current salary, target salary, and adjustment details
    """
    print(f"DEBUG: generate_recommendations called with budget_cap: {budget_cap}") # Debug statement
    try:
        # Create a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Calculate current total budget
        current_budget = df['salary'].sum()

        # Store these as 'initial_target_salary' before applying budget constraints
        df['initial_target_salary'] = df['target_salary'].copy() if 'target_salary' in df.columns else df['salary'].copy()

        if 'experience' in df.columns and df['experience'].sum() != 0: # Check if experience column exists and has non-zero sum
            exp_median = df['experience'].median()
            if exp_median != 0:
                df['exp_factor'] = df['experience'] / exp_median
                df['initial_target_salary'] = df['initial_target_salary'] * df['exp_factor']

        if 'gender' in df.columns and not df['gender'].empty: # Check if gender column exists and is not empty
            gender_medians = df.groupby('gender')['salary'].median()
            # Ensure there are at least two distinct genders to calculate ratio meaningfully
            if len(gender_medians) > 1:
                gender_ratio = gender_medians.min() / gender_medians.max()
                if gender_ratio < 0.95:  # If gender pay gap is more than 5%
                    df['gender_median_salary'] = df.groupby('gender')['salary'].transform('median')
                    # Avoid division by zero if gender_median_salary contains zeros
                    if (df['gender_median_salary'] != 0).all():
                        df['gender_factor'] = df['gender_median_salary'].max() / df['gender_median_salary']
                        df['initial_target_salary'] = df['initial_target_salary'] * df['gender_factor']

        # Now, determine final target_salary based on budget_cap and initial_target_salary
        # Calculate the total proposed budget if all initial_target_salaries were adopted
        total_proposed_initial_salary = df['initial_target_salary'].sum()

        if total_proposed_initial_salary <= budget_cap:
            # If the sum of initial_target_salaries is within the budget cap, adopt them directly
            df['target_salary'] = df['initial_target_salary'].copy()
        else:
            # If the sum of initial_target_salaries exceeds the budget cap, scale them down proportionally
            scale_factor = budget_cap / total_proposed_initial_salary
            df['target_salary'] = df['initial_target_salary'] * scale_factor

        # Ensure target salaries are not negative
        df['target_salary'] = df['target_salary'].clip(lower=0)

        # Calculate final adjustments and percentage adjustments
        df['adjustment'] = df['target_salary'] - df['salary']
        # Handle potential division by zero for adjustment_pct if salary is 0
        df['adjustment_pct'] = (df['adjustment'] / df['salary'].replace(0, 1) * 100).round(1)

        # Calculate final total and utilization
        final_total = df['target_salary'].sum()
        # Handle potential division by zero for utilization if total_organizational_budget is 0
        if total_organizational_budget != 0:
            utilization = (final_total / total_organizational_budget) * 100
        else:
            utilization = 0 # Or handle as per business logic, e.g., if total_organizational_budget is 0, utilization is undefined or 0

        # Create recommendations DataFrame
        recommendations = pd.DataFrame({
            'Employee': df['name'],
            'Current Salary': df['salary'].apply(lambda x: f'₹{x:,.2f}'),
            'Target Salary': df['target_salary'].apply(lambda x: f'₹{x:,.2f}'),
            'Adjustment': df['adjustment'].apply(lambda x: f'₹{x:,.2f}'),
            'Adjustment %': df['adjustment_pct'].apply(lambda x: f'{x:+.1f}%')
        })

        # Add utilization and final_total to recommendations attributes
        recommendations.attrs['utilization'] = utilization
        recommendations.attrs['final_total'] = final_total
        
        return recommendations
        
    except Exception as e:
        # Return a simple error message if something goes wrong
        return pd.DataFrame({
            'Employee': ['Error'],
            'Current Salary': ['Error'],
            'Target Salary': ['Error'],
            'Adjustment': ['Error'],
            'Adjustment %': [f'Error: {str(e)}']
        }) 