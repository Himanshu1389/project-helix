import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime

class CompensationChatbot:
    """
    A basic chatbot for answering questions about compensation data.
    This is a demo module that simulates AI responses without actual LLM integration.
    """
    
    def __init__(self, employee_data: pd.DataFrame, benchmark_data: pd.DataFrame):
        self.employee_data = employee_data
        self.benchmark_data = benchmark_data
        self.conversation_history = []
        self.context = self._build_context()
        
    def _build_context(self) -> Dict:
        """Build context about the organization's compensation data"""
        context = {
            'total_employees': len(self.employee_data),
            'departments': self.employee_data['department'].unique().tolist(),
            'roles': self.employee_data['role'].unique().tolist(),
            'total_salary_budget': self.employee_data['salary'].sum(),
            'avg_salary': self.employee_data['salary'].mean(),
            'gender_distribution': self.employee_data['gender'].value_counts().to_dict(),
            'experience_ranges': {
                'min': self.employee_data['experience'].min(),
                'max': self.employee_data['experience'].max(),
                'avg': self.employee_data['experience'].mean()
            }
        }
        return context
    
    def _analyze_compensation_patterns(self) -> Dict:
        """Analyze compensation patterns for intelligent responses"""
        patterns = {}
        
        # Department analysis
        dept_analysis = self.employee_data.groupby('department').agg({
            'salary': ['mean', 'median', 'std'],
            'experience': 'mean',
            'gender': 'count'
        }).round(2)
        patterns['department_analysis'] = dept_analysis.to_dict()
        
        # Role analysis
        role_analysis = self.employee_data.groupby('role').agg({
            'salary': ['mean', 'median', 'std'],
            'experience': 'mean'
        }).round(2)
        patterns['role_analysis'] = role_analysis.to_dict()
        
        # Gender pay gap analysis
        gender_gaps = {}
        for dept in self.employee_data['department'].unique():
            dept_data = self.employee_data[self.employee_data['department'] == dept]
            if len(dept_data['gender'].unique()) > 1:
                gender_medians = dept_data.groupby('gender')['salary'].median()
                if len(gender_medians) > 1:
                    gap = (1 - gender_medians.min() / gender_medians.max()) * 100
                    gender_gaps[dept] = round(gap, 2)
        patterns['gender_gaps'] = gender_gaps
        
        return patterns
    
    def _generate_response(self, query: str, patterns: Dict) -> str:
        """Generate responses based on query analysis"""
        
        query_lower = query.lower()
        
        # Compensation benchmarking queries
        if any(word in query_lower for word in ['benchmark', 'market', 'competitive', 'industry']):
            return self._handle_benchmarking_query(query, patterns)
        
        # Equity and fairness queries
        elif any(word in query_lower for word in ['equity', 'fair', 'gender', 'diversity', 'gap']):
            return self._handle_equity_query(query, patterns)
        
        # Budget and cost queries
        elif any(word in query_lower for word in ['budget', 'cost', 'expense', 'financial', 'total']):
            return self._handle_budget_query(query, patterns)
        
        # Employee-specific queries
        elif any(word in query_lower for word in ['employee', 'individual', 'person', 'raise', 'promotion']):
            return self._handle_employee_query(query, patterns)
        
        # Department queries
        elif any(word in query_lower for word in ['department', 'team', 'group']):
            return self._handle_department_query(query, patterns)
        
        # Role queries
        elif any(word in query_lower for word in ['role', 'position', 'job', 'title']):
            return self._handle_role_query(query, patterns)
        
        # General compensation strategy
        elif any(word in query_lower for word in ['strategy', 'policy', 'structure', 'philosophy']):
            return self._handle_strategy_query(query, patterns)
        
        # Default response
        else:
            return self._handle_general_query(query, patterns)
    
    def _handle_benchmarking_query(self, query: str, patterns: Dict) -> str:
        """Handle benchmarking and market-related queries"""
        
        # Find relevant benchmark data
        relevant_roles = []
        for role in self.benchmark_data['role'].unique():
            if role.lower() in query.lower():
                relevant_roles.append(role)
        
        if relevant_roles:
            role_data = self.benchmark_data[self.benchmark_data['role'].isin(relevant_roles)]
            market_median = role_data['market_median'].mean()
            company_median = role_data['company_median'].mean()
            
            if market_median > company_median:
                gap = ((market_median - company_median) / company_median) * 100
                return f"📊 **Market Analysis for {', '.join(relevant_roles)}:**\n\n" \
                       f"• **Market Median:** ₹{market_median:,.0f}\n" \
                       f"• **Company Median:** ₹{company_median:,.0f}\n" \
                       f"• **Competitiveness Gap:** {gap:.1f}% below market\n\n" \
                       f"💡 **Recommendation:** Consider adjusting compensation to within 90-110% of market median to maintain competitiveness."
            else:
                return f"📊 **Market Analysis for {', '.join(relevant_roles)}:**\n\n" \
                       f"• **Market Median:** ₹{market_median:,.0f}\n" \
                       f"• **Company Median:** ₹{company_median:,.0f}\n" \
                       f"• **Status:** Competitive with market\n\n" \
                       f"✅ Your compensation is well-aligned with market standards."
        
        return "📊 **General Market Insights:**\n\n" \
               "Based on your benchmark data, I can help analyze specific roles or provide overall market competitiveness assessment. " \
               "What specific role or department would you like me to analyze?"
    
    def _handle_equity_query(self, query: str, patterns: Dict) -> str:
        """Handle equity and fairness-related queries"""
        
        gender_gaps = patterns.get('gender_gaps', {})
        if gender_gaps:
            worst_dept = max(gender_gaps.items(), key=lambda x: x[1])
            best_dept = min(gender_gaps.items(), key=lambda x: x[1])
            
            response = f"⚖️ **Equity Analysis:**\n\n" \
                      f"• **Gender Pay Gap by Department:**\n"
            
            for dept, gap in gender_gaps.items():
                if gap > 5:
                    status = "⚠️ Needs Attention"
                elif gap > 2:
                    status = "🟡 Monitor Closely"
                else:
                    status = "✅ Good"
                
                response += f"  - {dept}: {gap:.1f}% {status}\n"
            
            response += f"\n• **Key Insights:**\n" \
                       f"  - Highest gap: {worst_dept[0]} ({worst_dept[1]:.1f}%)\n" \
                       f"  - Lowest gap: {best_dept[0]} ({best_dept[1]:.1f}%)\n\n" \
                       f"💡 **Recommendations:**\n" \
                       f"1. Review compensation in {worst_dept[0]} department\n" \
                       f"2. Implement regular equity audits\n" \
                       f"3. Consider bias training for managers"
            
            return response
        
        return "⚖️ **Equity Analysis:**\n\n" \
               "I can analyze gender pay gaps, experience-based disparities, and role-based equity issues. " \
               "Would you like me to focus on a specific department or provide overall equity metrics?"
    
    def _handle_budget_query(self, query: str, patterns: Dict) -> str:
        """Handle budget and financial-related queries"""
        
        total_budget = self.context['total_salary_budget']
        avg_salary = self.context['avg_salary']
        employee_count = self.context['total_employees']
        
        # Calculate budget efficiency metrics
        dept_efficiency = {}
        for dept in self.employee_data['department'].unique():
            dept_data = self.employee_data[self.employee_data['department'] == dept]
            dept_avg = dept_data['salary'].mean()
            dept_efficiency[dept] = (dept_avg / avg_salary) * 100
        
        response = f"💰 **Budget Analysis:**\n\n" \
                  f"• **Total Salary Budget:** ₹{total_budget:,.0f}\n" \
                  f"• **Average Salary:** ₹{avg_salary:,.0f}\n" \
                  f"• **Budget per Employee:** ₹{total_budget/employee_count:,.0f}\n\n" \
                  f"• **Department Budget Efficiency (vs. company avg):**\n"
        
        for dept, efficiency in dept_efficiency.items():
            if efficiency > 110:
                status = "🔴 Above Average"
            elif efficiency < 90:
                status = "🟡 Below Average"
            else:
                status = "🟢 Balanced"
            
            response += f"  - {dept}: {efficiency:.1f}% {status}\n"
        
        response += f"\n💡 **Budget Optimization Tips:**\n" \
                   f"1. Review high-cost departments for efficiency opportunities\n" \
                   f"2. Consider performance-based compensation adjustments\n" \
                   f"3. Implement budget allocation based on revenue contribution"
        
        return response
    
    def _handle_employee_query(self, query: str, patterns: Dict) -> str:
        """Handle employee-specific queries"""
        
        # Extract employee name if mentioned
        employee_names = self.employee_data['name'].tolist()
        mentioned_employee = None
        for name in employee_names:
            if name.lower() in query.lower():
                mentioned_employee = name
                break
        
        if mentioned_employee:
            emp_data = self.employee_data[self.employee_data['name'] == mentioned_employee].iloc[0]
            role_avg = self.employee_data[self.employee_data['role'] == emp_data['role']]['salary'].mean()
            dept_avg = self.employee_data[self.employee_data['department'] == emp_data['department']]['salary'].mean()
            
            comp_ratio = emp_data['salary'] / role_avg if role_avg > 0 else 1
            
            response = f"👤 **Employee Analysis: {mentioned_employee}**\n\n" \
                      f"• **Role:** {emp_data['role']}\n" \
                      f"• **Department:** {emp_data['department']}\n" \
                      f"• **Experience:** {emp_data['experience']} years\n" \
                      f"• **Current Salary:** ₹{emp_data['salary']:,.0f}\n" \
                      f"• **Role Average:** ₹{role_avg:,.0f}\n" \
                      f"• **Department Average:** ₹{dept_avg:,.0f}\n" \
                      f"• **Competitiveness Ratio:** {comp_ratio:.2f}\n\n" \
                      f"💡 **Recommendations:**\n"
            
            if comp_ratio < 0.9:
                response += f"  - Consider salary adjustment to align with role average\n" \
                           f"  - Review performance and contribution\n" \
                           f"  - Plan for next review cycle"
            elif comp_ratio > 1.1:
                response += f"  - Salary is above role average\n" \
                           f"  - Ensure performance justifies premium\n" \
                           f"  - Consider career development opportunities"
            else:
                response += f"  - Salary is well-aligned with role average\n" \
                           f"  - Maintain current compensation structure\n" \
                           f"  - Focus on performance and growth"
            
            return response
        
        return "👤 **Employee Analysis:**\n\n" \
               "I can provide detailed analysis for specific employees, including:\n" \
               "• Compensation competitiveness\n" \
               "• Performance alignment\n" \
               "• Career development recommendations\n\n" \
               "Please mention the employee's name for specific analysis."
    
    def _handle_department_query(self, query: str, patterns: Dict) -> str:
        """Handle department-specific queries"""
        
        # Find mentioned department
        mentioned_dept = None
        for dept in self.employee_data['department'].unique():
            if dept.lower() in query.lower():
                mentioned_dept = dept
                break
        
        if mentioned_dept:
            dept_data = self.employee_data[self.employee_data['department'] == mentioned_dept]
            dept_stats = {
                'count': len(dept_data),
                'avg_salary': dept_data['salary'].mean(),
                'total_budget': dept_data['salary'].sum(),
                'avg_experience': dept_data['experience'].mean(),
                'roles': dept_data['role'].unique().tolist()
            }
            
            return f"🏢 **Department Analysis: {mentioned_dept}**\n\n" \
                   f"• **Team Size:** {dept_stats['count']} employees\n" \
                   f"• **Average Salary:** ₹{dept_stats['avg_salary']:,.0f}\n" \
                   f"• **Total Budget:** ₹{dept_stats['total_budget']:,.0f}\n" \
                   f"• **Average Experience:** {dept_stats['avg_experience']:.1f} years\n" \
                   f"• **Roles:** {', '.join(dept_stats['roles'])}\n\n" \
                   f"💡 **Insights:**\n" \
                   f"This department represents {dept_stats['count']/len(self.employee_data)*100:.1f}% of your workforce " \
                   f"with a budget allocation of {dept_stats['total_budget']/self.context['total_salary_budget']*100:.1f}%."
        
        return "🏢 **Department Analysis:**\n\n" \
               "I can analyze specific departments including:\n" \
               "• Team composition and size\n" \
               "• Budget allocation\n" \
               "• Salary distribution\n" \
               "• Role diversity\n\n" \
               "Which department would you like me to analyze?"
    
    def _handle_role_query(self, query: str, patterns: Dict) -> str:
        """Handle role-specific queries"""
        
        # Find mentioned role
        mentioned_role = None
        for role in self.employee_data['role'].unique():
            if role.lower() in query.lower():
                mentioned_role = role
                break
        
        if mentioned_role:
            role_data = self.employee_data[self.employee_data['role'] == mentioned_role]
            role_stats = {
                'count': len(role_data),
                'avg_salary': role_data['salary'].mean(),
                'min_salary': role_data['salary'].min(),
                'max_salary': role_data['salary'].max(),
                'avg_experience': role_data['experience'].mean(),
                'departments': role_data['department'].unique().tolist()
            }
            
            return f"💼 **Role Analysis: {mentioned_role}**\n\n" \
                   f"• **Count:** {role_stats['count']} employees\n" \
                   f"• **Average Salary:** ₹{role_stats['avg_salary']:,.0f}\n" \
                   f"• **Salary Range:** ₹{role_stats['min_salary']:,.0f} - ₹{role_stats['max_salary']:,.0f}\n" \
                   f"• **Average Experience:** {role_stats['avg_experience']:.1f} years\n" \
                   f"• **Departments:** {', '.join(role_stats['departments'])}\n\n" \
                   f"💡 **Insights:**\n" \
                   f"This role has a salary spread of ₹{role_stats['max_salary'] - role_stats['min_salary']:,.0f}, " \
                   f"indicating {'high' if (role_stats['max_salary'] - role_stats['min_salary']) > role_stats['avg_salary'] else 'moderate'} variability."
        
        return "💼 **Role Analysis:**\n\n" \
               "I can analyze specific roles including:\n" \
               "• Salary distribution\n" \
               "• Experience requirements\n" \
               "• Department placement\n" \
               "• Market competitiveness\n\n" \
               "Which role would you like me to analyze?"
    
    def _handle_strategy_query(self, query: str, patterns: Dict) -> str:
        """Handle compensation strategy queries"""
        
        # Analyze current compensation structure
        salary_distribution = self.employee_data['salary'].describe()
        experience_salary_correlation = self.employee_data['salary'].corr(self.employee_data['experience'])
        
        return f"🎯 **Compensation Strategy Analysis:**\n\n" \
               f"• **Current Structure:**\n" \
               f"  - Salary range: ₹{salary_distribution['25%']:,.0f} - ₹{salary_distribution['75%']:,.0f}\n" \
               f"  - Experience-salary correlation: {experience_salary_correlation:.2f}\n" \
               f"  - Department distribution: {len(self.context['departments'])} departments\n\n" \
               f"💡 **Strategic Recommendations:**\n" \
               f"1. **Performance-Based Structure:** Implement merit-based increases\n" \
               f"2. **Market Alignment:** Regular benchmarking against industry standards\n" \
               f"3. **Equity Focus:** Address identified pay gaps\n" \
               f"4. **Career Progression:** Clear salary bands for advancement\n" \
               f"5. **Budget Optimization:** Align compensation with organizational goals"
    
    def _handle_general_query(self, query: str, patterns: Dict) -> str:
        """Handle general queries with contextual information"""
        
        return f"🤖 **Compensation Assistant:**\n\n" \
               f"I'm here to help with your compensation planning needs! Here's what I can assist with:\n\n" \
               f"📊 **Data Analysis:**\n" \
               f"• Market benchmarking\n" \
               f"• Equity analysis\n" \
               f"• Budget optimization\n" \
               f"• Employee-specific insights\n\n" \
               f"💡 **Ask me about:**\n" \
               f"• How competitive are our salaries?\n" \
               f"• Are there gender pay gaps?\n" \
               f"• How can we optimize our budget?\n" \
               f"• What should we pay for [specific role]?\n" \
               f"• How does [employee name] compare to peers?\n\n" \
               f"Your organization has {self.context['total_employees']} employees across {len(self.context['departments'])} departments."
    
    def process_query(self, user_query: str) -> str:
        """Process user query and generate intelligent response"""
        
        # Add to conversation history
        timestamp = datetime.now().strftime("%H:%M")
        self.conversation_history.append({
            'timestamp': timestamp,
            'user': user_query,
            'assistant': None
        })
        
        # Analyze patterns for intelligent response
        patterns = self._analyze_compensation_patterns()
        
        # Generate response
        response = self._generate_response(user_query, patterns)
        
        # Update conversation history
        self.conversation_history[-1]['assistant'] = response
        
        return response
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history for display"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_suggested_queries(self) -> List[str]:
        """Get suggested queries for user guidance"""
        return [
            "How competitive are our salaries compared to the market?",
            "Are there any gender pay gaps in our organization?",
            "How can we optimize our compensation budget?",
            "What should we pay for a Senior Software Engineer?",
            "How does John Smith compare to his peers?",
            "What's our compensation strategy for Engineering roles?",
            "Are we paying fairly across departments?",
            "How can we improve equity in our compensation?",
            "What's the ROI of our current compensation structure?",
            "How should we structure raises for high performers?"
        ]

def create_chatbot_interface(employee_data: pd.DataFrame, benchmark_data: pd.DataFrame):
    """Create the Streamlit interface for the compensation chatbot"""
    
    st.header("🤖 Compensation Planning Assistant")
    st.markdown("Ask me anything about compensation planning, equity analysis, market benchmarking, and more!")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = CompensationChatbot(employee_data, benchmark_data)
    
    chatbot = st.session_state.chatbot
    
    # Sidebar with suggested queries
    with st.sidebar:
        st.subheader("💡 Suggested Questions")
        suggested_queries = chatbot.get_suggested_queries()
        
        for i, query in enumerate(suggested_queries):
            if st.button(query, key=f"suggested_{i}"):
                st.session_state.user_input = query
                st.rerun()
        
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            chatbot.clear_history()
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat display area
        chat_container = st.container()
        
        with chat_container:
            # Display conversation history
            for message in chatbot.get_conversation_history():
                # User message
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 10px 0;">
                    <strong>👤 You ({message['timestamp']}):</strong><br>
                    {message['user']}
                </div>
                """, unsafe_allow_html=True)
                
                # Assistant response
                if message['assistant']:
                    st.markdown(f"""
                    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #2196f3;">
                        <strong>🤖 Assistant ({message['timestamp']}):</strong><br>
                        {message['assistant']}
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        # Quick stats
        st.subheader("📊 Quick Stats")
        st.metric("Total Employees", len(employee_data))
        st.metric("Departments", len(employee_data['department'].unique()))
        st.metric("Total Budget", f"₹{employee_data['salary'].sum()/100000:.1f}L")
        
        # Equity indicator
        if 'gender' in employee_data.columns:
            gender_data = employee_data.groupby('gender')['salary'].median()
            if len(gender_data) > 1:
                gap = (1 - gender_data.min() / gender_data.max()) * 100
                if gap > 5:
                    st.error(f"⚠️ Gender Gap: {gap:.1f}%")
                elif gap > 2:
                    st.warning(f"🟡 Gender Gap: {gap:.1f}%")
                else:
                    st.success(f"✅ Gender Gap: {gap:.1f}%")
    
    # User input
    st.markdown("---")
    user_input = st.text_input(
        "Ask me about compensation planning:",
        key="user_input",
        placeholder="e.g., How competitive are our salaries?",
        help="Ask questions about compensation, equity, benchmarking, or specific employees"
    )
    
    if user_input and st.button("Send", type="primary"):
        with st.spinner("🤖 Analyzing your question..."):
            response = chatbot.process_query(user_input)
            st.rerun()
    
    # Clear input after processing
    if 'user_input' in st.session_state:
        del st.session_state.user_input