import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.graph_objects as go

# Set the app to use the full browser width
st.set_page_config(
    page_title="Incremental Analysis",
    layout="wide"
)

def load_panel_data(panel_file):
    """
    Loads the panel file and validates that it contains the required sheets:
      - PURCHASE DATA
      - projection factor
    Returns two DataFrames (for PURCHASE DATA and projection factor) or (None, None) on error.
    """
    try:
        xl = pd.ExcelFile(panel_file, engine='openpyxl')
        required_sheets = ["PURCHASE DATA", "projection factor"]
        for sheet in required_sheets:
            if sheet not in xl.sheet_names:
                st.error(f"Panel file is missing the required sheet: '{sheet}'.")
                return None, None
        purchase_df = pd.read_excel(xl, sheet_name="PURCHASE DATA", engine='openpyxl')
        xf_df = pd.read_excel(xl, sheet_name="projection factor", engine='openpyxl')
        return purchase_df, xf_df
    except Exception as e:
        st.error(f"Error loading panel file: {e}")
        return None, None

def process_data(purchase_df, xf_df, product_list):
    """
    Processes the input data.
    
    Validates that:
      - PURCHASE DATA contains the exact columns 'SKU Name', 'SKU_Number', and 'HHID'
      - The product file has exactly one column with header 'SKU Name'
    
    Uses a progress bar to indicate processing steps.
    
    Returns the final merged DataFrame.
    """
    progress_bar = st.progress(0)
    
    # Step 1: Validate required columns in PURCHASE DATA
    required_columns = {"SKU Name", "SKU_Number", "HHID"}
    if not required_columns.issubset(purchase_df.columns):
        st.error(
            "Panel file's 'PURCHASE DATA' is missing required column(s): " +
            f"{required_columns - set(purchase_df.columns)}"
        )
        return None
    progress_bar.progress(10)
    
    # Step 2: Validate the product file
    if "SKU Name" not in product_list.columns or len(product_list.columns) != 1:
        st.error("Product file must have exactly one column with the header 'SKU Name'.")
        return None
    progress_bar.progress(20)
    
    st.success("Files validated successfully!")
    
    # Step 3: Filter purchase data by SKU Names in the product file
    filtered_purchase = purchase_df[purchase_df['SKU Name'].isin(product_list['SKU Name'])]
    progress_bar.progress(30)
    
    # Step 4: Remove duplicate combinations of HHID, SKU Name, and SKU_Number
    hhid_product_df = filtered_purchase[['HHID', 'SKU Name', 'SKU_Number']].drop_duplicates()
    hhid_product_df.rename(columns={'SKU Name': 'SKU_Name'}, inplace=True)
    progress_bar.progress(40)
    
    # Step 5: Merge with projection factor (from the panel file) on HHID
    df_raw = hhid_product_df.merge(xf_df[['HHID', 'XF']], on='HHID', how='left')
    df_raw.rename(columns={'XF': 'EF'}, inplace=True)
    progress_bar.progress(50)
    
    # Step 6: Calculate total EF using unique HHIDs
    base_ef_df = df_raw[['HHID', 'EF']].drop_duplicates()
    base_ef_sum = base_ef_df['EF'].sum()
    progress_bar.progress(60)
    
    # Step 7: Create pivot table to sum EF per SKU_Number and calculate Absolute Penetration
    pivot_df = df_raw.groupby('SKU_Number', as_index=False)['EF'].sum()
    pivot_df['Absolute Penetration'] = pivot_df['EF'] / base_ef_sum
    pivot_df = pivot_df.rename(columns={'SKU_Number': 'UPC12'})
    progress_bar.progress(70)
    
    # (Optionally display the pivot table)
    st.subheader("Pivot Table with Absolute Penetration")
    st.dataframe(pivot_df)
    
    # Step 8: Iteratively select SKU with the highest EF and remove corresponding HHIDs
    df_raw_copy = df_raw.copy()
    result = []
    while not df_raw_copy.empty:
        df_sum = df_raw_copy.groupby('SKU_Number').agg({'EF': 'sum'}).reset_index()
        top_upc = df_sum.loc[df_sum['EF'].idxmax()]
        result.append(top_upc)
        hh_ids_to_remove = df_raw_copy[df_raw_copy['SKU_Number'] == top_upc['SKU_Number']]['HHID'].unique()
        df_raw_copy = df_raw_copy[~df_raw_copy['HHID'].isin(hh_ids_to_remove)]
    progress_bar.progress(80)
    
    final_df = pd.DataFrame(result)
    # (Intermediate results such as df_raw and final_df are not displayed per user request)
    
    # Step 9: Compute Incremental Penetration from the iterative result
    total_xf = final_df['EF'].sum()
    final_df['Incremental Penetration'] = final_df['EF'] / total_xf
    progress_bar.progress(85)
    
    # Step 10: Merge pivot table values (after renaming) for Absolute Penetration
    pivot_df.rename(columns={'UPC12': 'SKU_Number'}, inplace=True)
    final_df = final_df.merge(pivot_df[['SKU_Number', 'Absolute Penetration']], on='SKU_Number', how='left')
    
    # Step 11: Compute Cannibalization Penetration, Cumulative Penetration, and White spaces
    final_df['Cannibalization Penetration'] = (final_df['Absolute Penetration'] - final_df['Incremental Penetration']).round(5)
    final_df['Cumulative Penetration'] = final_df['Incremental Penetration'].cumsum().round(5)
    final_df['White spaces'] = (
        final_df['Cumulative Penetration'] -
        final_df['Incremental Penetration'] -
        final_df['Cannibalization Penetration']
    ).round(5)
    progress_bar.progress(90)
    
    # Step 12: Merge with SKU mapping from purchase_df to add SKU Name and reorder columns
    sku_mapping = purchase_df[['SKU_Number', 'SKU Name']].drop_duplicates()
    final_df1 = final_df.merge(sku_mapping, on="SKU_Number", how="left")
    final_df1 = final_df1[
        ['SKU_Number', 'EF', 'Incremental Penetration', 'Absolute Penetration',
         'Cannibalization Penetration', 'Cumulative Penetration', 'White spaces', 'SKU Name']
    ]
    progress_bar.progress(100)
    
    return final_df1

def create_penetration_chart(final_df1, threshold=0.85):
    """
    Creates a stacked bar chart (waterfall-style) from the final table.
    
    The layers are:
      - Bottom: White spaces (displayed in white)
      - Middle: Cannibalization Penetration (grey)
      - Top: Incremental Penetration (bold color)
    
    SKUs are selected in descending order of Incremental Penetration until either
    the cumulative incremental penetration reaches the given threshold (fraction)
    or at least 10 SKUs are selected.
    
    Returns the Plotly Figure object, with y-values displayed as percentages.
    """
    # Sort the final dataframe by Incremental Penetration (descending)
    final_df1_sorted = final_df1.sort_values('Incremental Penetration', ascending=False).reset_index(drop=True)
    
    cumulative = 0.0
    i = 0
    # Ensure at least 10 SKUs are shown and/or cumulative incremental penetration is ≥ threshold
    while (i < len(final_df1_sorted)) and ((cumulative < threshold) or (i < 10)):
        cumulative += final_df1_sorted.iloc[i]['Incremental Penetration']
        i += 1
    subset_df = final_df1_sorted.iloc[:i]
    
    # Build the figure, converting fractional penetrations to percentages
    fig = go.Figure()
    
    # White spaces (bottom layer)
    fig.add_trace(
        go.Bar(
            x=subset_df['SKU Name'],
            y=subset_df['White spaces'] * 100,
            name='White spaces',
            marker_color='white'
        )
    )
    # Cannibalization (middle layer, grey)
    fig.add_trace(
        go.Bar(
            x=subset_df['SKU Name'],
            y=subset_df['Cannibalization Penetration'] * 100,
            name='Cannibalization Penetration',
            marker_color='gray'
        )
    )
    # Incremental (top layer, bold color)
    fig.add_trace(
        go.Bar(
            x=subset_df['SKU Name'],
            y=subset_df['Incremental Penetration'] * 100,
            name='Incremental Penetration',
            marker_color='#CB2026'
        )
    )
    
    # Adjust layout to show percentages and fill the screen
    fig.update_layout(
        barmode='stack',
        autosize=False,
        width=1600,   # Adjust as needed for your screen
        height=800,   # Adjust as needed for your screen
        title='Penetration by SKU',
        xaxis=dict(title='SKU Name', tickangle=-45),
        yaxis=dict(title='Penetration (%)'),
        legend=dict(title='Penetration Type'),
        margin=dict(l=40, r=40, t=80, b=80)
    )
    
    return fig

def main():
    st.title("Incremental Analysis")
    st.markdown("""
    **Instructions:**
    
    1. **Panel File (Excel):**  
       - Must contain a sheet called **PURCHASE DATA** with at least the columns `SKU Name`, `SKU_Number`, and `HHID`.  
       - Must contain a sheet called **projection factor** with at least the columns `HHID` and `XF`.
    
    2. **Product File (CSV):**  
       - Must have exactly one column with the header `SKU Name`.
    """)
    
    panel_file = st.file_uploader("Upload Panel File (Excel)", type=["xlsx"])
    product_file = st.file_uploader("Upload Product File (CSV)", type=["csv"])
    
    if panel_file is not None and product_file is not None:
        # Use session state to store the processed data so that the expensive processing isn't repeated on slider changes.
        if "final_df1" not in st.session_state:
            purchase_df, xf_df = load_panel_data(panel_file)
            if purchase_df is not None and xf_df is not None:
                try:
                    product_list = pd.read_csv(product_file)
                except Exception as e:
                    st.error(f"Error reading the product file: {e}")
                    return
                final_df1 = process_data(purchase_df, xf_df, product_list)
                if final_df1 is not None:
                    st.session_state["final_df1"] = final_df1
        
        # Only update the chart when the threshold slider is moved; data processing is not repeated.
        if "final_df1" in st.session_state:
            final_df1 = st.session_state["final_df1"]
            st.subheader("Final Processed Data")
            st.dataframe(final_df1)
            
            # Slider now displays percentage values (0% to 100%)
            threshold_percent = st.slider(
                "Select Cumulative Incremental Penetration Threshold (%) - Changing this updates the SKUs plotted in the chart",
                min_value=0,
                max_value=100,
                value=85,
                step=1
            )
            # Convert the percentage slider value to a fraction
            threshold = threshold_percent / 100.0
            st.markdown("---")  # Visual divider
            
            # Generate the chart using the selected threshold.
            fig = create_penetration_chart(final_df1, threshold)
            
            # Render the chart with full width.
            st.plotly_chart(fig, use_container_width=True)
            
            # Provide the Excel download link below the chart.
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine='xlsxwriter') as writer:
                final_df1.to_excel(writer, index=False, sheet_name="Output")
            towrite.seek(0)
            st.download_button(
                label="Download Final Data as Excel",
                data=towrite,
                file_name="premium_non_dairy_almond_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
