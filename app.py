
import pandas as pd
import streamlit as st
from io import BytesIO

# Function to process the uploaded file and generate the result DataFrame

def process_excel(file):

    pd.set_option('display.max_rows', None)

    filename = "Frame_Trial.xlsx"

    # Extracts case number into a list
    number_cases = pd.read_excel(
        filename,
        sheet_name="Loads - Load Case Titles",
        usecols="A",
        header=1,
        skiprows=[2]
    )

    case_list = []
    for index, row in number_cases.iterrows():
        case_list.append(row['Case'])

    # Extracts case titles into a list
    title_cases = pd.read_excel(
        filename,
        sheet_name="Loads - Load Case Titles",
        usecols="B",
        header=1,
        skiprows=[2]
    )

    # Iterates through "Title" (Titles)
    case_title_list = []
    for index, row in title_cases.iterrows():
        case_title_list.append(row['Title'])

    # Create load_cases_df
    load_cases_df = pd.DataFrame({'Case': case_list, 'Title': case_title_list})

    # Extracts number of nodes into a list
    number_nodes = pd.read_excel(
        filename,
        sheet_name="Structure - Nodes",
        usecols="A",
        header=0,
        skiprows=[1]
    )

    node_list = []
    for index, row in number_nodes.iterrows():
        node_list.append(row['Node'])

    # Dictionary to store DataFrames for each load case
    loads_dict = {}

    for load_case in case_list:
        key = f'Case_{load_case}'

        # Iterate through sheets to find the one ending with "Case {load_case}"
        found_sheet = None
        for sheet_name in pd.ExcelFile(filename).sheet_names:
            if sheet_name.endswith(f"and Moments - Case {load_case}"):
                found_sheet = sheet_name
                break

        # Loads the excel sheet into a dataframe
        if found_sheet:
            # Read the Excel sheet dynamically
            loads_dict[key] = pd.read_excel(filename,
                                            sheet_name=found_sheet,
                                            header=[1],
                                            skiprows=[2])
        else:
            print(f"Sheet for Case {load_case} not found!")

        # loads_dict[key] = pd.read_excel(filename,
        #                                 sheet_name=f"... Forces and Moments - Case {load_case}",
        #                                 header=[1],
        #                                 skiprows=[2]
        #                                 )

    for load_case, df in loads_dict.items():
        # Replace NaN values in the 'Memb' column with forward fill; fills blank spaces with previous value by index
        df['Memb'] = df['Memb'].ffill()

        # Update the DataFrame in the dictionary
        loads_dict[load_case] = df

    # loads_dict['Case_2']
    number_members_nodes = pd.read_excel(
        filename,
        sheet_name="Structure - Members",
        usecols="A,F,G",
        header=1,
        skiprows=[2]
    )

    # Dictonary for Nodal Reactions (Base Plates); these are to be used to ignore the nodes with restraints
    node_reactions_dict= pd.read_excel(
        filename,
        sheet_name= "Structure - Node Restraints",
        usecols="A",
        header=1,
        skiprows=[2]
    )

    # Creates boolean mask that checks if Nodes in the Reaction are 
    mask = number_members_nodes['Node A'].isin(node_reactions_dict['Node']) | number_members_nodes['Node B'].isin(node_reactions_dict['Node'])

    # Inverts the mask
    mask = ~mask

    # Overwrites the previous dataframe and removes Reaction Nodes
    number_members_nodes = number_members_nodes[mask]

    # Dictionary to store results
    node_members_dict = {}

    # Loop through nodes
    for node in node_list:
        # Filter rows to list only rows where Node A or B is equal to node in for loop
        filtered_rows = number_members_nodes[(number_members_nodes['Node A'] == node) | (number_members_nodes['Node B'] == node)]

        # Extract unique members and converts to a list for corresponding node
        unique_members = filtered_rows['Memb'].unique().tolist()

        # Add to dictionary where each key is a node and has unique members related to that node
        node_members_dict[f'Node_{node}'] = unique_members

    # Create a new DataFrame to store values
    result_df = pd.DataFrame()

    # Iterate through rows of the original DataFrame
    # First for loop gets the key-value pairs of the dictionary, and the embedded for loop goes thru the dataframe related to the key
    for load_case, df in loads_dict.items():
        for index, row in df.iterrows():
            node = int(row['Node'])
            memb = int(row['Memb'])

            # Check if the combination of 'Node' and 'Memb' is present in the dictionary
            key = f'Node_{node}'
            if key in node_members_dict and memb in node_members_dict[key]:

                # Extract specific values and add them to the result DataFrame
                values = row[['Node', 'Memb', 'Force', 'Shear',
                              'Shear.1', 'Torsion', 'Moment', 'Moment.1']].tolist()
                values.append(load_case)

                # Lookup the case title and append it to the values
                case_title = load_cases_df.loc[load_cases_df["Case"] == int(
                    load_case.split("_")[1]), "Title"].iloc[0]
                values.append(case_title)

                result_df = pd.concat([result_df, pd.DataFrame([values], columns=['Node No.', 'Member No.', 'Axial Force', 'Y-Axis Shear',
                                                                                  'Z-Axis Shear', 'X-Axis Torsion', 'Y-Axis Moment', 'Z-Axis Moment', 'Load Case', 'LC Title'])], ignore_index=True)

    # Drop the index column
    result_df = result_df.reset_index(drop=True)

    # Sort dataframe by nodes
    result_df = result_df.sort_values(
        by=['Node No.', 'Load Case', 'Member No.'])

    # Display the result DataFrame
    return result_df

# Function to handle download on button click


def download_excel(data_frame):
    # Create a BytesIO buffer to store the Excel file
    excel_buffer = BytesIO()

    # Use the pandas to_excel function to write the DataFrame to the buffer
    data_frame.to_excel(excel_buffer, index=False)

    # Set up Streamlit to download the buffer as a file
    st.download_button(
        label="Download Excel File",
        # key="download_button",
        # on_click=download_excel,
        # args=(data_frame,),
        data=excel_buffer,
        file_name="SGLoadsExtract.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# Streamlit app


def main():
    st.title("SpaceGass Excel Processing App for Connection Design")
    st.subheader("Version 0.3.1")

    st.caption("Created by: Emmanuel Domingo (Contact for any issues)")

    # File upload
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        # Process the uploaded file
        result_df = process_excel(uploaded_file)

        # Display the result DataFrame
        st.write(result_df)

        # Download button
        download_excel(result_df)


if __name__ == "__main__":
    main()
