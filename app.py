import pandas as pd
import streamlit as st
import numpy as np
import re
from io import BytesIO

# Function to process the uploaded file and generate the result DataFrame

def process_excel(file):

    pd.set_option('display.max_rows', None)

    filename = file

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

    ## Dictonary for Nodal Reactions (Base Plates); these are to be used to ignore the nodes with restraints
    #node_reactions_dict= pd.read_excel(
    #    filename,
    #    sheet_name= "Structure - Node Restraints",
    #    usecols="A",
    #    header=1,
     #   skiprows=[2]
    #)

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

    ##Removes Reaction Loads
    # mask = result_df['Node No.'].isin(node_reactions_dict['Node'])
    # mask = ~mask
    # result_df = result_df[mask]
                
    # Data Frame for Beam to Section
    beam_to_sect = pd.read_excel(
        filename,
        sheet_name="Structure - Members",
        usecols="A,H",
        header=1,
        skiprows=[2]
    )

    # Data Frame for Section to Name, and Mark
    sect_to_memb = pd.read_excel(
        filename,
        sheet_name="Structure - Section Properties",
        usecols="A,B,C",
        header=0,
        skiprows=[1]
    )


    # Maps and merges the member to it's appropriate section            
    result_df = pd.merge(
        result_df,
        pd.merge(beam_to_sect, sect_to_memb, on='Sect', how='left').rename(columns={'Memb': 'Member No.'}),
        on='Member No.',
        how='left'
    )

    # Rename Column to Section
    result_df = result_df.rename(columns={'Name': 'Section Property','Sect': 'Section #'})

    # Moves the columns to middle
    # Get a list of all the columns
    cols = list(result_df.columns)

    # Specify the columns you want to move and where you want to move them
    cols_to_move = ['Section #', 'Section Property', 'Mark']
    insert_after = 'Member No.'

    # Find the index of the column after which you want to insert the other columns
    idx = cols.index(insert_after)

    # Remove the columns to move from the list
    for col in cols_to_move:
       cols.remove(col)

    # Insert the columns at the correct position
    for col in reversed(cols_to_move):
        cols.insert(idx+1, col)

    # Reindex the DataFrame
    result_df = result_df[cols]
    
    # Drop the index column
    result_df = result_df.reset_index(drop=True)

    # Sort dataframe by nodes
    result_df = result_df.sort_values(
        by=['Node No.', 'Load Case', 'Member No.'])

    # Display the result DataFrame
    return result_df
    
#################

import pandas as pd

def process_reactions(file):
    
    pd.set_option('display.max_rows', None)

    filename = file

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
        sheet_name="Structure - Node Restraints",
        usecols="A",
        header=1,
        skiprows=[2]
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
            if sheet_name.endswith(f"Reactions - Case {load_case}"):
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

    # Create a new DataFrame to store values
    result_df = pd.DataFrame()

    # Iterate through rows of the original DataFrame
    # First for loop gets the key-value pairs of the dictionary, and the embedded for loop goes thru the dataframe related to the key
    for load_case, df in loads_dict.items():
        for index, row in df.iterrows():
            if str(row['Node']).isdigit():
                node = int(row['Node'])

                # Extract specific values and add them to the result DataFrame
                values = row[['Node','Force', 'Force.1', 'Force.2', 'Moment', 'Moment.1', 'Moment.2']].tolist()
                values.append(load_case)

                # Lookup the case title and append it to the values
                case_title = load_cases_df.loc[load_cases_df["Case"] == int(
                    load_case.split("_")[1]), "Title"].iloc[0]
                values.append(case_title)

                result_df = pd.concat([result_df, pd.DataFrame([values], columns=['Node No.', 'Axial Force', 'Y-Axis Shear',
                                                                                  'Z-Axis Shear', 'X-Axis Torsion', 'Y-Axis Moment', 'Z-Axis Moment', 'Load Case', 'LC Title'])], ignore_index=True)
            else:
                continue #skip the row
   
    # Drop the index column
    result_df = result_df.reset_index(drop=True)

    # Sort dataframe by nodes
    result_df = result_df.sort_values(by=['Node No.', 'Load Case'])

    # Display the result DataFrame
    return result_df

#################

def process_st_memb(file):
    # Open the file and read the content

    pd.set_option('display.max_rows', None)
    
    content = file.read().decode('utf-16')

    # Find the start and end indices of the relevant section
    start_index = content.find("STEEL MEMBER DESIGN DATA (m)")
    end_index = content.find("AS4100:2020 STEEL MEMBER DESIGN NOTES")

    # Extract the relevant section
    relevant_section = content[start_index:end_index].strip()

    # Find all occurrences of 'Group', 'Member list', and 'End' in the relevant section
    matches = re.findall(r'Group:\s*(\d+)\s*(.*?),\s*(.*?),\s*(.*?)$\s*Member list:\s*([\d,]+)', relevant_section, re.MULTILINE)

    # Create a dataframe from the matches
    result_df= pd.DataFrame(matches, columns=['Group', 'Type', 'End_A', 'End_B', 'Member List'])

    # Remove the '\r' from the 'End_A' and 'End_B' columns
    result_df['End_A'] = result_df['End_A'].str.replace('\r', '')
    result_df['End_B'] = result_df['End_B'].str.replace('\r', '')

    # Separate the members into it's own column
    result_df['Member List'] = result_df['Member List'].str.split(',')

    # Create a new DataFrame that includes the 'Group' column and the split 'Member List' columns
    result_df = pd.concat([result_df[['Group', 'Type', 'End_A', 'End_B']], result_df['Member List'].apply(pd.Series)], axis=1)

    # Renaming headers
    result_df.columns = ['Group', 'Type', 'End_A', 'End_B'] + ['Member_' + str(i+1) for i in range(result_df.shape[1]-4)]

    # Convert 'Group' to integer
    result_df['Group'] = result_df['Group'].map(lambda x: pd.to_numeric(x, errors='coerce'))

    # Fills NaN cells with a blank input
    result_df = result_df.fillna('')

    # Create a new dataframe with the required columns
    result_df_start_end = pd.DataFrame()
    result_df_start_end['Group'] = result_df['Group']
    result_df_start_end['Type'] = result_df['Type']
    result_df_start_end['End_A'] = result_df['End_A']
    result_df_start_end['End_B'] = result_df['End_B']
    result_df_start_end['Member_1'] = result_df['Member_1']
    result_df_start_end['Member_2'] = result_df.iloc[:, 5:].apply(lambda row: row[row != ''].values[-1] if row[row != ''].values[-1] != '' else None, axis=1)

    # Remove the decimal zero from the 'End Member' column
    result_df_start_end['Member_2'] = result_df_start_end['Member_2'].apply(lambda x: int(x) if x is not None else None)

    return result_df_start_end

#################

# Function to handle download on button click


def download_excel(df1, df2, df3):
    # Create a BytesIO buffer to store the Excel file
    excel_buffer = BytesIO()

    # Use the pandas to_excel function to write the DataFrame to the buffer
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df1.to_excel(writer, index=False, sheet_name="Beam End-to-End Loads")
        df2.to_excel(writer, index=False, sheet_name="Reaction Loads")
        df3.to_excel(writer, index=False, sheet_name='Steel Design Group-Members')

    # Set up Streamlit to download the buffer as a file
    st.download_button(
        label="Download Excel File",
        # key="download_button",
        # on_click=download_excel,
        # args=(data_frame,),
        data=excel_buffer.getvalue(),
        file_name="SGLoadsExtract.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

#################

def process_connect(df1, file):

    pd.set_option('display.max_rows', None)

    filename = file

    number_members_nodes = pd.read_excel(
    filename,
    sheet_name="Structure - Members",
    usecols="A,F,G",
    header=1,
    skiprows=[2]
    )

    node_coord = pd.read_excel(
    filename,
    sheet_name="Structure - Nodes",
    usecols="A,B,C,D",
    header=0,
    skiprows=[1]
    )

    # Creates a Dictionary of Member to Node
    member_to_node_A_dict = dict(zip(number_members_nodes['Memb'],number_members_nodes['Node A']))
    member_to_node_B_dict = dict(zip(number_members_nodes['Memb'],number_members_nodes['Node B']))

    # Creates a Dictionary of Node to Coordinates
    nodes_dict = node_coord.set_index('Node')[['X','Y','Z']].T.to_dict()

    # Error Fixing; converting objects into int64
    df1['Member_1'] = pd.to_numeric(df1['Member_1'], errors = "coerce")
    df1['Member_2'] = pd.to_numeric(df1['Member_2'], errors = "coerce")

    # Map the Members to the Nodes
    df1['Member_1_A'] = df1['Member_1'].map(member_to_node_A_dict)
    df1['Member_1_B'] = df1['Member_1'].map(member_to_node_B_dict)
    df1['Member_2_A'] = df1['Member_2'].map(member_to_node_A_dict)
    df1['Member_2_B'] = df1['Member_2'].map(member_to_node_B_dict)

    # Map the Coordinates for Each Node
    df1['Member_1_A_Coord'] = df1['Member_1_A'].map(nodes_dict)
    df1['Member_1_B_Coord'] = df1['Member_1_B'].map(nodes_dict)
    df1['Member_2_A_Coord'] = df1['Member_2_A'].map(nodes_dict)
    df1['Member_2_B_Coord'] = df1['Member_2_B'].map(nodes_dict)


    df2 = df1[['Group', 'Member_1', 'Member_1_A','Member_1_A_Coord','Member_1_B','Member_1_B_Coord', 'Member_2','Member_2_A','Member_2_A_Coord', 'Member_2_B','Member_2_B_Coord', 'End_A', 'End_B','Type']]

    # Function to check distances between two coordinates
    def euclidean_distance(coord1,coord2):
        return np.sqrt(sum((coord1[axis] - coord2[axis]) ** 2 for axis in ['X', 'Y', 'Z']))
    
    # Initialize a list for results
    nodes_pairs = []


    # Determines which nodes are the furthest from the data set.
    for _, row in df1.iterrows():
        # Get coordinates for Member 1 A and B
        coord_1a = row['Member_1_A_Coord']
        coord_1b = row['Member_1_B_Coord']
    
        # Get coordinates for Member 2 A and B
        coord_2a = row['Member_2_A_Coord']
        coord_2b = row['Member_2_B_Coord']

        # Calculate distances between all combinations of nodes
        distances = [
        euclidean_distance(coord_1a, coord_2a),
        euclidean_distance(coord_1a, coord_2b),
        euclidean_distance(coord_1b, coord_2a),
        euclidean_distance(coord_1b, coord_2b)
        ]

        # Find the maximum distance and corresponding nodes for this row
        max_distance = max(distances)

        # Determine which nodes have the maximum distance
        if max_distance == distances[0]:
            max_nodes = ('Member_1_A', 'Member_2_A')
        elif max_distance == distances[1]:
            max_nodes = ('Member_1_A', 'Member_2_B')
        elif max_distance == distances[2]:
            max_nodes = ('Member_1_B', 'Member_2_A')
        else:
            max_nodes = ('Member_1_B', 'Member_2_B')

        # Append the result for this row
        nodes_pairs.append(max_nodes)

    # Initialize an empty DataFrame for the result
    result_df = pd.DataFrame(columns=['Group', 'Start Member', 'Start Node', 'End Member', 'End Node'])

    # Iterate over the node pairs
    for i, (start_node, end_node) in enumerate(nodes_pairs):
        # Get the start and end members
        start_member = df1.loc[i, 'Member_1']
        end_member = df1.loc[i, 'Member_2']
        
        # Get the start and end nodes
        start_node_value = df1.loc[i, start_node]
        end_node_value = df1.loc[i, end_node]

        # Get Type of Member
        type = df1.loc[i, 'Type']

        # Get the End Connections
        end_type_A = df1.loc[i, 'End_A']
        end_type_B = df1.loc[i, 'End_B']
        
        # Get the group
        group = df1.loc[i, 'Group']
        
        # Append the row to the result DataFrame
        row_to_append = pd.DataFrame({
            'Group': [group],
            'Type': [type],
            'Start Member': [start_member],
            'Start Node': [start_node_value],
            'Start Connection': [end_type_A],
            'End Member': [end_member],
            'End Node': [end_node_value],
            'End Connection': [end_type_B]
        })
        result_df = pd.concat([result_df, row_to_append], ignore_index=True)

        result_df = result_df[['Group', 'Type','Start Member', 'Start Node', 'Start Connection', 'End Member', 'End Node', 'End Connection' ]]

    return result_df

#################

def filter_end(df1, df2):
    # Create a mask where 'Start Member' and 'Start Node' match 'Member No.' and 'Node No' in df1
    # and 'End Member' and 'End Node' match 'Member No.' and 'Node No' in df1
    mask = df1[['Member No.', 'Node No.']].apply(tuple, axis=1).isin(df2[['Start Member', 'Start Node']].apply(tuple, axis=1)) | df1[['Member No.', 'Node No.']].apply(tuple, axis=1).isin(df2[['End Member', 'End Node']].apply(tuple, axis=1))
    
    # Apply the mask to df1
    df1_filtered = df1[mask]

    return df1_filtered
    
#################

# Streamlit app


def main():
    st.title("SpaceGass Excel Processing App for Connection Design")
    st.subheader("Version 0.3.1")
    st.caption("Created by: Emmanuel Domingo (Contact for any issues)")

    # File upload
    uploaded_file_excel = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    uploaded_file_txt = st.file_uploader("Upload Text File", type=[".txt"])

    if uploaded_file_excel and uploaded_file_txt is not None:
        st.success("File uploaded successfully!")

        # Process the uploaded file
        df1 = process_excel(uploaded_file_excel)
        df2 = process_reactions(uploaded_file_excel)
        df3 = process_st_memb(uploaded_file_txt)

        # F
        beam_end_df = process_connect(df3,uploaded_file_excel)

        df4 = filter_end(df1,beam_end_df)

        # Display the result DataFrame
        st.write("Beam End-to-End Loads:")
        st.write(df4)
        
        st.write("Reaction Loads:")
        st.write(df2)

        st.write("Steel Design Member Groups / Members")
        st.write(beam_end_df)

        # Download button
        download_excel(df4, df2, df3)


if __name__ == "__main__":
    main()
