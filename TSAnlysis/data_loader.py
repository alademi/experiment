import os
import zipfile
from datetime import datetime, timedelta
from distutils.util import strtobool

import pandas as pd


# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(
        full_file_path_and_name,
        replace_missing_vals_with="NaN",
        value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                    len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                    len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                                numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)
        base_name = os.path.splitext(os.path.basename(full_file_path_and_name))[0]
        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
            base_name
        )




def extract_data(loaded_data, base_name, frequency):
    """
    Extracts series data from the loaded DataFrame and saves each series to a CSV file.

    Parameters:
    loaded_data (pd.DataFrame): The input DataFrame containing 'series_value', 'start_timestamp', and 'series_name' columns.

    Outputs:
    CSV files named after each series in the 'output/' directory, containing the time series data.
    """
    series_values = loaded_data["series_value"]
    if base_name != "weather_dataset" :
        start_dates = loaded_data["start_timestamp"]
    series_names = loaded_data["series_name"]

    print(base_name)

    # Check if the start_date is missing, and assign a dummy timestamp if needed
   # if pd.isna(start_dates):
    #    start_dates = datetime(1970, 1, 1)  # Assign default timestamp (1970-01-01)

    for i in range(loaded_data.shape[0]):
        if base_name != "weather_dataset" :
            start_date = start_dates.iloc[i]
        else:
            start_date = datetime(1970, 1, 1)

        series_name = series_names.iloc[i]
        values = series_values.iloc[i]

        dates = [start_date + frequency_offsets[frequency](offset) for offset in range(len(values))]

        output = pd.DataFrame(values, index=dates, columns=[series_name])
        output.index.name = 'timestamp'

        # Ensure the output directory exists
        dir = f"output/{base_name}"
        os.makedirs(dir, exist_ok=True)
        output.to_csv(os.path.join(dir, f"{series_name}.csv"), index=True)

# Define a dictionary to map frequencies to appropriate offsets
frequency_offsets = {
    '4_seconds' : lambda x : timedelta(seconds=x*4),
    'minutely': lambda x : timedelta(minutes=x),
    '10_minutes':lambda x : timedelta(minutes=10),
    'half_hourly': lambda x : timedelta(hours=x/2),
    'hourly': lambda x : timedelta(hours=x),
    'daily': lambda x: timedelta(days=x),
    'weekly': lambda x: timedelta(weeks=x),
    'monthly': lambda x: pd.DateOffset(months=x),
    'yearly': lambda x: pd.DateOffset(years=x)
}


def extract_files(directory):
    # Ensure the target directory exists
    target_dir = "tsf_data"
    os.makedirs(target_dir, exist_ok=True)

    # Iterate over all the files in the given directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if it's a file and not a directory
        if os.path.isfile(file_path):
            # Assume the file is a zip file for this example
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # Extract all the contents into the target directory
                    zip_ref.extractall(target_dir)
                    print(f"Extracted {filename} into {target_dir}")
            except zipfile.BadZipFile:
                print(f"Failed to extract {filename} because it is not a zip file.")
            except Exception as e:
                print(f"An error occurred: {e}")

extract_files("input")

for filename in os.listdir("tsf_data"):
    file_path = os.path.join("tsf_data", filename)
    loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length, base_name = convert_tsf_to_dataframe(
        file_path)
    if not os.path.isdir(f"output/{base_name}"):
        extract_data(loaded_data, base_name, frequency)