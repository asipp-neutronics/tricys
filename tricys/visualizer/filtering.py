import pandas as pd

operators = [
    ["ge ", ">="],
    ["le ", "<="],
    ["lt ", "<"],
    ["gt ", ">"],
    ["ne ", "!="],
    ["eq ", "="],
    ["contains "],
    ["datestartswith "],
    ["is blank"],
]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find("{") + 1 : name_part.rfind("}")]
                value_part = value_part.strip()
                v0 = value_part
                if (value_part.startswith('"') and value_part.endswith('"')) or (
                    value_part.startswith("'") and value_part.endswith("'")
                ):
                    v0 = value_part[1:-1]

                return name, operator_type[0].strip(), v0
    return None, None, None


def filter_dataframe(df: pd.DataFrame, filter_query: str) -> pd.DataFrame:
    """
    Filters a DataFrame using the Dash filter query syntax with custom extensions.

    Supported extensions:
    - {val1, val2}: Set membership (OR logic)
    - is blank: Checks for None, NaN, or empty string

    Returns:
        The filtered DataFrame.
    """
    if not filter_query:
        return df

    filtering_expressions = filter_query.split(" && ")
    dff = df.copy()

    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator == "is blank":
            dff = dff.loc[
                dff[col_name].isnull()
                | (dff[col_name] == "")
                | (dff[col_name].astype(str).str.strip() == "")
            ]
            continue

        if not col_name:
            continue

        # Handle Set Syntax {val1, val2}
        if filter_value.startswith("{") and filter_value.endswith("}"):
            # This is a set membership operation
            # Remove braces and split by comma
            vals = [v.strip() for v in filter_value[1:-1].split(",")]

            # Convert values to the column type if possible for comparison
            col_type = dff[col_name].dtype
            typed_vals = []
            for v in vals:
                try:
                    if pd.api.types.is_numeric_dtype(col_type):
                        typed_vals.append(float(v) if "." in v else int(v))
                    else:
                        typed_vals.append(v)
                except ValueError:
                    typed_vals.append(v)

            dff = dff.loc[dff[col_name].isin(typed_vals)]
            continue

        # Convert filter_value to number if column is numeric
        if pd.api.types.is_numeric_dtype(dff[col_name]):
            try:
                filter_value = float(filter_value)
            except ValueError:
                pass  # Keep as string if conversion fails

        if operator in ("eq", "ne", "lt", "le", "gt", "ge"):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == "contains":
            dff = dff.loc[
                dff[col_name]
                .astype(str)
                .str.contains(str(filter_value), case=False, regex=False)
            ]
        elif operator == "datestartswith":
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].astype(str).str.startswith(str(filter_value))]

    return dff
