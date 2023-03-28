import pandas as pd
import matplotlib.pyplot as plt


def injury_cause_type_count(data):
    # Filter out the "Grand Total" row.  We want to avoid the totals amount for this visualization.
    data = data.query('`Row Labels` != "Grand Total"')

    # Get the top 20 counts
    top_20 = data.nlargest(20, 'Count of INJURY_CAUSE')

    # Create a bar chart of the top 20 counts
    ax = top_20.plot(kind='bar', x='Row Labels', y='Count of INJURY_CAUSE', figsize=(20, 25))

    # Set the title and axis labels
    plt.title('Top 20 Injury Causes by Count')
    plt.xlabel('Injury Cause')
    plt.ylabel('Count')

    # Add labels to the top of each bar
    for container in ax.containers:
        for i, rect in enumerate(container):
            # Get the height of the rectangle
            height = rect.get_height()
            # Get the label for the bar
            label = top_20.iloc[i]['Count of INJURY_CAUSE']
            # Add the label to the top of the bar
            ax.annotate(f'{label}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom')

    # Show the chart
    plt.show()


def injury_cause_type_by_normalized_incurred(data):
    # Filter out the "Grand Total" row.  We want to avoid the totals amount for this visualization.
    data = data.query('`Row Labels` != "Grand Total"')

    # Remove any non-numeric characters from the 'Sum of Normalized_Incurred' column
    data.loc[:, 'Sum of Normalized_Incurred'] = data.loc[:, 'Sum of Normalized_Incurred'].str.replace(',', '')

    # Convert the 'Sum of Normalized_Incurred' column to numeric data type
    data.loc[:, 'Sum of Normalized_Incurred'] = pd.to_numeric(data['Sum of Normalized_Incurred'])

    # Get the top 20 counts
    top_20 = data.nlargest(20, 'Sum of Normalized_Incurred')

    # Create a bar chart of the top 20 counts
    ax = top_20.plot(kind='bar', x='Row Labels', y='Sum of Normalized_Incurred', figsize=(20, 25))

    # Set the title and axis labels
    plt.title('Top 20 Sum of Normalized Incurred')
    plt.xlabel('Injury Cause')
    plt.ylabel('Normalized Incurred')

    # Add labels to the top of each bar
    for container in ax.containers:
        for i, rect in enumerate(container):
            # Get the height of the rectangle
            height = rect.get_height()
            # Get the label for the bar
            label = top_20.iloc[i]['Sum of Normalized_Incurred']
            # Add the label to the top of the bar
            ax.annotate(f'${label:,.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom')

    # Show the chart
    plt.show()
