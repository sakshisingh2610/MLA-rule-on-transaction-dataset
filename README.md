# MLA-rule-on-transaction-dataset
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
# Loading the Data
data = pd.read_csv("C:\\Users\\DELL\\Downloads\\Online_Retail.csv\\Online_Retail.csv")
data.head()
# Exploring the columns of the data
data.columns
# Exploring the different regions of transactions
data.Country.unique()
# Stripping extra spaces in the description
data['Description'] = data['Description'].str.strip()
# Dropping the rows without any invoice number
data.dropna(axis = 0, subset =['InvoiceNo'], inplace = True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')
# Dropping all transactions which were done on credit
data = data[~data['InvoiceNo'].str.contains('C')]
# Transactions done in France
basket_France = (data[data['Country'] =="France"]
		.groupby(['InvoiceNo', 'Description'])['Quantity']
		.sum().unstack().reset_index().fillna(0)
		.set_index('InvoiceNo'))

# Transactions done in the United Kingdom
basket_UK = (data[data['Country'] =="United Kingdom"]
		.groupby(['InvoiceNo', 'Description'])['Quantity']
		.sum().unstack().reset_index().fillna(0)
		.set_index('InvoiceNo'))

# Transactions done in Portugal
basket_Por = (data[data['Country'] =="Portugal"]
		.groupby(['InvoiceNo', 'Description'])['Quantity']
		.sum().unstack().reset_index().fillna(0)
		.set_index('InvoiceNo'))

basket_Sweden = (data[data['Country'] =="Sweden"]
		.groupby(['InvoiceNo', 'Description'])['Quantity']
		.sum().unstack().reset_index().fillna(0)
		.set_index('InvoiceNo'))

# Defining the hot encoding function to make the data suitable
# for the concerned libraries
def hot_encode(x):
	if(x<= 0):
		return 0
	if(x>= 1):
		return 1
    
# Encoding the datasets
basket_encoded = basket_France.applymap(hot_encode)
basket_France = basket_encoded

basket_encoded = basket_UK.applymap(hot_encode)
basket_UK = basket_encoded

basket_encoded = basket_Por.applymap(hot_encode)
basket_Por = basket_encoded

basket_encoded = basket_Sweden.applymap(hot_encode)
basket_Sweden = basket_encoded

# Building the model
frq_items = apriori(basket_France, min_support = 0.05, use_colnames = True)

#France
# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())

#UK
frq_items = apriori(basket_UK, min_support = 0.01, use_colnames = True, low_memory=True)
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())

#Portugal
frq_items = apriori(basket_Por, min_support = 0.05, use_colnames = True)
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())

#sweden
frq_items = apriori(basket_Sweden, min_support = 0.05, use_colnames = True)
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())

import networkx as nx
import matplotlib.pyplot as plt

# Generate the association rules
association_rules = association_rules(frq_items, metric="lift", min_threshold=1)

# Convert the association rules to a list
association_rules_list = list(association_rules)

# Create a networkx graph
G = nx.Graph()

# Add the association rules to the graph
for rule in association_rules_list:
    antecedents = rule[0]
    consequents = rule[1]

    for antecedent in antecedents:
        for consequent in consequents:
            G.add_edge(antecedent, consequent)

# Plot the graph
nx.draw_networkx(G)
plt.show()

# After sorting the association rules for France, you can create a bar graph

# Set the number of top rules you want to visualize
top_n_rules = 10

# Select the top N rules with highest confidence and lift
top_rules = rules.head(top_n_rules)

# Create a bar graph to represent the top N rules for France
plt.figure(figsize=(10, 6))
plt.barh(range(top_n_rules), top_rules['confidence'], align='center', alpha=0.7, label='Confidence')
plt.barh(range(top_n_rules), top_rules['lift'], align='center', alpha=0.7, color='red', label='Lift')
plt.yticks(range(top_n_rules), top_rules['antecedents'].astype(str) + ' -> ' + top_rules['consequents'].astype(str))
plt.xlabel('Metrics Value')
plt.title(f'Top {top_n_rules} Association Rules for France')
plt.legend()
plt.gca().invert_yaxis()  # Invert the y-axis to display the most significant rules at the top
plt.show()

import matplotlib.pyplot as plt

# Calculate the item frequencies in the UK dataset
item_frequencies = basket_UK.sum().sort_values(ascending=False)

# Set the number of top items you want to visualize
top_n_items = 10

# Select the top N items with the highest purchase frequencies
top_items = item_frequencies.head(top_n_items)

# Create a bar chart to represent the top N items and their frequencies
plt.figure(figsize=(10, 6))
top_items.plot(kind='bar', color='skyblue')
plt.title(f'Top {top_n_items} Purchased Items in the UK')
plt.xlabel('Item Description')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Calculate the item frequencies in the Sweden dataset
item_frequencies_sweden = basket_Sweden.sum().sort_values(ascending=False)

# Set the number of top items you want to visualize
top_n_items_sweden = 10

# Select the top N items with the highest purchase frequencies in Sweden
top_items_sweden = item_frequencies_sweden.head(top_n_items_sweden)

# Create a bar chart to represent the top N items and their frequencies in Sweden
plt.figure(figsize=(10, 6))
top_items_sweden.plot(kind='bar', color='lightcoral')
plt.title(f'Top {top_n_items_sweden} Purchased Items in Sweden')
plt.xlabel('Item Description')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Calculate the item frequencies in the Portugal dataset
item_frequencies_portugal = basket_Por.sum().sort_values(ascending=False)

# Set the number of top items you want to visualize
top_n_items_portugal = 10

# Select the top N items with the highest purchase frequencies in Portugal
top_items_portugal = item_frequencies_portugal.head(top_n_items_portugal)

# Create a bar chart to represent the top N items and their frequencies in Portugal
plt.figure(figsize=(10, 6))
top_items_portugal.plot(kind='bar', color='lightgreen')
plt.title(f'Top {top_n_items_portugal} Purchased Items in Portugal')
plt.xlabel('Item Description')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Calculate the item frequencies in the France dataset
item_frequencies_france = basket_France.sum().sort_values(ascending=False)

# Set the number of top items you want to visualize
top_n_items_france = 10

# Select the top N items with the highest purchase frequencies in France
top_items_france = item_frequencies_france.head(top_n_items_france)

# Create a bar chart to represent the top N items and their frequencies in France
plt.figure(figsize=(10, 6))
top_items_france.plot(kind='bar', color='lightblue')
plt.title(f'Top {top_n_items_france} Purchased Items in France')
plt.xlabel('Item Description')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()





