from gather_data_from_an_excel_file import retrieve_data
import pandas
import csv
"""input x, which means getting tickers from corresponding rows from A2 to x= A3285,"""
x="A700"
stock_names, stock_data, stocks_categories = retrieve_data(x)
#
## writing stock names into csv, part
with open('stock_names_700.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(stock_names)
## writing stock names into csv, part
#
## we extract prices r=ln(equation)
import numpy
particular_stock_daily_prices=[]
daily_close_prices=[]
r_particular_stocks=[]
r_all_stocks=[]
dict_data_for_corr_matrix={}
j=0
                         # prices
for stock_data_frame in stock_data:
    r_particular_stocks=[0]*1000
    particular_stock_daily_prices=[]
    # Iterate over each row 
    for index, rows in stock_data_frame.iterrows():
        particular_stock_daily_prices= stock_data_frame['Adj Close'].values
    daily_close_prices.append(particular_stock_daily_prices)
    for i in range(1,len(particular_stock_daily_prices-5)):
        
        # The correlation coefficient is defined as  
        r=numpy.log(particular_stock_daily_prices[i]/particular_stock_daily_prices[i-1])
        r_particular_stocks[i]=r 
    # NOTE: that..
    r_particular_stocks.pop(0)
    # that we are removing first element, because of previous day
    # as if price for previous day is unavailable    
    r_all_stocks.append(r_particular_stocks)    	
    # Adding a new key value pair
    stock_name=stock_names[j]
    dict_data_for_corr_matrix[stock_name] = r_particular_stocks
    j=j+1
from pandas import DataFrame
df_for_corr_m = DataFrame(dict_data_for_corr_matrix,columns=stock_names)

# CORRELATION MATRIX
correlation_matrix=df_for_corr_m.corr()
correlation_matrix.to_csv('correlation_matrix_700.csv')
distance_matrix=correlation_matrix
# CORRELATION MATRIX IS SAVED
#
import numpy
import pandas as pd
correlation_matrix = pd.read_csv('correlation_matrix_700.csv', error_bad_lines=False)
#stock_names = pd.read_csv('stock_names_700.csv', error_bad_lines=False)
matrix = correlation_matrix.drop("Unnamed: 0", axis=1)

###############################################################################
#################Correlattion matrix is saved above"###########################
###############################################################################

# renaming columns to indices
new_names=[]
for k in range(0, 583):
    new_names.append(str(k))
matrix.columns=new_names
# renaming columns to indices

# A metric distance between a pair of stocks
array_matrix=matrix.to_numpy()
distance_matrix=array_matrix
for i in range(0, 583):
    for j in range(0, 583):
        distance_matrix[i,j]=numpy.sqrt( 2*(1-array_matrix[i,j]) )
        
distance_matrix_df=pd.DataFrame(distance_matrix)
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
###############################################################################
#################Distance matrix is created above##############################
###############################################################################

#############ORIGINAL NETWORK##################################################
G = nx.from_pandas_adjacency(distance_matrix_df)
print(nx.info(G))

#############THRESHOLDED NETWORK###############################################
threshold=1.50
distance_matrix_dictionary=distance_matrix_df.to_dict()
distance_matrix_dict_thresh=distance_matrix_dictionary
for i in range(0, 583):
    for j in range(0, 583):
        if distance_matrix_dictionary[i][j] > threshold:
            if i == j:
                continue
            if i < j:
                print(distance_matrix_dictionary[i][j])
                distance_matrix_dict_thresh[i][j]=1
            else:
                None
        else:
            distance_matrix_dict_thresh[i][j]=0
        
distance_matrix_dict_thresh_df=pd.DataFrame(distance_matrix_dict_thresh)
G_thresholded = nx.from_pandas_adjacency(distance_matrix_dict_thresh_df)
print(nx.info(G_thresholded))

#############MINIMUM SPANNING TREE#############################################
mst=nx.minimum_spanning_tree(G)
print(nx.info(G))


#############relabels the nodes to match the  stocks names#####################
G = nx.relabel_nodes(G,lambda x: stock_names[x])
mapping = dict(zip(stock_names,stocks_categories ))
nx.set_node_attributes(G, mapping, 'group')
    
    
###############################################################################
###############################################################################
degree_centrality=nx.degree_centrality(mst)
degree=nx.degree_centrality(mst)
eigenvector_centrality= nx.eigenvector_centrality(mst)
closeness_centrality= nx.closeness_centrality(mst)
betweenness_centrality= nx.betweenness_centrality(mst)
   
nx.set_node_attributes(mst, degree_centrality, "degree centrality")
nx.set_node_attributes(mst, degree,  "degree")
nx.set_node_attributes(mst, eigenvector_centrality, "eigenvector centrality")
nx.set_node_attributes(mst, closeness_centrality, "closeness centrality")
nx.set_node_attributes(mst, betweenness_centrality, "betweenness centrality")

df_degree=DataFrame.from_dict(degree, orient='index', columns=["degree"])
df_degree.insert(0, "stock names", stock_names, True) 
df_degree.insert(2, "stock names", stocks_categories, True) 
df_degree=DataFrame.sort_values(df_degree, by='degree', axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')

df_betweenness_centrality=DataFrame.from_dict(betweenness_centrality, orient='index', columns=["betweenness_centrality"])
df_betweenness_centrality.insert(0, "stock names", stock_names, True) 
df_betweenness_centrality.insert(2, "stock names", stocks_categories, True) 
df_betweenness_centrality=DataFrame.sort_values(df_betweenness_centrality, by='betweenness_centrality', axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')
###############################################################################
###############################################################################
degree_centrality_thresh=nx.degree_centrality(G_thresholded)
degree_thresh=nx.degree_centrality(G_thresholded)
eigenvector_centrality_thresh= nx.eigenvector_centrality(G_thresholded)
closeness_centrality_thresh= nx.closeness_centrality(G_thresholded)
betweenness_centrality_thresh= nx.betweenness_centrality(G_thresholded)
   
nx.set_node_attributes(G_thresholded, degree_centrality_thresh, "degree centrality")
nx.set_node_attributes(G_thresholded, degree_thresh,  "degree")
nx.set_node_attributes(G_thresholded, eigenvector_centrality_thresh, "eigenvector centrality")
nx.set_node_attributes(G_thresholded, closeness_centrality_thresh, "closeness centrality")
nx.set_node_attributes(G_thresholded, betweenness_centrality_thresh, "betweenness centrality")


df_degree_thresh=DataFrame.from_dict(degree_thresh, orient='index', columns=["degree"])
df_degree_thresh.insert(0, "stock names", stock_names, True) 
df_degree_thresh.insert(2, "stock names", stocks_categories, True) 
df_degree_thresh=DataFrame.sort_values(df_degree_thresh, by='degree', axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')

df_betweenness_centrality_thresh=DataFrame.from_dict(betweenness_centrality_thresh, orient='index', columns=["betweenness_centrality"])
df_betweenness_centrality_thresh.insert(0, "stock names", stock_names, True) 
df_betweenness_centrality_thresh.insert(2, "stock names", stocks_categories, True) 
df_betweenness_centrality_thresh=DataFrame.sort_values(df_betweenness_centrality_thresh, by='betweenness_centrality', axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')

###############################################################################
###############################################################################



###############################################################################
# Colouring nodes 
###############################################################################
# create empty list for node colors
node_color = []
Consumer_Cyclicals =0
Financials, Industrials =0,0
Healthcare, Technology, Basic_Materials=0,0,0
Energy, Utilities =0,0
Consumer_Non_Cyclicals =0
Telecommunications_Services = 0
All_ = 0
# for each node in the graph
for node in G.nodes(data=True):
    All_ +=1
    # if the node has the attribute then attach colour
    
    if 'Consumer Cyclicals' in node[1]['group']:
        node_color.append('blue')
        Consumer_Cyclicals+=1
    elif 'Financials' in node[1]['group']:
        node_color.append('red')
        Financials+=1
    elif 'Industrials' in node[1]['group']:
        node_color.append('green')
        Industrials+=1
    elif 'Healthcare' in node[1]['group']:
        node_color.append('yellow')
        Healthcare+=1
    elif 'Technology' in node[1]['group']:
        node_color.append('orange')
        Technology+=1
    elif 'Basic Materials' in node[1]['group']:
        node_color.append('brown') 
        Basic_Materials+=1
    elif 'Energy' in node[1]['group']:
        node_color.append('pink')
        Energy+=1
    elif 'Utilities' in node[1]['group']:
        node_color.append('violet')
        Utilities+=1
    elif 'Consumer Non-Cyclicals' in node[1]['group']:
        node_color.append('cyan')
        Consumer_Non_Cyclicals+=1
    elif 'Telecommunications Services' in node[1]['group']:
        node_color.append('grey')
        Telecommunications_Services +=1
original=[]
original.append(Consumer_Cyclicals)
original.append(Financials)
original.append(Industrials)
original.append(Healthcare)
original.append(Technology)
original.append(Basic_Materials)
original.append(Energy)
original.append(Utilities)
original.append(Consumer_Non_Cyclicals)
original.append(Telecommunications_Services)
original.append(All_)
###############################################################################
# Colouring nodes 
###############################################################################

def printing_network(G,node_color):
    #crates a list for edges and for the weights
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())

    #positions
    positions=nx.spring_layout(G)
    
    #Figure size
    plt.figure(figsize=(15,15))

    #draws nodes
    nc = nx.draw_networkx_nodes(G,positions,node_color=node_color,
                           node_size=150,alpha=0.8)
    
#    #Styling for labels
#    nx.draw_networkx_labels(G, positions, font_size=15, 
#                            font_family='sans-serif')
        
    #draws the edges
    nx.draw_networkx_edges(G, positions, edge_list=edges,style='dashed',width=0.1)
    cmap = plt.cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)

    #saves image
    plt.savefig("part1.png", format="PNG")
    plt.colorbar(nc)
    # displays the graph without axis
    plt.axis('off')
    plt.show() 

printing_network(G,node_color)
printing_network(mst,node_color)
printing_network(G_thresholded,node_color)

nx.write_gexf(G,"Original_network_700.gexf")
nx.write_gexf(mst,"MST_700.gexf")
nx.write_gexf(G_thresholded,"Thresholded network_700.gexf")






values = df_degree['stock names'].to_numpy()
values=values.tolist()[:20]

Consumer_Cyclicals =0
Financials, Industrials =0,0
Healthcare, Technology, Basic_Materials=0,0,0
Energy, Utilities =0,0
Consumer_Non_Cyclicals =0
Others = 0
All_ = 0
current=[]
for i in values:
    All_ = +1   
    if i[1]=='Consumer Cyclicals':
        Consumer_Cyclicals+=1
    elif i[1]=='Financials':
        Financials+=1    
    elif 'Industrials' in  i[1]:
        Industrials+=1       
    elif 'Healthcare' in  i[1]:
        Healthcare+=1
        print('yes')
    elif 'Technology' in  i[1]:
        Technology+=1
    elif 'Basic Materials' in  i[1]:
        Basic_Materials+=1
    elif 'Energy' in  i[1]:
        Energy+=1
    elif 'Utilities' in  i[1]:
        Utilities+=1
    elif 'Consumer Non-Cyclicals' in  i[1]:
        Consumer_Non_Cyclicals+=1
    elif 'Telecommunications Services'  in  i[1]:
        Telecommunications_Services=+1
        
current.append(Consumer_Cyclicals)
current.append(Financials)
current.append(Industrials)
current.append(Healthcare)
current.append(Technology)
current.append(Basic_Materials)
current.append(Energy)
current.append(Utilities)
current.append(Consumer_Non_Cyclicals)
current.append(Telecommunications_Services)
current.append(All_)


def division(n, d):
    return n / d if d else 0





values_betweenness = df_betweenness_centrality['stock names'].to_numpy()
values_betweenness=values_betweenness.tolist()[:20]

Consumer_Cyclicals =0
Financials, Industrials =0,0
Healthcare, Technology, Basic_Materials=0,0,0
Energy, Utilities =0,0
Consumer_Non_Cyclicals =0
Others = 0
All_ = 0
current_betweenness=[]
for i in values_betweenness:
    All_ = +1   
    if i[1]=='Consumer Cyclicals':
        Consumer_Cyclicals+=1
    elif i[1]=='Financials':
        Financials+=1    
    elif 'Industrials' in  i[1]:
        Industrials+=1       
    elif 'Healthcare' in  i[1]:
        Healthcare+=1
        print('yes')
    elif 'Technology' in  i[1]:
        Technology+=1
    elif 'Basic Materials' in  i[1]:
        Basic_Materials+=1
    elif 'Energy' in  i[1]:
        Energy+=1
    elif 'Utilities' in  i[1]:
        Utilities+=1
    elif 'Consumer Non-Cyclicals' in  i[1]:
        Consumer_Non_Cyclicals+=1
    elif 'Telecommunications Services'  in  i[1]:
        Telecommunications_Services=+1
        
current_betweenness.append(Consumer_Cyclicals)
current_betweenness.append(Financials)
current_betweenness.append(Industrials)
current_betweenness.append(Healthcare)
current_betweenness.append(Technology)
current_betweenness.append(Basic_Materials)
current_betweenness.append(Energy)
current_betweenness.append(Utilities)
current_betweenness.append(Consumer_Non_Cyclicals)
current_betweenness.append(Telecommunications_Services)
current_betweenness.append(All_)

percentage=[division(i, j) for i, j in zip(current, original)]
percentage_current_betweenness=[division(i, j) for i, j in zip(current_betweenness, original)]