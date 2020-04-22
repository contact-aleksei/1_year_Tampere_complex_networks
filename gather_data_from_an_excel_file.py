import yfinance as yf
import openpyxl

def retrieve_data(x):
    """input x which means row from A2 to x= A3285"""
    fname = 'us.xlsx'
    wb = openpyxl.load_workbook(fname)
    sheet = wb.get_sheet_by_name('Sheet1')
    stock_names=[]
    stock_data=[]
    rest_of_strings=[]
    stock_categories=[]
    i=1
    for rowOfCellObjects in sheet['A2':x]:     #3285']:
        for cellObj in rowOfCellObjects:
            
            string=cellObj.value
            
            if string != "RWFC.PK":
            # "RWFC.PK" gives nan somehow, removing this exception manually
            
                separator = '.'
                if separator in string:
                    rest_of_string = string.split(separator, 1)[0]
                    if rest_of_string not in rest_of_strings:
                        rest_of_strings.append(rest_of_string)
                        yf_data=yf.download(rest_of_string, start="2017-01-01", end="2019-12-31")
                    else:
                        continue
                else:
                    yf_data=yf.download(string, start="2017-01-01", end="2019-12-31")
            
            
            
            
            
            
                if yf_data.shape==(753,6):
                    stock_names.append(cellObj.value)
                    stock_data.append(yf_data)
                    i+=1
                    category_cell='C'+str(i)
                    stock_categories.append(sheet[category_cell].value)
                      
                
    return stock_names, stock_data, stock_categories
