
import os
import uuid
import re
import configparser
import requests
import redis
from sqlalchemy import create_engine
import json
import pandas as pd
from decimal import Decimal
from sqlalchemy import text
import hashlib
import argparse
from fastapi import FastAPI, Request
import uvicorn, json, datetime
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import re
#from sklearn.preprocessing import MinMaxScaler

config = configparser.RawConfigParser()
config.read('ConfigFile.properties')

rediscache = redis.StrictRedis(host=config.get('RedisSection', 'url'), charset='utf-8', ssl=True, decode_responses=True, port=config.get('RedisSection', 'port'))
queryttl=7*24*60*60

stylesmall = "<style> table { border-collapse: collapse; width: 100%;} th { text-align: center; padding: 4px;font-family: Arial !important; font-size: 12px !important;} td {text-align: left !important;padding: 4px !important;font-family: Arial !important; font-size: 12px !important;color: black !important;} tr:nth-child(even){background-color: #eee} tr:nth-child(odd){background-color: #fff} th {background-color: #fff;color: black;}</style>"
stylelarge = "<style> table { border-collapse: collapse; width: 100%;} th { text-align: center; padding: 4px;font-family: Arial !important; font-size: 16px !important;} td {text-align: left !important;padding: 4px !important;font-family: Arial !important; font-size: 16px !important;color: black !important;} tr:nth-child(even){background-color: #eee} tr:nth-child(odd){background-color: #fff} th {background-color: #fff;color: black;}</style>"

# string utility method
def check_substring_single_space(main_string, sub_string):
    main_string_processed = re.sub(' +', ' ', main_string).strip().lower()
    sub_string_processed = re.sub(' +', ' ', sub_string).strip().lower()
    return sub_string_processed in main_string_processed

# string utility method to format column names
def format_column_name(s):
    if s.casefold() == "index":
        return ""
    if s.startswith("vw"):
        underscore_index = s.find('_')
        modified_string = s[underscore_index+1:].replace('_', ' ') if underscore_index != -1 else s
    else:
        modified_string = s.replace('_', ' ')
    return ' '.join(word.capitalize() for word in modified_string.split())

def getmd5hash(inputstr):
    inputstr = re.sub(' +', ' ',inputstr).strip()
    return hashlib.md5(inputstr.encode('utf-8')).hexdigest()

# set chat cache
def set_chat_cache(sid, chat):
    rediscache.set(getmd5hash("sessionid" + sid),chat,ex=1200)


# get chat cache
def get_chat_cache(sid):
    chat = rediscache.get(getmd5hash("sessionid" + sid))
    if chat is not None:
        return chat
    else:
        return " none."

# set redis cache
def set_query_cache(qryprompt, text):
    rediscache.set(getmd5hash(qryprompt),text,ex=queryttl)


# get redis cache
def get_query_cache(qryprompt):
    returnsql = rediscache.get(getmd5hash(qryprompt))
    if returnsql is not None:
        return returnsql
    else:
        return ""

# string utility method to format column names
def format_column_name(s):
    if s.casefold() == "index":
        return ""

    if s.startswith("vw"):
        underscore_index = s.find('_')
        modified_string = s[underscore_index+1:].replace('_', ' ') if underscore_index != -1 else s
    else:
        modified_string = s.replace('_', ' ')
    return ' '.join(word.capitalize() for word in modified_string.split())

# DB engine
engine = create_engine(
    f'oracle+oracledb://:@',
    connect_args={
        "user": config.get('DatabaseSection', 'database.user'),
        "password": config.get('DatabaseSection', 'database.password'),
        "dsn": config.get('DatabaseSection', 'database.dsn'),
        "config_dir": config.get('DatabaseSection', 'database.config'),
        "wallet_location": config.get('DatabaseSection', 'database.config'),
        "wallet_password": config.get('DatabaseSection', 'database.walletpsswd'),
    })

def normalize_spaces(text):
    """Normalize spaces in the input text."""
    return re.sub(r'\s+', ' ', text).strip()


# global varibales
modified_prompt = ""
df = pd.DataFrame()
column_names = []
qry_display =""
graphFlag=False


def query_oci_llm(llmqry: str ):
    import oci
    compartment_id = config.get('OCI', 'serviceendpoint.ocid')
    CONFIG_PROFILE = "DEFAULT"
    ociconfig = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)
    # Service endpoint
    endpoint = config.get('OCI', 'serviceendpoint.url')
    endpointmodel = config.get('OCI', 'serviceendpoint.model')
    generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=ociconfig, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))
    chat_detail = oci.generative_ai_inference.models.ChatDetails()
    content = oci.generative_ai_inference.models.TextContent()
    content.text = llmqry
    message = oci.generative_ai_inference.models.Message()
    message.role = "USER"
    message.content = [content]
    chat_request = oci.generative_ai_inference.models.GenericChatRequest()
    chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
    chat_request.messages = [message]
    chat_request.max_tokens = 600
    chat_request.temperature = 1
    chat_request.frequency_penalty = 0
    chat_request.top_p = 0.75
    chat_request.top_k = -1
    chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=endpointmodel)
    chat_detail.chat_request = chat_request
    chat_detail.compartment_id = compartment_id
    chat_response = generative_ai_inference_client.chat(chat_detail)
    # Print result
    print("*****************Chat Result**************************")
    print(chat_response.data.chat_response.choices[0].message.content[0].text)
    returnStr = chat_response.data.chat_response.choices[0].message.content[0].text
    return returnStr.strip().upper()

def converse_with_llm_oci(prompt: str, sid: str):
    msg = """
    ## Instructions
    Below is a current question posted by the user along with the previous question. Figure out if the current question is a follow up. If this question is a follow up to previous question then return YES otherwise return NO.
    {0}
    """
    previous = get_chat_cache(sid)
    curr = prompt
    msg = msg.format("Current question: " + curr + " .Previous question: " + previous)

    global graphFlag
    if check_graphing_request(prompt):
        prompt = previous
        graphFlag = True
    else:
        graphFlag = False
        if query_oci_llm(msg) == 'YES':
            if len(prompt.split()) > 12:
                prompt = curr
            else:
                prompt = curr + ". " + previous
        else:
            prompt = curr
    set_chat_cache(sid, prompt)
    return prompt

def check_graphing_request(prompt: str):
    msg = """
    ## Instructions
    Below is a request posted by the user. Figure out if the request is to make a graph. The request must contain the word 'graph'. If it is a graphing request then return YES otherwise return NO.
    {0}
    """
    msg = msg.format(prompt)
    if query_oci_llm(msg) == 'YES':
        return True
    else:
        return False


def query(prompt: str, userid: str):
    # get the query from the AI model hosted on GPUs or cache
    query_cache = ""
    query_cache = get_query_cache(prompt)
    filter = config.get('QueryResult', 'filter.un')
    if filter != "N" and (userid is None or userid == ""):
        return "Database query could not be created. Please check the application config settings."

    if filter == "N":
        secFlg = False
    else:
        secFlg = True

    if query_cache != "":
        query_clean = query_cache
        query_clean = add_remove_un(secFlg, query_clean, userid)
        if check_substring_single_space(query_clean, "I do not know"):
            return "Database query could not be created. Please try again using a different prompt."
    else:
        query_raw = get_sql(prompt)
        # see how the query was constructed
        # if the SQL Generator AI returned -- I do not know -- then we have a problem!!
        if check_substring_single_space(query_raw, "I do not know"):
            return "Database query could not be created. Please try again using a different prompt."
        # clean up the raw query
        query_clean = clean_query(query_raw)
        query_clean = add_remove_un(secFlg, query_clean, userid)

    # global dataframe to display data from the database
    try:
        connection = engine.connect()
        query_clean = remove_quotes(query_clean)
        try:
            query_result = connection.execute(text(re.sub(r'\s*SELECT', 'SELECT',query_clean)))
            set_query_cache(prompt,query_clean)
        except Exception as e:
            if str(e).find("ORA-00937") != -1:
                query_raw = get_sql(query_clean + " -- please add a group by clause")
                if check_substring_single_space(query_raw, "I do not know"):
                    return "Database query was not be created. Please try again using a different prompt."
                query_clean = clean_query(query_raw)
                query_clean = remove_quotes(query_clean)
                query_clean = add_remove_un(secFlg, query_clean, userid)
                qry_display = query_clean
                query_result = connection.execute(text(re.sub(r'\s*SELECT', 'SELECT',query_clean)))
            elif (str(e).find("ORA-00942") != -1):
                query_raw = get_sql(query_clean + " -- use ACCOUNT_PAYABLES_TBL table only. do not JOIN tables")
                if (check_substring_single_space(query_raw, "I do not know")):
                    return "Database query was not be created. Please try again using a different prompt."
                query_clean = clean_query(query_raw)
                query_clean = remove_quotes(query_clean)
                query_clean = add_remove_un(secFlg, query_clean, userid)
                qry_display = query_clean
                query_result = connection.execute(text(re.sub(r'\s*SELECT', 'SELECT',query_clean)))
            else:
                raise
        column_names = query_result.keys()
        results = query_result.fetchall()
        list_of_results = [list(row) for row in results]

        converted_list = [tuple(item) for item in list_of_results]
    except Exception as e:
        print(e)
        return "Database query returned an error. Please try again using a different prompt."
    finally:
        connection.close()
        engine.dispose()

    # populate the global DF
    df = pd.DataFrame(converted_list, columns=column_names)
    df = df.rename(columns=lambda x: format_column_name(x))
    # htmldataframe = df.to_html(index=False).replace('\n','')
    if len(column_names) > 3:
        style = stylesmall
    else:
        style = stylelarge

    if graphFlag:
        chrthtml = draw_bar_chart(df)
        htmldataframe = """<html><head>{1}</head>{0}<table border=2><tr><td>{2}</td></tr></table></html>""".format(df.to_html(index=False).replace('\n',''),style,chrthtml)
    else:
        htmldataframe = """<html><head>{1}</head>{0}</html>""".format(df.to_html(index=False).replace('\n',''),style)

    print(htmldataframe)
    return htmldataframe



def draw_bar_chart(df_input):
    df_input = convert_to_numeric_if_possible(df_input)
    df=df_input.copy()

    # Automatically detect categorical and numerical columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    numerical_columns = df.select_dtypes(include=['number']).columns

    if len(categorical_columns) == 0 or len(numerical_columns) == 0:
        raise ValueError("DataFrame must have at least one categorical column and one numerical column")

    if (len(numerical_columns) > 2):
        df = df.iloc[:7, :]
    else:
        df = df.iloc[:15, :]

    # Assuming the first categorical column for x-axis
    x_col = categorical_columns[0]

    scale_columns=True
    if scale_columns:
        #scaler = MinMaxScaler()
        #df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        flattened_data = df[numerical_columns].values.flatten()
        # Perform Min-Max Scaling on the flattened data
        min_val = np.min(flattened_data)
        max_val = np.max(flattened_data)
        scaled_data = (flattened_data - min_val) / (max_val - min_val)
        # Reshape the scaled data back to original DataFrame shape
        df[numerical_columns] = pd.DataFrame(scaled_data.reshape(df[numerical_columns].shape), columns=df[numerical_columns].columns)

    # Plot the bar chart with all numerical columns
    df.plot(kind='bar', x=x_col, y=numerical_columns, colormap='viridis')

    # Add labels and title
    plt.xlabel(x_col)
    plt.ylabel(', '.join(numerical_columns))
    plt.title('Bar Chart of Numerical Columns by ' + x_col)

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)

    # Convert the BytesIO object to a base64 string
    img_str = base64.b64encode(buf.read()).decode('utf-8')

    # Create HTML code to embed the image
    html = f"<img style='display:block;width:400px;height:400px;' src='data:image/png;base64,{img_str}' alt='Chart'>"
    return html



def convert_to_numeric_if_possible(df):
    for col in df.select_dtypes(include=['object']).columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except Exception as e:
            print(e)
            continue
    return df


def sql_add_un_filter(rawsql, upnCondition ):
    strng1 = upnCondition + " and "
    strng2 = " where " + upnCondition + " "
    # Check for "where" (case insensitive)
    where_match = re.search(r'\bwhere\b', rawsql, re.IGNORECASE)
    if where_match:
        where_index = where_match.end()
        modified_sql = rawsql[:where_index] + " " + strng1 + " " + rawsql[where_index:]
        return modified_sql
    # Check for "group by" (case insensitive)
    group_by_match = re.search(r'\bgroup by\b', rawsql, re.IGNORECASE)
    if group_by_match:
        group_by_index = group_by_match.start()
        modified_sql = rawsql[:group_by_index] + " " + strng2 + " " + rawsql[group_by_index:]
        return modified_sql
    # Check for "order by" (case insensitive)
    order_by_match = re.search(r'\border by\b', rawsql, re.IGNORECASE)
    if order_by_match:
        order_by_index = order_by_match.start()
        modified_sql = rawsql[:order_by_index] + " " + strng2 + " " + rawsql[order_by_index:]
        return modified_sql
    # Check for "fetch" (case insensitive)
    fetch_match = re.search(r'\bfetch\b', rawsql, re.IGNORECASE)
    if fetch_match:
        fetch_index = fetch_match.start()
        modified_sql = rawsql[:fetch_index] + " " + strng2 + " " + rawsql[fetch_index:]
        return modified_sql
    # If none of the conditions are satisfied, append strng2 at the end
    modified_sql = rawsql + " " + strng2
    return modified_sql

def add_remove_un(securityFlag, qry, updatedUpn):
    pattern = r"upper\(upn\) = upper\('.*?'\)"
    changedPattern = "upper(upn) = upper('" + updatedUpn + "')"
    if securityFlag:
        if re.search(pattern, qry):
            return re.sub(pattern, changedPattern, qry)
        else:
            return sql_add_un_filter(qry,changedPattern)
    else:
        if re.search(pattern, qry):
           return re.sub(pattern, "1 = 1", qry)
        else:
           return qry


# This Function gets the query
def get_sql(quest):
    url = config.get('GenAISQLGenerator', 'genaisqlgenerator.url')
    data = {
    'question': quest
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error: {response.status_code}")
        return "Error fetching SQL"

# query clean up utility method - clean iLIKE & LIMIT
max_rows = config.get('QueryResult', 'max.resultset')
def clean_query(text):
    str = "".join(text.replace(r'\r', '  ').replace(r'\n', '  ').splitlines())
    cleanstr = str.replace(";","")
    cleanstr = cleanstr.replace('"','')
    cleanstr_nospaces = cleanstr.strip()
    # detect ilike
    pattern = r'(?i)([^\s]+)\s+ilike\s+' + r"'([^']+)'"
    # change text
    modified_str = re.sub(pattern, lambda match: f"UPPER({match.group(1)}) LIKE UPPER('{match.group(2)}')", cleanstr_nospaces)

    # Regular expression to find the pattern "LIMIT" followed by spaces and a number
    pattern2 = r"\sLIMIT\s+(\d+)\s*"
    match2 = re.search(pattern2, modified_str)

    if match2:
        modified_str = re.sub(pattern2, f" fetch first {match2.group(1)} rows only", modified_str)
    else:
        modified_str = modified_str.strip() + " fetch first " + max_rows + " rows only"

    modified_str = modified_str.replace("::FLOAT","")

    return modified_str

def remove_quotes(s):
    if s.startswith("'\"") and s.endswith("\"'"):
        return s[2:-2]
    return s

app = FastAPI()

@app.post("/")
async def create_item(request: Request):
    print("debug")
    print(request.url.query)
    j = await request.body()
    json_post_raw = await request.json()
    print(json_post_raw)
    json_post = json.dumps(json_post_raw)
    print(json_post)
    json_post_list = json.loads(json_post)
    question = json_post_list.get('question')
    oda_sessionid = json_post_list.get('sessionid')
    oda_userName = ""
    try:
        oda_userName = json_post_list.get('userName')
        if oda_userName is None:
            oda_userName = ""
    except Exception as e:
        oda_userName = ""

    print(question)
    print(oda_sessionid)
    print(oda_userName)
    q = converse_with_llm_oci(question,oda_sessionid)
    print(q)
    response = query(q,oda_userName)
    return response

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
