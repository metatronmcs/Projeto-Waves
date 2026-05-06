import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

picos = [
    {"nome": "Itauna-RJ", "lat": -22.93, "lon": -42.49, "orientacao": 170},
    {"nome": "Maresias-SP", "lat": -23.79, "lon": -45.55, "orientacao": 180},
    {"nome": "Cacimba-PE", "lat": -3.85, "lon": -32.44, "orientacao": 320},
    {"nome": "Itamambuca-SP", "lat": -23.40, "lon": -45.00, "orientacao": 140},
    {"nome": "Joaquina-SC", "lat": -27.63, "lon": -48.44, "orientacao": 90},
    {"nome": "Guarda-SC", "lat": -27.90, "lon": -48.58, "orientacao": 100},
    {"nome": "Rosa-SC", "lat": -28.13, "lon": -48.64, "orientacao": 110},
    {"nome": "BaiaFormosa-RN", "lat": -6.37, "lon": -35.00, "orientacao": 70},
    {"nome": "Itacarezinho-BA", "lat": -14.33, "lon": -38.93, "orientacao": 100},
    {"nome": "Arpoador-RJ", "lat": -22.99, "lon": -43.19, "orientacao": 160}
]

DATA_INICIO_PROJETO = "2024-01-01"
OUTPUT_DIR = 'data'
FILE_PATH = os.path.join(OUTPUT_DIR, 'surf_raw_data.csv')

def pipeline_ingestao_surf(lista_picos, start_date, end_date):
    df_final = pd.DataFrame()
    
    dt_start = datetime.strptime(start_date, "%Y-%m-%d").date()
    hoje = datetime.today().date()
    
    # Seleção dinâmica de API (Arquivos e Forecast)
    if dt_start < (hoje - timedelta(days=120)):
        # Para dados históricos, usamos o arquivo de vento e marine normal (que também tem histórico):
        url_marine = "https://marine-api.open-meteo.com/v1/marine" # Marine também lida com histórico
        url_weather = "https://archive-api.open-meteo.com/v1/archive" # Vento arquivo histórico
        api_mode = "HISTÓRICO"
    else:
        # Para dados recentes e futuros, usamos o forecast (que tem previsão de vento e ondas):
        url_marine = "https://marine-api.open-meteo.com/v1/marine"
        url_weather = "https://api.open-meteo.com/v1/forecast"
        api_mode = "FORECAST"
    
    print(f"   [Modo: {api_mode}] de {start_date} até {end_date}")

    for pico in lista_picos:
        params_base = {
            "latitude": pico['lat'], "longitude": pico['lon'],
            "start_date": start_date, "end_date": end_date,
            "timezone": "America/Sao_Paulo"
        }
        
        try:            
            res_m = requests.get(url_marine, params={**params_base, "hourly": ["wave_height", "wave_period", "wave_direction"]}).json()
            res_w = requests.get(url_weather, params={**params_base, "hourly": ["wind_speed_10m", "wind_direction_10m"]}).json()
            
            if 'hourly' not in res_m or 'hourly' not in res_w:
                continue

            df_m = pd.DataFrame(res_m['hourly'])
            df_w = pd.DataFrame(res_w['hourly'])
            df_temp = pd.merge(df_m, df_w, on="time")
            
            # Renomeação das colunas
            df_temp = df_temp.rename(columns={
                "time": "data_hora", "wave_height": "tamanho_onda",
                "wave_period": "periodo_onda", "wave_direction": "direcao_onda",
                "wind_speed_10m": "velocidade_vento", "wind_direction_10m": "direcao_vento"
            })
            
            df_temp['pico_nome'] = pico['nome']
            df_temp['pico_orientacao'] = pico['orientacao']
            df_temp['lat'] = pico['lat']
            df_temp['lon'] = pico['lon']
            
            df_final = pd.concat([df_final, df_temp], ignore_index=True)
            time.sleep(0.4)
            
        except Exception as e:
            print(f"Erro em {pico['nome']}: {e}")
            
    return df_final

def salvar_dados_robusto(df_novo):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    if os.path.exists(FILE_PATH):
        df_antigo = pd.read_csv(FILE_PATH)
        df_total = pd.concat([df_antigo, df_novo])
        df_total['data_hora'] = pd.to_datetime(df_total['data_hora'], format='mixed')
        
        df_total = df_total.drop_duplicates(subset=['data_hora', 'pico_nome'], keep='last')
        df_total = df_total.sort_values(by=['data_hora', 'pico_nome'])
        
        df_total.to_csv(FILE_PATH, index=False, encoding='utf-8')
    else:
        df_novo['data_hora'] = pd.to_datetime(df_novo['data_hora'], format='mixed')
        df_novo.to_csv(FILE_PATH, index=False, encoding='utf-8')

print("Iniciando Ingestão de Dados de Surf...")

# 1. Verifica progresso anterior
if os.path.exists(FILE_PATH):
    df_check = pd.read_csv(FILE_PATH)
    ultima_data_str = pd.to_datetime(df_check['data_hora']).max()
    cursor_data = ultima_data_str.date() + timedelta(days=1)
    print(f"Retomando do dia: {cursor_data}")
else:
    cursor_data = datetime.strptime(DATA_INICIO_PROJETO, "%Y-%m-%d").date()
    print(f"Criando nova base histórica desde: {cursor_data}")

# 2. Loop Mensal
data_final_forecast = (datetime.today() + timedelta(days=7)).date()

while cursor_data <= data_final_forecast:
    proximo_mes = cursor_data + relativedelta(months=1)
    fim_do_bloco = proximo_mes - timedelta(days=1)
    
    if fim_do_bloco > data_final_forecast:
        fim_do_bloco = data_final_forecast
        
    s_str = cursor_data.strftime("%Y-%m-%d")
    e_str = fim_do_bloco.strftime("%Y-%m-%d")
    
    print(f"Processando bloco: {s_str} até {e_str}")
    
    df_bloco = pipeline_ingestao_surf(picos, s_str, e_str)
    
    if not df_bloco.empty:
        salvar_dados_robusto(df_bloco)
        print(f"Bloco concluído e salvo.")
    
    cursor_data = proximo_mes
    time.sleep(1)

print(f"Processo Finalizado! Dados em: {os.path.abspath(FILE_PATH)}")