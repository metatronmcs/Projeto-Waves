import pandas as pd
import numpy as np
import os
from pathlib import Path

from ingestão_raw import salvar_dados_robusto

def classificar_nivel_surf(row):
    # Logica para mar calmo (Ondas até 1.0m)
    if row['tamanho_onda'] <= 1.0:
        if row['potencia_onda'] < 10:
            return 'Iniciante - Mar Calmo (Escola)'
        elif row['vento_tipo'] == 'Terral_Offshore':
            return 'Iniciante - Mar Calmo (Glassy)'
        else:
            return 'Iniciante - Mar Calmo (Regular)'
    # Logica para mar agitado (ondas acima de 1.0m)
    else:
        if row['tamanho_onda'] > 2.0 or row['potencia_onda'] > 30:
            return 'Avancado - Mar Pesado (Desafiador)'
        elif row['vento_tipo'] == 'Maral_Onshore':
            return 'Avancado - Mar Agitado (Choppy)'            
        else:
            return 'Avancado - Mar Agitado (Performance)'

def gerar_camada_silver():
    caminho_script = Path(__file__).parent 
    raiz_projeto = caminho_script.parent
    caminho_bronze = raiz_projeto / "data" / "bronze" / "surf_raw_data.csv"
    caminho_silver = raiz_projeto / "data" / "silver" / "surf_silver_data.csv"

    print(f"Iniciando Processamento de Dados para Camada Silver...")
    print(f"Procurando bronze em: {caminho_bronze}")

    if not caminho_bronze.exists():
        print(f"Erro: O arquivo não existe nesse caminho!")
        return
        
    df = pd.read_csv(caminho_bronze)
    df['data_hora'] = pd.to_datetime(df['data_hora'], format='mixed')

    # Lógica de Vento
    def classificar_vento(row):
        diff = abs(row['direcao_vento'] - row['pico_orientacao']) % 360
        if diff > 180: diff = 360 - diff
        if diff >= 135: return 'Terral_Offshore'
        elif diff <= 45: return 'Maral_Onshore'
        else: return 'Lado_Sideshore'

    df['vento_tipo'] = df.apply(classificar_vento, axis=1)

    # Engenharia das variáveis
    df['potencia_onda'] = df['tamanho_onda'] * df['periodo_onda']
    df['mes'] = df['data_hora'].dt.month
    df['hora'] = df['data_hora'].dt.hour
    df['dia_semana'] = df['data_hora'].dt.dayofweek
    
    # Classificação de Nível de Surf
    df['target_regra_manual'] = df.apply(classificar_nivel_surf, axis=1)

    # Arredondamento
    df = df.round({'tamanho_onda': 2, 'periodo_onda': 2, 'velocidade_vento': 2, 'potencia_onda': 2})

    # Garante que a pasta silver exista e salva
    caminho_silver.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho_silver, index=False)
    
    print(f"Sucesso! Arquivo salvo em: {caminho_silver}")
    print(f"Total de linhas processadas: {len(df)}")

if __name__ == "__main__":
    gerar_camada_silver()