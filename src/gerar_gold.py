import pandas as pd
import joblib
import os
import holidays
from pathlib import Path

def gerar_camada_gold():
    print("Iniciando o pipeline da Camada Gold...")
    caminho_script = Path(__file__).parent 
    raiz_projeto = caminho_script.parent
    
    caminho_silver = raiz_projeto / "data" / "silver" / "surf_silver_data.csv"
    caminho_gold = raiz_projeto / "data" / "gold" / "surf_gold_data.csv"
    pasta_modelos = raiz_projeto / "modelos"

    if not caminho_silver.exists():
        print(f"Arquivo Silver não encontrado em: {caminho_silver}")
        return

    print("Carregando dados e artefatos de ML...")
    df = pd.read_csv(caminho_silver)
    df['data_hora'] = pd.to_datetime(df['data_hora'])
    
    # Carregando da pasta /models que definimos no treino
    modelo = joblib.load(pasta_modelos / 'modelo_surf.pkl')
    le_pico = joblib.load(pasta_modelos / 'encoder_pico.pkl')
    le_vento = joblib.load(pasta_modelos / 'encoder_vento.pkl')

    print("Traduzindo dados e gerando predições...")
    # Tradução dos nomes para IDs usando os encoders
    df['pico_id'] = le_pico.transform(df['pico_nome'])
    df['vento_id'] = le_vento.transform(df['vento_tipo'])
    features = ['tamanho_onda', 'periodo_onda', 'direcao_onda', 'velocidade_vento', 
                'direcao_vento', 'pico_id', 'vento_id', 'potencia_onda', 'mes', 'hora']
    
    df['predicao_modelo'] = modelo.predict(df[features])

    # Lógica de Aproveitamento
    print("Analisando feriados nacionais e estaduais...")
    feriados_nacionais = holidays.BR()
    feriados_estaduais = {
        'SP': holidays.BR(state='SP'),
        'RJ': holidays.BR(state='RJ'),
        'PE': holidays.BR(state='PE'),
        'SC': holidays.BR(state='SC'),
        'RN': holidays.BR(state='RN'),
        'BA': holidays.BR(state='BA')
    }

    def calcular_aproveitamento(row):
        # Ignora mar sem condição mínima
        if "Escola" in row['predicao_modelo']:
            return "Sem Surf (Muito Pequeno)"        
        estado_pico = row['pico_nome'].split('-')[-1]        
        # Verifica Feriado ou Fim de Semana
        final_semana = row['dia_semana'] >= 5
        
        # Verifica feriado Nacional OU Estadual
        feriados_locais = feriados_estaduais.get(estado_pico, feriados_nacionais)
        feriados_locais = row['data_hora'] in feriados_nacionais or row['data_hora'] in feriados_locais
        
        praia_cheia = final_semana or feriados_locais
        
        # Define a qualidade baseada na predição do modelo
        condicao_top = "Glassy" in row['predicao_modelo'] or "Performance" in row['predicao_modelo']
        
        if condicao_top:
            if not praia_cheia:
                return "DIA DE OURO (Top + Vazio)"
            else:
                return "Condicao Top com Feriado"
        
        return "Condicao Ok + Vazia" if not praia_cheia else "Condicao Ok + Cheia"

    # Aplica a lógica de aproveitamento
    df['indice_aproveitamento'] = df.apply(calcular_aproveitamento, axis=1)

    # Salvar o resultado na camada Gold
    print("Salvando resultado final...")
    caminho_gold.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(caminho_gold, index=False)
    
    print(f"Camada Gold gerada com sucesso!")
    print(f"Local: {caminho_gold}")
    print(f"Total de previsões realizadas: {len(df)}")

# Executar
if __name__ == "__main__":
    gerar_camada_gold()